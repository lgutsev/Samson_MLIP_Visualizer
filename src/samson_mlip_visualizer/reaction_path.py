"""Reaction paths: IRC from a transition state, and QST2/QST3-style path searches.

:func:`irc` follows the intrinsic reaction coordinate (Fukui) downhill from a
saddle point in both directions: steepest descent in mass-weighted coordinates,
started along the imaginary mode, with the step shortened whenever the energy
would rise. It is the counterpart of Gaussian's ``IRC`` keyword and confirms
which minima a transition state connects.

:func:`qst` is the counterpart of Gaussian's ``Opt=QST2`` / ``QST3``: from a
reactant and a product (QST2), plus optionally a transition-state guess (QST3),
it builds a path by image-dependent pair-potential interpolation (IDPP; Smidstrup
et al. 2014), relaxes it as a climbing-image nudged elastic band (Henkelman et
al. 2000), and refines the highest image to a first-order saddle with P-RFO.
Reactant and product must contain the same atoms in the same order, and should
be relaxed minima.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

from .engine import relax
from .ts import TSResult, prfo_search
from .vibrations import harmonic_frequencies


class _Stop(Exception):
    pass


_INTERPOLATION = {"allow_shared_calculator": True, "method": "improvedtangent"}


@dataclass(frozen=True)
class PathFrame:
    positions: np.ndarray
    energy_ev: float
    arc: float  # mass-weighted distance from the transition state, Å·amu^½ (signed)


@dataclass
class IRCResult:
    ts_energy_ev: float
    forward: list[PathFrame] = field(default_factory=list)
    reverse: list[PathFrame] = field(default_factory=list)
    forward_minimum_ev: float | None = None
    reverse_minimum_ev: float | None = None
    stopped: bool = False

    def frames(self, ts_positions: np.ndarray) -> list[PathFrame]:
        """The whole path in order: reverse end → transition state → forward end."""
        ts = PathFrame(np.asarray(ts_positions), self.ts_energy_ev, 0.0)
        return [*reversed(self.reverse), ts, *self.forward]


def _imaginary_mode(atoms: Atoms) -> np.ndarray:
    frequencies = harmonic_frequencies(atoms)
    if frequencies.n_imaginary == 0:
        raise ValueError(
            "The structure has no imaginary mode, so it is not a transition state; "
            "run a TS search first."
        )
    mode = np.zeros((len(atoms), 3))
    mode[list(frequencies.free_indices)] = frequencies.modes[0]
    return mode


def irc(
    atoms: Atoms,
    *,
    step: float = 0.1,
    max_steps: int = 150,
    fmax: float = 0.02,
    mode: np.ndarray | None = None,
    relax_ends: bool = True,
    on_progress: Callable[[str, int, float, float, np.ndarray], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> IRCResult:
    """Follow the IRC from the transition state in ``atoms``, both directions.

    ``step`` is the arc length per step in mass-weighted coordinates (Å·amu^½).
    Each branch stops when the largest atomic force drops below ``fmax`` or after
    ``max_steps``. ``relax_ends`` then relaxes copies of both end points to report
    the minima energies (the path frames themselves are unchanged).
    ``on_progress(direction, step, energy, max_force, positions)``; ``direction``
    is ``"forward"`` (along the mode's sign) or ``"reverse"``. The atoms are left
    at the transition-state geometry.
    """
    if step <= 0 or max_steps < 1:
        raise ValueError("step must be positive and max_steps at least 1")
    ts_positions = atoms.get_positions().copy()
    ts_energy = float(atoms.get_potential_energy())
    direction_mode = _imaginary_mode(atoms) if mode is None else np.asarray(mode, float)
    sqrt_mass = np.sqrt(atoms.get_masses())[:, None]
    weighted = direction_mode * sqrt_mass
    weighted /= np.linalg.norm(weighted)
    result = IRCResult(ts_energy_ev=ts_energy)

    def forces_and_energy(positions):
        atoms.set_positions(positions)
        return atoms.get_forces(), float(atoms.get_potential_energy())

    try:
        for name, sign in (("forward", 1.0), ("reverse", -1.0)):
            frames = result.forward if name == "forward" else result.reverse
            positions = ts_positions + sign * step * weighted / sqrt_mass
            forces, energy = forces_and_energy(positions)
            arc = step
            frames.append(PathFrame(positions.copy(), energy, sign * arc))
            trial = step
            for index in range(1, max_steps):
                if should_stop and should_stop():
                    raise _Stop
                max_force = float(np.linalg.norm(forces, axis=1).max())
                if on_progress:
                    on_progress(name, index, energy, max_force, positions.copy())
                if max_force < fmax:
                    break
                gradient = -forces / sqrt_mass  # mass-weighted gradient
                norm = np.linalg.norm(gradient)
                if norm == 0:
                    break
                while True:
                    candidate = positions - trial * (gradient / norm) / sqrt_mass
                    new_forces, new_energy = forces_and_energy(candidate)
                    if new_energy <= energy or trial < step / 64:
                        break
                    trial /= 2
                if new_energy > energy:
                    break  # cannot descend further at this resolution
                arc += trial
                positions, forces, energy = candidate, new_forces, new_energy
                frames.append(PathFrame(positions.copy(), energy, sign * arc))
                trial = min(step, trial * 1.5)
    except _Stop:
        result.stopped = True
    finally:
        atoms.set_positions(ts_positions)

    if relax_ends and not result.stopped:
        for name in ("forward", "reverse"):
            frames = result.forward if name == "forward" else result.reverse
            end = atoms.copy()
            end.calc = atoms.calc
            end.set_positions(frames[-1].positions)
            relaxed = relax(end, fmax=min(fmax, 0.02), max_steps=500, optimizer="LBFGS")
            setattr(result, f"{name}_minimum_ev", relaxed.evaluation.energy_ev)
        atoms.set_positions(ts_positions)
        atoms.get_potential_energy()
    return result


@dataclass
class QSTResult:
    images: list[np.ndarray]
    energies_ev: list[float]
    highest_image: int
    barrier_forward_ev: float
    barrier_reverse_ev: float
    neb_converged: bool
    neb_steps: int
    stopped: bool = False
    ts: TSResult | None = None
    ts_positions: np.ndarray | None = None


def _check_endpoints(structures: Sequence[Atoms]) -> None:
    first = structures[0]
    for other in structures[1:]:
        if other.get_chemical_symbols() != first.get_chemical_symbols():
            raise ValueError(
                "Reactant, product (and guess) must contain the same atoms in the same "
                "order; renumber them before a QST search."
            )
        if (other.pbc != first.pbc).any() or not np.allclose(other.cell, first.cell):
            raise ValueError("Reactant, product (and guess) must share one cell and PBC")


def qst(
    reactant: Atoms,
    product: Atoms,
    calculator,
    *,
    guess: Atoms | None = None,
    images: int = 7,
    fmax: float = 0.05,
    max_steps: int = 500,
    climb: bool = True,
    refine: bool = True,
    refine_fmax: float = 0.01,
    remove_rigid_motion: bool | None = None,
    refine_internal: bool | None = None,
    on_progress: Callable[[int, list[float], float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> QSTResult:
    """QST2 (``guess=None``) or QST3 path search between two minima.

    ``images`` interior images are interpolated (with ``guess`` as the middle
    image for QST3), relaxed as an NEB (climbing image after a first loose pass),
    and, with ``refine``, the highest image is refined by :func:`prfo_search`.
    ``remove_rigid_motion`` (default: for non-periodic systems) removes overall
    rotation and translation from the band; ``refine_internal`` is passed to
    :func:`prfo_search`. ``on_progress(step, energies, max_force)`` reports the band.
    """
    from ase.mep import NEB
    from ase.optimize import FIRE

    if images < 1:
        raise ValueError("images must be at least 1")
    structures = [reactant, product] if guess is None else [reactant, guess, product]
    _check_endpoints(structures)
    mic = bool(reactant.pbc.any())

    def image_of(source: Atoms) -> Atoms:
        copy = source.copy()
        copy.calc = calculator
        return copy

    if guess is None:
        band = [image_of(reactant)] + [image_of(reactant) for _ in range(images)]
        band.append(image_of(product))
        NEB(band, **_INTERPOLATION).interpolate(method="idpp", mic=mic)
    else:
        # The guess is one of the interior images, in the middle of the band.
        left = (images - 1) // 2
        right = images - 1 - left
        first = [image_of(reactant)] + [image_of(reactant) for _ in range(left)] + [image_of(guess)]
        second = [first[-1]] + [image_of(guess) for _ in range(right)] + [image_of(product)]
        NEB(first, **_INTERPOLATION).interpolate(method="idpp", mic=mic)
        NEB(second, **_INTERPOLATION).interpolate(method="idpp", mic=mic)
        band = first + second[1:]

    neb = NEB(
        band,
        climb=False,
        allow_shared_calculator=True,
        remove_rotation_and_translation=(
            not mic if remove_rigid_motion is None else remove_rigid_motion
        ),
        method="improvedtangent",
    )
    optimizer = FIRE(neb, logfile=None)
    state = {"step": 0}

    def energies() -> list[float]:
        return [float(image.get_potential_energy()) for image in band]

    def report() -> None:
        if should_stop and should_stop():
            raise _Stop
        if on_progress:
            forces = neb.get_forces().reshape(-1, 3)
            on_progress(state["step"], energies(), float(np.linalg.norm(forces, axis=1).max()))
        state["step"] += 1

    optimizer.attach(report, interval=1)
    stopped = converged = False
    try:
        # A loose plain-NEB pass first, so the climbing image climbs a sensible band.
        converged = bool(optimizer.run(fmax=max(fmax, 0.5) if climb else fmax, steps=max_steps))
        if climb:
            neb.climb = True
            remaining = max(1, max_steps - optimizer.nsteps)
            converged = bool(optimizer.run(fmax=fmax, steps=remaining))
    except _Stop:
        stopped = True

    path_energies = energies()
    highest = int(np.argmax(path_energies[1:-1])) + 1
    result = QSTResult(
        images=[image.get_positions().copy() for image in band],
        energies_ev=path_energies,
        highest_image=highest,
        barrier_forward_ev=path_energies[highest] - path_energies[0],
        barrier_reverse_ev=path_energies[highest] - path_energies[-1],
        neb_converged=converged,
        neb_steps=state["step"],
        stopped=stopped,
    )
    if refine and not stopped:
        candidate = image_of(band[highest])
        result.ts = prfo_search(
            candidate,
            fmax=refine_fmax,
            max_steps=max_steps,
            internal=refine_internal,
            should_stop=should_stop,
        )
        result.ts_positions = candidate.get_positions().copy()
        result.barrier_forward_ev = result.ts.evaluation.energy_ev - path_energies[0]
        result.barrier_reverse_ev = result.ts.evaluation.energy_ev - path_energies[-1]
    return result
