"""Single-ended transition-state search with ASE's dimer method.

The dimer method climbs from a guess geometry to the nearest first-order saddle
point using only forces: it rotates a short "dimer" to find the lowest-curvature
mode, then walks uphill along it and downhill along every other direction. It
needs a starting geometry near the transition state and a guess for the reaction
direction:

- ``pair``: stretch a bond that forms or breaks in the reaction;
- ``use_hessian``: the softest vibrational mode of the guess geometry (costs one
  finite-difference Hessian, so meant for molecules; the most robust choice for
  a free molecule, whose zero-curvature rotations otherwise attract the dimer);
- neither: a random displacement of the free atoms.

Always confirm the result with :func:`vibrations.harmonic_frequencies`: a
transition state has exactly one imaginary mode.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from .engine import Evaluation, evaluate
from .sanity import check_sane
from .vibrations import _free_indices, _rigid_body_basis, harmonic_frequencies


@dataclass(frozen=True)
class TSResult:
    evaluation: Evaluation
    curvature: float
    steps: int
    converged: bool
    stopped: bool


class _StopSearch(Exception):
    pass


class _Converged(Exception):
    pass


def initial_mode(
    atoms: Atoms,
    *,
    pair: tuple[int, int] | None = None,
    use_hessian: bool = False,
    magnitude: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Initial displacement along ``pair``, the softest Hessian mode, or at random.

    The returned array (Å, shape ``(n_atoms, 3)``) has norm ``magnitude``. When no
    atom is fixed, overall translation/rotation is removed from it.
    """
    if pair is not None and use_hessian:
        raise ValueError("Choose either an atom pair or the Hessian mode, not both")
    free = _free_indices(atoms)
    vector = np.zeros((len(atoms), 3))
    if use_hessian:
        frequencies = harmonic_frequencies(atoms)
        vector[list(frequencies.free_indices)] = frequencies.modes[0]
    elif pair is not None:
        i, j = pair
        if i == j or not (0 <= i < len(atoms) and 0 <= j < len(atoms)):
            raise ValueError(f"Invalid atom pair {i}-{j} for {len(atoms)} atoms")
        axis = atoms.get_distance(i, j, mic=True, vector=True)
        axis = axis / np.linalg.norm(axis)
        if i in free:
            vector[i] = -axis
        if j in free:
            vector[j] = axis
        if not vector.any():
            raise ValueError(f"Atoms {i} and {j} are both fixed; choose a movable pair")
    else:
        rng = np.random.default_rng(seed)
        vector[free] = rng.normal(size=(len(free), 3))
    if len(free) == len(atoms) and len(atoms) > 1:
        sqrt_mass = np.sqrt(atoms.get_masses())[:, None]
        basis = _rigid_body_basis(atoms)
        weighted = (vector * sqrt_mass).ravel()
        weighted -= basis @ (basis.T @ weighted)
        vector = weighted.reshape(-1, 3) / sqrt_mass
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("The initial direction is pure rigid-body motion; choose another")
    return vector * (magnitude / norm)


def dimer_search(
    atoms: Atoms,
    *,
    fmax: float = 0.05,
    max_steps: int = 500,
    pair: tuple[int, int] | None = None,
    use_hessian: bool = False,
    displacement: float = 0.05,
    seed: int | None = None,
    min_distance: float | None = 0.5,
    trajectory: str | Path | None = None,
    on_progress: Callable[[int, float, float, float, np.ndarray], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TSResult:
    """Search for a first-order saddle point near the current geometry.

    ``on_progress(step, energy, max_force, curvature, positions)`` is called after
    every dimer translation. ``max_force`` is the true (unmodified) maximum atomic
    force; convergence requires it below ``fmax``, as for a minimum. A negative
    ``curvature`` (eV/Å^2) means the dimer has found an uphill direction.
    """
    from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate

    if fmax <= 0:
        raise ValueError("fmax must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    free = set(_free_indices(atoms))
    mask = [index in free for index in range(len(atoms))]
    displacement_vector = initial_mode(
        atoms, pair=pair, use_hessian=use_hessian, magnitude=displacement, seed=seed
    )

    control = DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        mask=mask,
        logfile=None,
    )
    dimer_atoms = MinModeAtoms(atoms, control, random_seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dimer_atoms.displace(displacement_vector=displacement_vector, log=False)

    traj = str(trajectory) if trajectory else None
    translator = MinModeTranslate(dimer_atoms, logfile=None, trajectory=traj)
    state = {"step": 0}

    def report() -> None:
        if should_stop and should_stop():
            raise _StopSearch
        check_sane(atoms, min_distance=min_distance)
        current = evaluate(atoms)
        curvature = float(dimer_atoms.get_curvature())
        if on_progress:
            on_progress(
                state["step"],
                current.energy_ev,
                current.max_force_ev_per_angstrom,
                curvature,
                atoms.get_positions().copy(),
            )
        state["step"] += 1
        if state["step"] > 1 and current.max_force_ev_per_angstrom <= fmax and curvature < 0:
            raise _Converged

    translator.attach(report, interval=1)
    stopped = converged = False
    try:
        # The dimer's own test uses mode-inverted forces, whose per-atom maximum
        # differs from the true one; give it a stricter target so the true-force
        # check in ``report`` decides convergence.
        translator.run(fmax=0.5 * fmax, steps=max_steps)
    except _StopSearch:
        stopped = True
    except _Converged:
        converged = True

    final = evaluate(atoms)
    curvature = float(dimer_atoms.get_curvature())
    if not stopped:
        converged = final.max_force_ev_per_angstrom <= fmax and curvature < 0
    return TSResult(
        evaluation=final,
        curvature=curvature,
        steps=max(0, state["step"] - 1),
        converged=converged,
        stopped=stopped,
    )

