"""Backend-independent molecular dynamics, including distance-constrained MD.

This drives ASE's integrators with the same guards as the relaxation loop:
close contacts and committee spread abort the run, and an optional temperature
ceiling catches the energy blow-up an MLIP produces when it leaves its training
distribution. Fixed atom-pair distances are held with RATTLE (ASE
``FixBondLengths``); the mean force along each constrained distance is
accumulated, which is the raw ingredient of a constrained-MD (blue-moon /
thermodynamic-integration) free-energy profile.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.constraints import FixAtoms, FixBondLengths, FixCom

from .engine import ModelUncertaintyError, _committee_spread
from .sanity import check_sane

ENSEMBLES = ("Langevin", "Bussi", "NoseHooverChain", "NVE")
# Integrators that apply RATTLE position *and* momentum corrections every step.
_CONSTRAINT_SAFE = ("Langevin", "Bussi", "NVE")


@dataclass(frozen=True)
class DistanceConstraint:
    """Hold the distance between atoms ``i`` and ``j`` fixed during MD.

    ``target`` (Å) moves the pair to that distance before the run; ``None`` keeps
    the starting distance.
    """

    i: int
    j: int
    target: float | None = None


@dataclass(frozen=True)
class ConstraintForce:
    i: int
    j: int
    distance: float
    mean_force_ev_per_angstrom: float
    std_ev_per_angstrom: float
    samples: int


@dataclass(frozen=True)
class MDFrame:
    step: int
    time_fs: float
    potential_ev: float
    kinetic_ev: float
    temperature_k: float
    positions: np.ndarray
    constraint_forces: tuple[float, ...] = ()

    @property
    def total_ev(self) -> float:
        return self.potential_ev + self.kinetic_ev


@dataclass(frozen=True)
class MDResult:
    steps: int
    time_fs: float
    stopped: bool
    mean_temperature_k: float
    energy_drift_mev_per_atom_ps: float | None
    constraint_forces: tuple[ConstraintForce, ...]


class _StopMD(Exception):
    pass


class TemperatureLimitError(RuntimeError):
    """Raised when the instantaneous temperature exceeds the caller's ceiling."""


def parse_pairs(text: str) -> list[DistanceConstraint]:
    """Parse ``"0-1, 4-7:1.5"`` into distance constraints (0-based atom indices).

    ``:value`` sets a target distance in Å.
    """
    constraints = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pair, _, target = chunk.partition(":")
        try:
            i, j = (int(part) for part in pair.replace(" ", "").split("-"))
        except ValueError as exc:
            raise ValueError(
                f"Cannot read atom pair {chunk!r}; use 0-based indices like '3-7' or '3-7:1.2'"
            ) from exc
        constraints.append(DistanceConstraint(i, j, float(target) if target else None))
    return constraints


def md_warnings(atoms: Atoms, *, timestep_fs: float, ensemble: str, dtype: str | None = None):
    """Human-readable cautions about MD settings; empty when nothing stands out."""
    messages = []
    has_hydrogen = "H" in atoms.get_chemical_symbols()
    if has_hydrogen and timestep_fs > 1.0:
        messages.append(
            f"A {timestep_fs:g} fs timestep with hydrogen present is likely unstable; "
            "0.5 fs is the usual choice for flexible X-H bonds."
        )
    elif timestep_fs > 2.0:
        messages.append(f"A {timestep_fs:g} fs timestep is large for all-atom MD.")
    if ensemble == "NVE" and dtype == "float32":
        messages.append(
            "NVE with float32 forces shows visible energy drift; use float64 when the "
            "drift figure matters."
        )
    return messages


def _fixed_indices(atoms: Atoms) -> set[int]:
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed.update(int(index) for index in constraint.get_indices())
    return fixed


def _apply_distance_constraints(atoms: Atoms, pairs: Sequence[DistanceConstraint]) -> None:
    fixed = _fixed_indices(atoms)
    n_atoms = len(atoms)
    for pair in pairs:
        if pair.i == pair.j or not (0 <= pair.i < n_atoms and 0 <= pair.j < n_atoms):
            raise ValueError(f"Invalid constrained pair {pair.i}-{pair.j} for {n_atoms} atoms")
        if pair.target is not None:
            if pair.target <= 0:
                raise ValueError("Constraint target distances must be positive")
            if pair.i in fixed and pair.j in fixed:
                raise ValueError(f"Atoms {pair.i} and {pair.j} are both fixed; cannot move them")
            # Move the free atom(s) so a fixed partner stays put.
            fix = 0.0 if pair.i in fixed else 1.0 if pair.j in fixed else 0.5
            atoms.set_distance(pair.i, pair.j, pair.target, fix=fix, mic=True)
    if pairs:
        atoms.constraints = [
            *atoms.constraints,
            FixBondLengths([(pair.i, pair.j) for pair in pairs]),
        ]


def _separating_forces(atoms: Atoms, pairs: Sequence[DistanceConstraint]) -> tuple[float, ...]:
    """Model force along each constrained pair (positive pushes the atoms apart).

    Uses the relative-coordinate force ``(m_i F_j - m_j F_i) / (m_i + m_j)``
    projected on the pair axis, from the unconstrained model forces.
    """
    if not pairs:
        return ()
    forces = atoms.get_forces(apply_constraint=False)
    masses = atoms.get_masses()
    values = []
    for pair in pairs:
        axis = atoms.get_distance(pair.i, pair.j, mic=True, vector=True)
        axis = axis / np.linalg.norm(axis)
        mi, mj = masses[pair.i], masses[pair.j]
        relative = (mi * forces[pair.j] - mj * forces[pair.i]) / (mi + mj)
        values.append(float(relative @ axis))
    return tuple(values)


def _make_integrator(
    ensemble: str,
    atoms: Atoms,
    *,
    timestep_fs: float,
    temperature_k: float,
    friction_per_fs: float,
    tdamp_fs: float,
    rng: np.random.Generator,
):
    timestep = timestep_fs * units.fs
    if ensemble == "Langevin":
        from ase.md.langevin import Langevin

        return Langevin(
            atoms,
            timestep,
            temperature_K=temperature_k,
            friction=friction_per_fs / units.fs,
            fixcm=False,
            rng=rng,
        )
    if ensemble == "Bussi":
        from ase.md.bussi import Bussi

        return Bussi(
            atoms, timestep, temperature_K=temperature_k, taut=tdamp_fs * units.fs, rng=rng
        )
    if ensemble == "NoseHooverChain":
        from ase.md.nose_hoover_chain import NoseHooverChainNVT

        return NoseHooverChainNVT(
            atoms, timestep, temperature_K=temperature_k, tdamp=tdamp_fs * units.fs
        )
    if ensemble == "NVE":
        from ase.md.verlet import VelocityVerlet

        return VelocityVerlet(atoms, timestep)
    raise ValueError(f"Unknown ensemble {ensemble!r}; choose one of {', '.join(ENSEMBLES)}")


def run_md(
    atoms: Atoms,
    *,
    ensemble: str = "Langevin",
    temperature_k: float = 300.0,
    timestep_fs: float = 0.5,
    steps: int = 1000,
    friction_per_fs: float = 0.01,
    tdamp_fs: float = 100.0,
    initialize_velocities: bool = True,
    seed: int | None = None,
    distance_constraints: Sequence[DistanceConstraint] = (),
    fix_com: bool | None = None,
    report_interval: int = 10,
    min_distance: float | None = 0.5,
    max_force_std: float | None = None,
    max_temperature_k: float | None = None,
    trajectory: str | Path | None = None,
    trajectory_interval: int = 10,
    on_progress: Callable[[MDFrame], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> MDResult:
    """Run MD on ``atoms`` (which must already carry a calculator).

    ``on_progress`` receives an :class:`MDFrame` every ``report_interval`` steps
    and after the last step; ``should_stop`` is polled every step. The atoms keep
    their final positions and momenta. Existing constraints (e.g. ``FixAtoms``
    from SAMSON) are respected; ``distance_constraints`` and ``fix_com`` are added.
    ``fix_com=None`` fixes the centre of mass for Langevin runs without fixed atoms.
    """
    if ensemble not in ENSEMBLES:
        raise ValueError(f"Unknown ensemble {ensemble!r}; choose one of {', '.join(ENSEMBLES)}")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if timestep_fs <= 0:
        raise ValueError("timestep_fs must be positive")
    if ensemble != "NVE" and temperature_k <= 0:
        raise ValueError("A thermostat needs a positive temperature")
    if distance_constraints and ensemble not in _CONSTRAINT_SAFE:
        raise ValueError(
            f"Distance constraints are supported with {', '.join(_CONSTRAINT_SAFE)}, "
            f"not {ensemble}."
        )
    report_interval = max(1, int(report_interval))

    pairs = list(distance_constraints)
    _apply_distance_constraints(atoms, pairs)
    has_fixed_atoms = bool(_fixed_indices(atoms))
    if fix_com is None:
        # Langevin noise otherwise makes the whole system drift; ASE recommends
        # FixCom over Langevin's own (inexact) fixcm correction.
        fix_com = ensemble == "Langevin" and not has_fixed_atoms
    if fix_com:
        if has_fixed_atoms:
            raise ValueError("Fixing the centre of mass is redundant with fixed atoms")
        atoms.constraints = [*atoms.constraints, FixCom()]

    rng = np.random.default_rng(seed)
    if initialize_velocities:
        from ase.md import velocitydistribution

        # thermalize_momenta replaces MaxwellBoltzmannDistribution from ASE 3.29.
        thermalize = getattr(velocitydistribution, "thermalize_momenta", None)
        if thermalize is None:
            thermalize = velocitydistribution.MaxwellBoltzmannDistribution
        thermalize(atoms, temperature_K=temperature_k, rng=rng)
        if not has_fixed_atoms:
            velocitydistribution.Stationary(atoms)
            if not atoms.pbc.any():
                # Linear molecules have a zero principal moment; ASE handles it but warns.
                with np.errstate(invalid="ignore", divide="ignore"):
                    velocitydistribution.ZeroRotation(atoms)
        # A small system's random draw, minus the removed momenta and constrained
        # degrees of freedom, can land far from the target; start exactly on it.
        current = atoms.get_temperature()
        if temperature_k > 0 and current > 0:
            atoms.set_momenta(atoms.get_momenta() * np.sqrt(temperature_k / current))

    integrator = _make_integrator(
        ensemble,
        atoms,
        timestep_fs=timestep_fs,
        temperature_k=temperature_k,
        friction_per_fs=friction_per_fs,
        tdamp_fs=tdamp_fs,
        rng=rng,
    )

    trajectory_path = Path(trajectory) if trajectory else None
    if trajectory_path is not None and trajectory_path.exists():
        trajectory_path.unlink()
    constraint_samples: list[tuple[float, ...]] = []
    temperatures: list[float] = []
    totals: list[tuple[float, float]] = []
    state = {"step": 0}

    def frame() -> MDFrame:
        step = state["step"]
        return MDFrame(
            step=step,
            time_fs=step * timestep_fs,
            potential_ev=float(atoms.get_potential_energy()),
            kinetic_ev=float(atoms.get_kinetic_energy()),
            temperature_k=float(atoms.get_temperature()),
            positions=atoms.get_positions().copy(),
            constraint_forces=_separating_forces(atoms, pairs),
        )

    def write_frame() -> None:
        from ase.io import write

        write(trajectory_path, atoms, append=trajectory_path.exists())

    def observe() -> None:
        step = state["step"]
        if step > 0 and should_stop and should_stop():
            raise _StopMD
        check_sane(atoms, min_distance=min_distance)
        if max_force_std is not None:
            _, force_std = _committee_spread(atoms.calc)
            if force_std is not None and force_std > max_force_std:
                raise ModelUncertaintyError(
                    f"Committee force spread {force_std:.3f} eV/Å exceeds the "
                    f"{max_force_std:.3f} eV/Å limit at MD step {step}; the model is "
                    "extrapolating."
                )
        temperature = float(atoms.get_temperature())
        if max_temperature_k is not None and temperature > max_temperature_k:
            raise TemperatureLimitError(
                f"Temperature reached {temperature:.0f} K at step {step}, above the "
                f"{max_temperature_k:.0f} K ceiling. The trajectory is likely unstable "
                "(timestep too large, or the model is outside its training data)."
            )
        temperatures.append(temperature)
        if pairs:
            constraint_samples.append(_separating_forces(atoms, pairs))
        if trajectory_path is not None and step % max(1, trajectory_interval) == 0:
            write_frame()
        if step % report_interval == 0 or step == steps:
            current = frame()
            totals.append((current.time_fs, current.total_ev))
            if on_progress:
                on_progress(current)
        state["step"] += 1

    integrator.attach(observe, interval=1)
    stopped = False
    try:
        integrator.run(steps)
    except _StopMD:
        stopped = True
    # The observer runs once before the first step and once after every step.
    completed = state["step"] if stopped else max(0, state["step"] - 1)

    drift = None
    if ensemble == "NVE" and len(totals) >= 2 and totals[-1][0] > totals[0][0]:
        (t0, e0), (t1, e1) = totals[0], totals[-1]
        drift = (e1 - e0) / len(atoms) / ((t1 - t0) / 1000.0) * 1000.0

    summaries = []
    if constraint_samples:
        samples = np.asarray(constraint_samples)
        for column, pair in enumerate(pairs):
            summaries.append(
                ConstraintForce(
                    i=pair.i,
                    j=pair.j,
                    distance=float(atoms.get_distance(pair.i, pair.j, mic=True)),
                    mean_force_ev_per_angstrom=float(samples[:, column].mean()),
                    std_ev_per_angstrom=float(samples[:, column].std()),
                    samples=int(samples.shape[0]),
                )
            )

    return MDResult(
        steps=completed,
        time_fs=completed * timestep_fs,
        stopped=stopped,
        mean_temperature_k=float(np.mean(temperatures)) if temperatures else 0.0,
        energy_drift_mev_per_atom_ps=drift,
        constraint_forces=tuple(summaries),
    )
