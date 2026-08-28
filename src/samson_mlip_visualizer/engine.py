"""Backend-independent ASE evaluation and relaxation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from .sanity import check_sane

ProgressCallback = Callable[[int, float, float, np.ndarray], None]
StopCallback = Callable[[], bool]

_OPTIMIZERS = ("FIRE", "LBFGS", "BFGS", "PreconLBFGS")


@dataclass(frozen=True)
class Evaluation:
    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    max_force_ev_per_angstrom: float
    energy_std_ev: float | None = None
    max_force_std_ev_per_angstrom: float | None = None


@dataclass(frozen=True)
class RelaxationResult:
    evaluation: Evaluation
    steps: int
    converged: bool
    stopped: bool


class ModelUncertaintyError(RuntimeError):
    """Raised when committee spread exceeds the caller's allowed threshold."""


class _StopRelaxation(Exception):
    pass


def _max_force(forces: np.ndarray) -> float:
    return float(np.linalg.norm(forces, axis=1).max(initial=0.0))


def _committee_spread(calc) -> tuple[float | None, float | None]:
    """Energy std and max per-atom force std from a committee, if the calculator
    exposes one (MACE populates ``energy_comm`` / ``forces_comm``)."""
    results = getattr(calc, "results", None) or {}

    energy_std = None
    energy_comm = results.get("energy_comm")
    if energy_comm is not None:
        values = np.asarray(energy_comm, dtype=float).ravel()
        if values.size > 1:
            energy_std = float(values.std())

    force_std = None
    forces_comm = results.get("forces_comm")
    if forces_comm is not None:
        stack = np.asarray(forces_comm, dtype=float)
        if stack.ndim == 3 and stack.shape[0] > 1 and stack.shape[1]:
            per_atom = np.linalg.norm(stack.std(axis=0), axis=1)
            force_std = float(per_atom.max())

    return energy_std, force_std


def _make_driver(name: str, atoms: Atoms, *, maxstep: float, trajectory):
    traj = str(trajectory) if trajectory else None
    if name == "FIRE":
        from ase.optimize import FIRE

        return FIRE(atoms, maxstep=maxstep, trajectory=traj, logfile=None)
    if name == "LBFGS":
        from ase.optimize import LBFGS

        return LBFGS(atoms, maxstep=maxstep, trajectory=traj, logfile=None)
    if name == "BFGS":
        from ase.optimize import BFGS

        return BFGS(atoms, maxstep=maxstep, trajectory=traj, logfile=None)
    if name == "PreconLBFGS":
        try:
            from ase.optimize.precon import PreconLBFGS
        except ImportError as exc:  # pragma: no cover - depends on ASE build
            raise ValueError("PreconLBFGS is unavailable in this ASE build") from exc

        return PreconLBFGS(
            atoms, maxstep=maxstep, trajectory=traj, logfile=None, precon="auto"
        )
    raise ValueError(f"Unknown optimizer {name!r}; choose one of {', '.join(_OPTIMIZERS)}")


def evaluate(atoms: Atoms) -> Evaluation:
    """Return potential energy and constraint-aware forces in ASE units."""
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    energy_std, force_std = _committee_spread(atoms.calc)
    return Evaluation(energy, forces, _max_force(forces), energy_std, force_std)


def relax(
    atoms: Atoms,
    *,
    fmax: float = 0.05,
    max_steps: int = 250,
    max_step: float = 0.1,
    optimizer: str = "FIRE",
    min_distance: float | None = 0.5,
    max_drift: float | None = None,
    max_force_std: float | None = None,
    trajectory: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
) -> RelaxationResult:
    """Relax positions with an ASE local optimizer and report each geometry.

    Cell vectors are intentionally not optimized: this is the safe default for
    surfaces, vacuum slabs, and passivated clusters embedded in a periodic cell.

    ``min_distance`` / ``max_drift`` abort the run (``StructureSanityError``) when
    the geometry collapses or runs away. ``max_force_std`` aborts
    (``ModelUncertaintyError``) when a model committee's force spread at a step
    exceeds the threshold, i.e. the model is extrapolating.
    """
    if fmax <= 0:
        raise ValueError("fmax must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    driver = _make_driver(optimizer, atoms, maxstep=max_step, trajectory=trajectory)
    start_positions = atoms.get_positions().copy()
    stopped = False

    def report() -> None:
        if should_stop and should_stop():
            raise _StopRelaxation
        check_sane(
            atoms,
            min_distance=min_distance,
            reference_positions=start_positions,
            max_drift=max_drift,
        )
        current = evaluate(atoms)
        if (
            max_force_std is not None
            and current.max_force_std_ev_per_angstrom is not None
            and current.max_force_std_ev_per_angstrom > max_force_std
        ):
            raise ModelUncertaintyError(
                f"Committee force spread {current.max_force_std_ev_per_angstrom:.3f} eV/Å "
                f"exceeds the {max_force_std:.3f} eV/Å limit; the model is extrapolating "
                "for this geometry."
            )
        if on_progress:
            on_progress(
                driver.get_number_of_steps(),
                current.energy_ev,
                current.max_force_ev_per_angstrom,
                atoms.get_positions().copy(),
            )

    driver.attach(report, interval=1)
    try:
        driver.run(fmax=fmax, steps=max_steps)
    except _StopRelaxation:
        stopped = True

    final = evaluate(atoms)
    return RelaxationResult(
        evaluation=final,
        steps=driver.get_number_of_steps(),
        converged=final.max_force_ev_per_angstrom <= fmax,
        stopped=stopped,
    )
