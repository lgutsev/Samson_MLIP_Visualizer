"""Backend-independent ASE evaluation and relaxation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

ProgressCallback = Callable[[int, float, float, np.ndarray], None]
StopCallback = Callable[[], bool]


@dataclass(frozen=True)
class Evaluation:
    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    max_force_ev_per_angstrom: float


@dataclass(frozen=True)
class RelaxationResult:
    evaluation: Evaluation
    steps: int
    converged: bool
    stopped: bool


class _StopRelaxation(Exception):
    pass


def _max_force(forces: np.ndarray) -> float:
    return float(np.linalg.norm(forces, axis=1).max(initial=0.0))


def evaluate(atoms: Atoms) -> Evaluation:
    """Return potential energy and constraint-aware forces in ASE units."""
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return Evaluation(energy, forces, _max_force(forces))


def relax(
    atoms: Atoms,
    *,
    fmax: float = 0.05,
    max_steps: int = 250,
    max_step: float = 0.1,
    trajectory: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
) -> RelaxationResult:
    """Relax positions with ASE FIRE and report each accepted geometry.

    Cell vectors are intentionally not optimized: this is the safe default for
    surfaces, vacuum slabs, and passivated clusters embedded in a periodic cell.
    """
    if fmax <= 0:
        raise ValueError("fmax must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    optimizer = FIRE(
        atoms,
        maxstep=max_step,
        trajectory=str(trajectory) if trajectory else None,
        logfile=None,
    )
    stopped = False

    def report() -> None:
        if should_stop and should_stop():
            raise _StopRelaxation
        current = evaluate(atoms)
        if on_progress:
            on_progress(
                optimizer.get_number_of_steps(),
                current.energy_ev,
                current.max_force_ev_per_angstrom,
                atoms.get_positions().copy(),
            )

    optimizer.attach(report, interval=1)
    try:
        optimizer.run(fmax=fmax, steps=max_steps)
    except _StopRelaxation:
        stopped = True

    final = evaluate(atoms)
    return RelaxationResult(
        evaluation=final,
        steps=optimizer.get_number_of_steps(),
        converged=final.max_force_ev_per_angstrom <= fmax,
        stopped=stopped,
    )
