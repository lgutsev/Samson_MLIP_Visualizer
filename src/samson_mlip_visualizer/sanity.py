"""Cheap physical-plausibility checks for MLIP-driven geometry changes.

Machine-learned potentials have out-of-distribution "holes" where the predicted
forces are unphysical, and a local optimizer will happily walk into them:
atoms collapse onto each other, or a fragment drifts away. These checks let the
caller stop with a clear message instead of writing a broken geometry back to
SAMSON. See e.g. Fu et al., "Forces are not Enough" (arXiv:2210.07237) and the
"Overlapping Atoms" failure metric in MOFSimBench (npj Comput. Mater. 2025).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


class StructureSanityError(RuntimeError):
    """Raised when a geometry has become physically implausible."""


def close_contacts(atoms: Atoms, min_distance: float) -> list[tuple[int, int, float]]:
    """Return ``(i, j, distance)`` for atom pairs closer than ``min_distance`` (Å).

    Nearest pair first. Periodic images are honored through ASE's neighbor list.
    """
    if len(atoms) < 2 or min_distance <= 0:
        return []
    first, second, distance = neighbor_list("ijd", atoms, float(min_distance))
    pairs = {
        (int(min(a, b)), int(max(a, b)), round(float(d), 6))
        for a, b, d in zip(first, second, distance, strict=True)
    }
    return sorted(pairs, key=lambda item: item[2])


def max_displacement(atoms: Atoms, reference_positions: np.ndarray) -> float:
    """Largest per-atom displacement (Å) from a reference set of positions."""
    delta = atoms.get_positions() - np.asarray(reference_positions, dtype=float)
    if delta.size == 0:
        return 0.0
    return float(np.linalg.norm(delta, axis=1).max())


def check_sane(
    atoms: Atoms,
    *,
    min_distance: float | None = 0.5,
    reference_positions: np.ndarray | None = None,
    max_drift: float | None = None,
) -> None:
    """Raise :class:`StructureSanityError` when the geometry looks broken."""
    if min_distance is not None:
        contacts = close_contacts(atoms, min_distance)
        if contacts:
            i, j, distance = contacts[0]
            raise StructureSanityError(
                f"Atoms {i} and {j} are {distance:.3f} Å apart, below the "
                f"{min_distance:.3f} Å floor. The model is producing unphysical forces "
                "for this configuration."
            )
    if max_drift is not None and reference_positions is not None:
        drift = max_displacement(atoms, reference_positions)
        if drift > max_drift:
            raise StructureSanityError(
                f"An atom has moved {drift:.2f} Å from its starting position, past the "
                f"{max_drift:.2f} Å limit. Stopping before the geometry runs away."
            )
