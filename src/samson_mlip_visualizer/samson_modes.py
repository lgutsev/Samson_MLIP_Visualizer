"""Show a vibrational mode in SAMSON: an animated path and displacement arrows.

The geometry comes from :mod:`modes`; this module turns it into SAMSON nodes.
Both are added as one undo step each and can be deleted in Document View.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .modes import mode_arrows, mode_frames
from .samson_bridge import SamsonStructure, sync_positions

# SAMSON's internal length unit is the picometre.
_PM_PER_ANGSTROM = 100.0


def _samson():
    from samson import SAMSON

    return SAMSON


def full_mode(structure: SamsonStructure, frequencies, index: int) -> np.ndarray:
    """Mode ``index`` as an ``(n_atoms, 3)`` array (zeros on fixed atoms)."""
    mode = np.zeros((len(structure.samson_atoms), 3))
    mode[list(frequencies.free_indices)] = frequencies.modes[index]
    return mode


def add_mode_path(
    structure: SamsonStructure,
    mode: np.ndarray,
    *,
    name: str,
    max_displacement: float = 0.3,
    frames: int = 24,
) -> Any:
    """Add a SAMSON path that oscillates the structure along ``mode``.

    Frame 0 is the current geometry, so stepping the path back to 0 restores it.
    """
    import samson

    SAMSON = _samson()
    x0 = structure.ase_atoms.get_positions()
    conformations = samson.SBNodeIndexer()
    with SAMSON.holding(f"Add {name}"):
        try:
            for index, positions in enumerate(
                mode_frames(x0, mode, max_displacement=max_displacement, frames=frames)
            ):
                sync_positions(structure, positions, samson=SAMSON, process_events=False)
                atoms = samson.SBNodeIndexer()
                for atom in structure.samson_atoms:
                    atoms.addNode(atom)
                conformations.addNode(samson.SBConformation(f"{name} frame {index}", atoms))
        finally:
            sync_positions(structure, x0, samson=SAMSON, process_events=False)
        path = samson.SBPath(name, conformations)
        path.create()
        SAMSON.getActiveDocument().addChild(path)
    return path


def add_mode_arrows(
    structure: SamsonStructure,
    mode: np.ndarray,
    *,
    name: str,
    max_length: float = 1.0,
    **style,
) -> Any:
    """Add a mesh of arrows showing each atom's displacement in ``mode``."""
    import samson

    SAMSON = _samson()
    mesh = mode_arrows(structure.ase_atoms.get_positions(), mode, max_length=max_length, **style)
    count = len(mesh.positions)
    surface = samson.SBSurface(
        len(mesh.triangles),
        count,
        mesh.triangles.ravel().tolist(),
        (mesh.positions * _PM_PER_ANGSTROM).ravel().tolist(),
        mesh.normals.ravel().tolist(),
        mesh.colors.ravel().tolist(),
        [0] * count,
        [0] * count,
        [0.0] * (2 * count),
    )
    node = samson.SBMesh([surface])
    node.name = name
    with SAMSON.holding(f"Add {name}"):
        node.create()
        SAMSON.getActiveDocument().addChild(node)
    return node


class PathPlayer:
    """Loop a SAMSON path by stepping it on a Qt timer (SAMSON's main thread)."""

    def __init__(self, interval_ms: int = 40):
        from PySide6 import QtCore

        self._timer = QtCore.QTimer()
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._advance)
        self.path = None

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def play(self, path: Any) -> None:
        self.stop()
        self.path = path
        self._timer.start()

    def stop(self) -> None:
        """Stop and return the structure to frame 0 (the original geometry)."""
        self._timer.stop()
        if self.path is not None:
            try:
                self.path.currentStep = 0
            except Exception:  # noqa: BLE001 - the path may have been deleted
                pass
        self.path = None

    def _advance(self) -> None:
        try:
            steps = int(self.path.numberOfSteps)
            self.path.currentStep = (int(self.path.currentStep) + 1) % steps
        except Exception:  # noqa: BLE001 - deleted path: stop quietly
            self._timer.stop()
            self.path = None
