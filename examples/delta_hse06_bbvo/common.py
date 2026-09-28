"""Shared pieces of the HSE06 Δ-learning example for Ba₂BiVO₆ (BBVO).

The model is MACE-MP-0 small plus a correction trained on HSE06 − MACE-MP-0:
MACE-MP-0 was trained on Materials Project PBE+U data (U = 3.25 eV on V in
oxides), so it plays the role of the PBE+U calculation, and no DFT runs at
simulation time. Both PBE+U and HSE06 are computed on every labeled frame, so
the residual can be split into HSE06 − PBE+U (the functional) and PBE+U −
MACE-MP-0 (the baseline's own error).

Paths come from environment variables:

- ``DELTA_DIR``: where frames, packages, models, and results go (default:
  ``D:\\MLIP_Work_Folder\\delta_hse06_bbvo`` when D: exists, else this folder);
- ``MACE_FOUNDATION``: the baseline (default: MACE-MP-0 small in the MACE folder
  of ``samson_mlip_visualizer.paths``).
"""

import math
import os
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from samson_mlip_visualizer.paths import foundation_model

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
_DEFAULT = Path(r"D:\MLIP_Work_Folder\delta_hse06_bbvo")
WORK = Path(os.environ.get("DELTA_DIR", _DEFAULT if _DEFAULT.anchor and
                           Path(_DEFAULT.anchor).exists() else HERE))
FOUNDATION = os.environ.get("MACE_FOUNDATION", str(foundation_model()))
KSPACING = 0.30  # Å⁻¹: 5×5×5 on the 10-atom cell (the mesh of the existing HSE06 run), 3×3×3 on 40
# PBE+U (VASP, 520 eV, U(V) = 3.25 eV) primitive cell, fcc lattice, rock-salt B-site order.
A_PRIM = 4.2436937690695196  # Å; the conventional cubic cell is 2 × this = 8.4874 Å
U_O, V_O = 0.7258100020151517, 0.2741899979848483
STEPS_PER_MODEL = 2640
BATCH = 4
VALID_FRACTION = 0.1


def epochs(n, steps=STEPS_PER_MODEL):
    return max(1, round(steps / math.ceil(n * (1 - VALID_FRACTION) / BATCH)))


def primitive(scale=1.0):
    """Ba₂BiVO₆, 10 atoms, at the PBE+U geometry (lattice scaled by ``scale``).
    Atoms are grouped by species (Ba, V, Bi, O), as VASP needs them."""
    a = A_PRIM * scale
    frac = [[0.25] * 3, [0.75] * 3, [0.5] * 3, [0.0] * 3,
            [U_O, V_O, V_O], [V_O, U_O, U_O], [V_O, U_O, V_O], [U_O, V_O, U_O],
            [V_O, V_O, U_O], [U_O, U_O, V_O]]
    return Atoms("Ba2VBiO6", scaled_positions=frac, cell=[[0, a, a], [a, 0, a], [a, a, 0]],
                 pbc=True)


def conventional(scale=1.0):
    """The 40-atom cubic cell (a = 8.49 Å), atoms grouped by species."""
    cell = make_supercell(primitive(scale), np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]))
    return grouped(cell)


def grouped(atoms):
    """Atoms reordered so each species is contiguous, in the order Ba, V/Nb/Ta, Bi, O."""
    rank = {"Ba": 0, "V": 1, "Nb": 2, "Ta": 3, "Bi": 4, "O": 5}
    order = sorted(range(len(atoms)), key=lambda i: rank[atoms[i].symbol])
    out = atoms[order]
    out.calc = None
    return out


def rigid_scaled_cells(frames):
    """The isotropically scaled, unrattled primitive cells among ``frames``, as
    (scale, frame) pairs sorted by scale: ions exactly at the PBE+U fractional
    positions (the rattled copies share their cells, so the cell alone does not
    tell them apart)."""
    ideal = primitive().get_scaled_positions()
    cells = {}
    for frame in frames:
        if (frame.info.get("group") == "strain" and len(frame) == 10
                and np.allclose(frame.cell.angles(), 60.0, atol=1e-6)
                and np.allclose(frame.get_scaled_positions(), ideal, atol=1e-6)):
            cells[round(frame.cell.lengths()[0] / (A_PRIM * np.sqrt(2)), 4)] = frame
    return sorted(cells.items())


def run_dir(dry):
    """Where a run's models and results go: the dry run or the real campaign."""
    return WORK / ("dry_run" if dry else "campaign")


def labels(dry):
    """Every labeled frame of a run: the pristine campaign and, if collected, the
    doped and dilute ones (``doped_campaign/``, ``dilute/``; for the dry run
    ``dry_run/labeled_doped.extxyz`` and ``dry_run/labeled_dilute.extxyz``)."""
    from ase.io import read

    files = ([WORK / "dry_run" / f"labeled{tag}.extxyz" for tag in ("", "_doped", "_dilute")]
             if dry else [WORK / "campaign" / "labeled.extxyz",
                          WORK / "doped_campaign" / "labeled.extxyz",
                          WORK / "dilute" / "labeled.extxyz"])
    return [frame for path in files if path.exists() for frame in read(path, ":")]


def held_out(frame):
    """Test frames: the 600 K pristine trajectory and the ``*_test`` frames (doped
    compositions and the dilute 80-atom cells)."""
    group = frame.info["group"]
    return group == "md600" or group.endswith("_test")


def mace_mp0(device="cuda"):
    from samson_mlip_visualizer.calculators import create_calculator

    return create_calculator("mace", FOUNDATION, device=device)
