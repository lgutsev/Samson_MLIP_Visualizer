"""Shared pieces of the Δ-learning example for Ni(CO)₄ → Ni(CO)₃ + CO.

A metal–ligand bond breaking, where semi-empirical methods and GGAs are weak.
Reference: PBE0/def2-TZVP (Psi4). Baselines for the corrections: GFN2-xTB,
GFN1-xTB, and MACE-MP-0 small (the last as a rehearsal of an MLIP baseline, the
design of the periodic HSE06 example).

Paths come from environment variables:

- ``DELTA_DIR``: where data, models, and results go (default: this folder);
- ``MACE_FOUNDATION``: the foundation model (default: MACE-MP-0 small in the
  MACE folder of ``samson_mlip_visualizer.paths``).
"""

import math
import os
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms

from samson_mlip_visualizer.paths import foundation_model

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
WORK = Path(os.environ.get("DELTA_DIR", HERE))
FOUNDATION = os.environ.get("MACE_FOUNDATION", str(foundation_model()))
POOL = WORK / "pool.extxyz"  # PBE0-labeled training candidates (make_data.py)
TEST_SET = WORK / "test_set.extxyz"  # PBE0-labeled held-out frames (make_data.py)
NI, C1, O1 = 0, 1, 2  # the CO that leaves: atoms 1 (C) and 2 (O)
BASELINES = ("gfn2", "gfn1", "mace")  # what the corrections sit on
STEPS = 2640  # optimizer steps per model, as in the HCN example
BATCH = 4
VALID_FRACTION = 0.1


def epochs(n):
    """Epochs for STEPS optimizer steps on n structures (minus the validation share)."""
    return max(1, round(STEPS / math.ceil(n * (1 - VALID_FRACTION) / BATCH)))


def pbe0():
    """The reference. Psi4 stays under 2 GB here: above that, its CC and some
    DFT paths report a spurious out-of-memory on Windows."""
    from samson_mlip_visualizer.psi4_backend import Psi4Calculator, find_psi4

    return Psi4Calculator(find_psi4(), method="pbe0", basis="def2-tzvp", memory_mb=1900)


def xtb(method):
    from samson_mlip_visualizer.xtb_backend import XTBCalculator, find_xtb

    return XTBCalculator(find_xtb(), method=method)


def mace_mp0(device="cuda"):
    from samson_mlip_visualizer.calculators import create_calculator

    return create_calculator("mace", FOUNDATION, device=device)


def baseline(name, device="cuda"):
    return mace_mp0(device) if name == "mace" else xtb(name)


def ni_co4(r_nic=1.84, r_co=1.15):
    """Tetrahedral Ni(CO)₄; the first CO (atoms 1, 2) lies along +x+y+z."""
    symbols, positions = ["Ni"], [[0.0, 0.0, 0.0]]
    for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        v = np.array(v, float) / np.sqrt(3)
        symbols += ["C", "O"]
        positions += [r_nic * v, (r_nic + r_co) * v]
    return Atoms(symbols, positions=positions)
