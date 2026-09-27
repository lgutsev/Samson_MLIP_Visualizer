"""Shared pieces of the Δ-learning experiment for HCN ⇌ HNC: GFN-xTB + a MACE
correction to PBE/def2-TZVP, against direct fine-tuning of MACE-MP-0.

Paths come from environment variables:

- ``DELTA_DIR``: where data, models, and results go (default: this folder);
- ``MACE_FOUNDATION``: the foundation model for the direct fine-tunes (default:
  MACE-MP-0 small in the MACE folder of ``samson_mlip_visualizer.paths``).
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
# The 87 PBE-labeled structures of the HCN fine-tune (docs/fine_tuning.md): 29 frames
# of the MACE-MP-0 IRC and 58 rattled copies. REF_energy is PBE shifted by a constant.
POOL = HERE.parent / "hpc_smoke_tests" / "hcn_training_87_structures.extxyz"
TEST_SET = WORK / "test_set.extxyz"
C, N, H = 0, 1, 2  # atom order in every structure here
PBE_BARRIER, PBE_REACTION = 1.997, 0.662  # Psi4 PBE/def2-TZVP, optimized minima and TS
BASELINES = ("gfn1", "gfn2")
# The step budget of the original fine-tune (120 epochs of 22 batches), for every model.
STEPS = 2640
BATCH = 4
VALID_FRACTION = 0.1


def epochs(n):
    """Epochs for STEPS optimizer steps on n structures (minus the validation share)."""
    return max(1, round(STEPS / math.ceil(n * (1 - VALID_FRACTION) / BATCH)))


def pbe():
    from samson_mlip_visualizer.psi4_backend import Psi4Calculator, find_psi4

    return Psi4Calculator(find_psi4(), method="pbe", basis="def2-tzvp")


def xtb(method):
    from samson_mlip_visualizer.xtb_backend import XTBCalculator, find_xtb

    return XTBCalculator(find_xtb(), method=method)


def hcn(r_ch=1.076, r_cn=1.158, angle=180.0):
    """H–C≡N with the given C–H and C–N lengths (Å) and H–C–N angle (degrees)."""
    t = np.radians(180.0 - angle)
    return Atoms("CNH", positions=[[0, 0, 0], [0, 0, r_cn],
                                   [0, r_ch * np.sin(t), -r_ch * np.cos(t)]])


def hnc(r_nh=0.996, r_cn=1.169):
    """Linear H–N≡C (atom order C, N, H)."""
    return Atoms("CNH", positions=[[0, 0, 0], [0, 0, r_cn], [0, 0, r_cn + r_nh]])


def ts_guess():
    """The MACE-MP-0 transition state, rounded: the starting point for every TS search."""
    t = np.radians(68.3)
    return Atoms(
        "CNH", positions=[[0, 0, 0], [0, 0, 1.204], [0, 1.208 * np.sin(t), 1.208 * np.cos(t)]]
    )
