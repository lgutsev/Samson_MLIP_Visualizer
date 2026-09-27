"""Shared pieces of the SN2 checks: paths, calculators, and a cached ωB97X-D reference.

Run the scripts that import this with SAMSON's Python (mace-torch, CUDA torch)
and the package on the path. Paths default to the run documented in the
README; ``SN2_WORK`` overrides the work folder.
"""

import os
from pathlib import Path

from ase.io import read

from samson_mlip_visualizer.reference_cache import CachedReference as LibraryCachedReference

WORK = Path(os.environ.get("SN2_WORK", r"D:\MLIP_Work_Folder\sn2_F_CH3Cl"))
MACE_DIR = Path(r"D:\MLIP_Work_Folder\cache\mace")
STOCK = MACE_DIR / "20231210mace128L0_energy_epoch249model"
TUNED_NAME = "SN2-F-CH3Cl_wB97XD-def2TZVPD_from-MACE-MP-0-small"
COMMITTEE = [MACE_DIR / "finetuned" / TUNED_NAME / f"{TUNED_NAME}_seed{s}.model" for s in (1, 2, 3)]
AIMNET_PYTHON = Path(r"D:\MLIP_Work_Folder\envs\mlip\python.exe")
AIMNET_MODEL = r"D:\MLIP_Work_Folder\cache\aimnet\aimnet2_wb97m_d3_0.pt"
REFERENCE = "ωB97X-D/def2-TZVPD"
C, CL, F = 0, 4, 5  # atom order of the [CH3FCl]- frames: C H H H Cl F
KCAL = 23.0605  # kcal/mol per eV


def mace(paths):
    from mace.calculators import MACECalculator

    paths = [str(p) for p in paths] if isinstance(paths, list | tuple) else str(paths)
    return MACECalculator(model_paths=paths, device="cuda", default_dtype="float64")


def stock():
    return mace(STOCK)


def committee():
    return mace(COMMITTEE)


def aimnet(charge=-1):
    from samson_mlip_visualizer.aimnet2_backend import AIMNet2Calculator

    return AIMNet2Calculator(AIMNET_PYTHON, model=AIMNET_MODEL, charge=charge)


def psi4_reference(charge=-1):
    from samson_mlip_visualizer.psi4_backend import Psi4Calculator, find_psi4

    return Psi4Calculator(find_psi4(), method="wb97x-d", basis="def2-tzvpd", charge=charge,
                          threads=8, memory_mb=1900)


class CachedReference(LibraryCachedReference):
    """The library's :class:`~samson_mlip_visualizer.reference_cache.CachedReference`,
    defaulting to ωB97X-D/def2-TZVPD at charge -1 for these scripts."""

    def __init__(self, cache_file, factory=psi4_reference, charge=-1, **kwargs):
        super().__init__(cache_file, factory, charge=charge, **kwargs)

def reference_cache():
    """The shared ωB97X-D cache, seeded with the fine-tune's training labels
    (``finetune/train.extxyz``: raw Psi4 energies and forces)."""
    cached = CachedReference(WORK / "wb97xd_cache.json")
    train = WORK / "finetune" / "train.extxyz"
    if train.exists():
        before = len(cached.cache)
        for image in read(train, ":"):
            cached.add(image.numbers, image.positions, image.info["raw_energy"],
                       image.arrays["REF_forces"])
        if len(cached.cache) != before:
            cached.save()
    return cached


def training_set():
    return read(WORK / "finetune" / "train.extxyz", ":")
