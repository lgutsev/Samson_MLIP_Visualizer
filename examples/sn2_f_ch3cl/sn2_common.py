"""Shared pieces of the SN2 checks: paths, calculators, and a cached ωB97X-D reference.

Run the scripts that import this with SAMSON's Python (mace-torch, CUDA torch)
and the package on the path. Paths default to the run documented in the
README; ``SN2_WORK`` overrides the work folder.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

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


def geometry_key(numbers, positions, charge):
    """Identifies a structure to 1e-5 Å (rounding noise in files stays below that)."""
    rounded = np.round(np.asarray(positions, float), 5) + 0.0  # + 0.0 turns -0.0 into 0.0
    data = f"{charge}|{list(map(int, numbers))}|".encode() + rounded.tobytes()
    return hashlib.sha1(data).hexdigest()


class CachedReference(Calculator):
    """ωB97X-D (or any calculator from ``factory``), each structure computed once.

    Results are kept in a JSON file keyed by :func:`geometry_key` and written
    after every new calculation, so the same frame evaluated for several models,
    or by an interrupted run started again, costs nothing the second time.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, cache_file, factory=psi4_reference, charge=-1, **kwargs):
        super().__init__(**kwargs)
        self.cache_file = Path(cache_file)
        self.factory, self.charge = factory, charge
        self.cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        self.computed = 0
        self._inner = None

    def add(self, numbers, positions, energy, forces):
        key = geometry_key(numbers, positions, self.charge)
        self.cache[key] = {"energy_ev": float(energy), "forces": np.asarray(forces).tolist()}

    def save(self):
        # Merge with the file first: another script may have added entries since.
        if self.cache_file.exists():
            self.cache = {**json.loads(self.cache_file.read_text()), **self.cache}
        temporary = self.cache_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.cache))
        temporary.replace(self.cache_file)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        key = geometry_key(self.atoms.numbers, self.atoms.positions, self.charge)
        if key not in self.cache:
            if self._inner is None:
                self._inner = self.factory(self.charge)
            probe = self.atoms.copy()
            probe.calc = self._inner
            self.add(probe.numbers, probe.positions, probe.get_potential_energy(),
                     probe.get_forces())
            self.computed += 1
            self.save()
        entry = self.cache[key]
        energy = entry["energy_ev"]
        self.results = {"energy": energy, "free_energy": energy,
                        "forces": np.array(entry["forces"])}


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
