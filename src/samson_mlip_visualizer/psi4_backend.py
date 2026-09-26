"""DFT (and other quantum chemistry) through Psi4, as an ASE calculator.

The reference level the foundation MLIPs were trained on is PBE, so a PBE
calculation on the same geometry says directly whether a surprising MLIP number
is the model or the functional. Psi4 runs in its own environment, in a
long-lived worker process (:mod:`.psi4_worker`), so SAMSON's Python (numpy 1.24)
never imports it and Psi4 is imported once, not per call. Install it once::

    micromamba create -n qm -c conda-forge python=3.11 psi4

then choose that environment's ``python`` executable as the model file. Any
method Psi4 has gradients for works (``pbe``, ``pbe0``, ``b3lyp-d3bj``, ``hf``,
``mp2``...); cost grows steeply with system size, so this is for molecules and
small clusters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree

from .worker_process import WorkerCalculator

WORKER = Path(__file__).with_name("psi4_worker.py")
# The def2 basis sets cover H through Rn; for other basis sets the element list
# is not known here, and the element guard says so instead of guessing.
_DEF2_ELEMENTS = frozenset(chemical_symbols[1:87])


def _site_packages(python: Path) -> list[Path]:
    root = python.parent
    return [root / "Lib" / "site-packages", *sorted(root.parent.glob("lib/python3*/site-packages"))]


def has_psi4(python: str | Path) -> bool:
    """Whether ``python`` is an interpreter whose environment contains Psi4."""
    python = Path(python)
    return python.is_file() and any((site / "psi4").is_dir() for site in _site_packages(python))


def psi4_version(python: str | Path) -> str | None:
    """Psi4's version from its metadata file, without importing it."""
    for site in _site_packages(Path(python)):
        metadata = site / "psi4" / "metadata.py"
        if metadata.is_file():
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return None


def find_psi4() -> Path | None:
    """A Python with Psi4: ``$PSI4_PYTHON``, then common conda/micromamba envs."""
    configured = os.environ.get("PSI4_PYTHON")
    if configured and has_psi4(configured):
        return Path(configured)
    home = Path.home()
    name = "python.exe" if sys.platform == "win32" else "bin/python"
    for root in ("micromamba", "miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        for env in sorted((home / root / "envs").glob("*")):
            if has_psi4(env / name):
                return env / name
    return None


class Psi4Calculator(WorkerCalculator):
    """Energies and forces from Psi4 (``psi4.gradient``) in a worker process.

    ``method`` and ``basis`` are Psi4 names (default PBE/def2-TZVP);
    ``multiplicity`` > 1 switches to an unrestricted reference; ``options`` are
    extra Psi4 options. Molecules only: periodic structures are refused.
    """

    label_name = "Psi4"

    def __init__(
        self,
        python: str | Path,
        *,
        method: str = "pbe",
        basis: str = "def2-tzvp",
        charge: int = 0,
        multiplicity: int = 1,
        threads: int | None = None,
        memory_mb: int = 2000,
        options: dict | None = None,
        timeout: float = 3600.0,
        **kwargs,
    ):
        super().__init__(python, timeout=timeout, **kwargs)
        if multiplicity < 1:
            raise ValueError("The multiplicity must be at least 1")
        if not method.strip() or not basis.strip():
            raise ValueError("Psi4 needs a method and a basis set")
        self.method = method.strip().lower()
        self.basis = basis.strip().lower()
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.threads = threads or max(1, min(8, (os.cpu_count() or 2) // 2))
        self.memory_mb = int(memory_mb)
        self.options = dict(options or {})
        self.supported_elements = _DEF2_ELEMENTS if self.basis.startswith("def2") else None

    def worker_script(self) -> Path:
        return WORKER

    def request(self, atoms) -> dict:
        return {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.positions.tolist(),
            "method": self.method,
            "basis": self.basis,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "threads": self.threads,
            "memory_mb": self.memory_mb,
            "options": self.options,
        }

    def results_from(self, answer: dict, atoms) -> dict:
        gradient = np.asarray(answer["gradient"], float).reshape(len(atoms), 3)
        energy = float(answer["energy"]) * Hartree
        return {"energy": energy, "free_energy": energy, "forces": -gradient * Hartree / Bohr}

    def describe_error(self, error: str) -> str:
        return f"Psi4 {self.method}/{self.basis}: {error}"
