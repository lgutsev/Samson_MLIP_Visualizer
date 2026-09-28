"""UMA (Meta FAIR's Universal Models for Atoms, fairchem) as an ASE calculator.

UMA is one network with task heads trained on several datasets. Its ``omol`` task
(OMol25, ωB97M-V/def2-TZVPD) takes the molecular charge and spin multiplicity as
inputs, which makes it, like AIMNet2, an MLIP for ions and polar reactions, over a
far larger part of the periodic table. The other tasks (``omat`` inorganic
materials, ``oc20`` catalysis, ``odac`` MOFs, ``omc`` molecular crystals) take no
charge or spin.

fairchem needs a newer numpy and torch than SAMSON's Python allows, so it runs in
its own environment in a worker process (:mod:`.uma_worker`). Install it once::

    micromamba create -p <envs>/mlip -c conda-forge python=3.12 pip
    <envs>/mlip/python -m pip install fairchem-core

then choose that environment's ``python`` as the model file. ``model`` is a UMA
checkpoint: the path of a downloaded ``.pt`` (the facebook/UMA repository is gated;
accept its licence on Hugging Face first), or a fairchem name such as
``uma-s-1p1``, which fairchem downloads to its own cache on first use. The model
runs in float32.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .aimnet2_backend import _site_packages
from .worker_process import WorkerCalculator

WORKER = Path(__file__).with_name("uma_worker.py")
DEFAULT_TASK = "omol"
TASKS = ("omol", "omat", "oc20", "odac", "omc")


def has_fairchem(python: str | Path) -> bool:
    """Whether ``python`` is an interpreter whose environment contains fairchem."""
    python = Path(python)
    return python.is_file() and any(
        (site / "fairchem" / "core").is_dir() for site in _site_packages(python)
    )


def fairchem_version(python: str | Path) -> str | None:
    """fairchem-core's version from its dist-info folder, without importing it."""
    for site in _site_packages(Path(python)):
        for info in sorted(site.glob("fairchem_core-*.dist-info")):
            return info.name[len("fairchem_core-") : -len(".dist-info")]
    return None


def find_fairchem() -> Path | None:
    """A Python with fairchem: ``$FAIRCHEM_PYTHON``, then common conda/micromamba envs."""
    configured = os.environ.get("FAIRCHEM_PYTHON")
    if configured and has_fairchem(configured):
        return Path(configured)
    home = Path.home()
    name = "python.exe" if sys.platform == "win32" else "bin/python"
    for root in ("micromamba", "miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        for env in sorted((home / root / "envs").glob("*")):
            if has_fairchem(env / name):
                return env / name
    return None


class UMACalculator(WorkerCalculator):
    """Energies and forces from UMA in a worker process.

    ``model`` is a checkpoint path or fairchem name; ``task`` a UMA task head;
    ``charge`` and ``multiplicity`` are the molecule's (``omol`` only);
    ``device`` is ``cpu`` or ``cuda`` (the worker's torch decides whether CUDA
    works). Periodic cells are passed on for the materials tasks. Single atoms (a
    free F⁻) need the isolated-atom references: ``atom_refs``, or
    ``iso_atom_elem_refs.yaml`` beside the checkpoint or in ``references`` beside it.
    """

    label_name = "UMA"
    periodic = True  # the materials tasks (omat, oc20, odac, omc) take cells

    def __init__(
        self,
        python: str | Path,
        *,
        model: str | None = None,
        task: str = DEFAULT_TASK,
        charge: int = 0,
        multiplicity: int = 1,
        device: str = "cpu",
        atom_refs: str | Path | None = None,
        timeout: float = 900.0,
        **kwargs,
    ):
        super().__init__(python, timeout=timeout, **kwargs)
        if atom_refs is not None and not Path(atom_refs).expanduser().is_file():
            raise ValueError(f"UMA atom references file does not exist: {atom_refs}")
        self.atom_refs = str(Path(atom_refs).expanduser()) if atom_refs is not None else None
        model = (model or "").strip()
        if not model:
            raise ValueError(
                "Choose a UMA checkpoint: a .pt file such as uma-s-1p1.pt, or a fairchem "
                "name such as uma-s-1p1"
            )
        if model.lower().endswith(".pt") and not Path(model).expanduser().is_file():
            raise ValueError(f"UMA checkpoint does not exist: {model}")
        task = (task or DEFAULT_TASK).strip().lower()
        if task not in TASKS:
            raise ValueError(f"UMA task must be one of {', '.join(TASKS)}, not {task!r}")
        if multiplicity < 1:
            raise ValueError("The multiplicity must be at least 1")
        if task != "omol" and (charge != 0 or multiplicity != 1):
            raise ValueError(f"Charge and multiplicity apply to the omol task, not {task}")
        self.model = str(Path(model).expanduser()) if model.lower().endswith(".pt") else model
        self.task = task
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.device = device
        self.supported_elements = None  # UMA covers nearly the whole periodic table

    def worker_script(self) -> Path:
        return WORKER

    def request(self, atoms) -> dict:
        periodic = bool(np.any(atoms.pbc))
        return {
            "model": self.model,
            "atom_refs": self.atom_refs,
            "task": self.task,
            "device": self.device,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "numbers": [int(z) for z in atoms.numbers],
            "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist() if periodic else None,
            "pbc": [bool(p) for p in atoms.pbc],
        }

    def results_from(self, answer: dict, atoms) -> dict:
        energy = float(answer["energy"])
        forces = np.asarray(answer["forces"], float).reshape(len(atoms), 3)
        return {"energy": energy, "free_energy": energy, "forces": forces}

    def describe_error(self, error: str) -> str:
        return f"UMA ({Path(self.model).name}, {self.task}, charge {self.charge}): {error}"
