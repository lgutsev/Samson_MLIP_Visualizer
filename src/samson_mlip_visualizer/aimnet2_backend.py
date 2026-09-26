"""AIMNet2, a charge-aware MLIP, as an ASE calculator.

MACE-MP-0 and MACE-OFF have no notion of total charge, so an anion such as the
Cl⁻ + CH₃Cl SN2 complex is evaluated as a neutral (radical) system. AIMNet2
(Isayev group; ωB97M-D3 reference) takes the molecular charge and
multiplicity as inputs and adds explicit long-range Coulomb, which makes it the
MLIP to use for ions and polar reactions of organic molecules (H, B, C, N, O,
F, Si, P, S, Cl, As, Se, Br, I).

aimnet needs a newer numpy than SAMSON's Python allows, so it runs in its own
environment in a worker process (:mod:`.aimnet2_worker`). Install it once::

    micromamba create -p <envs>/mlip -c conda-forge python=3.12 pip
    <envs>/mlip/python -m pip install aimnet

then choose that environment's ``python`` as the model file. ``model`` picks
the network: an aimnet registry name (``aimnet2`` = ωB97M-D3, downloaded on
first use) or the path of a downloaded ``.pt`` file. The model runs in float32.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .worker_process import WorkerCalculator

WORKER = Path(__file__).with_name("aimnet2_worker.py")
DEFAULT_MODEL = "aimnet2"
# The ωB97M-D3 and B97-3c AIMNet2 models; other registry models (Pd, reaction,
# open-shell variants) differ, and the worker's own species check covers them.
AIMNET2_ELEMENTS = frozenset("H B C N O F Si P S Cl As Se Br I".split())


def _site_packages(python: Path) -> list[Path]:
    root = python.parent
    return [root / "Lib" / "site-packages", *sorted(root.parent.glob("lib/python3*/site-packages"))]


def has_aimnet(python: str | Path) -> bool:
    """Whether ``python`` is an interpreter whose environment contains aimnet."""
    python = Path(python)
    return python.is_file() and any((site / "aimnet").is_dir() for site in _site_packages(python))


def aimnet_version(python: str | Path) -> str | None:
    """aimnet's version from its dist-info folder, without importing it."""
    for site in _site_packages(Path(python)):
        for info in sorted(site.glob("aimnet-*.dist-info")):
            return info.name[len("aimnet-") : -len(".dist-info")]
    return None


def find_aimnet() -> Path | None:
    """A Python with aimnet: ``$AIMNET_PYTHON``, then common conda/micromamba envs."""
    configured = os.environ.get("AIMNET_PYTHON")
    if configured and has_aimnet(configured):
        return Path(configured)
    home = Path.home()
    name = "python.exe" if sys.platform == "win32" else "bin/python"
    for root in ("micromamba", "miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        for env in sorted((home / root / "envs").glob("*")):
            if has_aimnet(env / name):
                return env / name
    return None


class AIMNet2Calculator(WorkerCalculator):
    """Energies and forces from AIMNet2 in a worker process.

    ``charge`` and ``multiplicity`` are the molecule's; ``model`` is a registry
    name or a ``.pt`` path; ``device`` is ``cpu`` or ``cuda`` (the worker's
    torch decides whether CUDA works). Molecules only.
    """

    label_name = "AIMNet2"

    def __init__(
        self,
        python: str | Path,
        *,
        model: str = DEFAULT_MODEL,
        charge: int = 0,
        multiplicity: int = 1,
        device: str = "cpu",
        timeout: float = 600.0,
        **kwargs,
    ):
        super().__init__(python, timeout=timeout, **kwargs)
        if multiplicity < 1:
            raise ValueError("The multiplicity must be at least 1")
        model = (model or DEFAULT_MODEL).strip()
        if model.lower().endswith((".pt", ".jpt")) and not Path(model).expanduser().is_file():
            raise ValueError(f"AIMNet2 model file does not exist: {model}")
        self.model = model
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.device = device
        name = Path(model).stem.lower().replace("-", "_")
        known = name.startswith(("aimnet2_wb97m", "aimnet2_b973c")) or name == "aimnet2"
        self.supported_elements = AIMNET2_ELEMENTS if known else None

    def worker_script(self) -> Path:
        return WORKER

    def request(self, atoms) -> dict:
        model = self.model
        if model.lower().endswith((".pt", ".jpt")):
            model = str(Path(model).expanduser())
        return {
            "model": model,
            "device": self.device,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "numbers": [int(z) for z in atoms.numbers],
            "positions": atoms.positions.tolist(),
        }

    def results_from(self, answer: dict, atoms) -> dict:
        energy = float(answer["energy"])
        forces = np.asarray(answer["forces"], float).reshape(len(atoms), 3)
        return {"energy": energy, "free_energy": energy, "forces": forces}

    def describe_error(self, error: str) -> str:
        return f"AIMNet2 ({self.model}, charge {self.charge}): {error}"
