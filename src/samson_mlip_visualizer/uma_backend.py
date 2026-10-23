"""UMA (Meta FAIR's Universal Models for Atoms, fairchem) as an ASE calculator.

UMA is one network with task heads trained on several datasets. Its ``omol`` task
(OMol25, ωB97M-V/def2-TZVPD) takes the molecular charge and spin multiplicity as
inputs, which makes it, like AIMNet2, an MLIP for ions and polar reactions, over a
far larger part of the periodic table. The other tasks (``omat`` inorganic
materials, ``oc20``/``oc25`` catalysis, ``oc22`` oxides, ``odac`` MOFs, ``omc``
molecular crystals) take no charge or spin.

Which checkpoint, task, dispersion correction and inference mode run is a
:class:`~.uma_config.UMAConfig`, chosen by preset (``standard`` uma-s-1p2p1,
``d4rev``, ``r2scan``, or ``custom``) and validated before the worker starts.

fairchem needs a newer numpy and torch than SAMSON's Python allows, so it runs in
its own environment in a worker process (:mod:`.uma_worker`); see
``docs/uma.md`` for the pinned installation. Choose that environment's ``python``
as the model file. ``model`` is a UMA checkpoint: a fairchem name such as
``uma-s-1p2p1``, downloaded to the Hugging Face cache on first use (facebook/UMA
is gated: accept its licence and log in first), or the path of a downloaded
``.pt``. Workers are shared: a second calculator with the same Python reuses the
running process and its loaded predictors.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from pathlib import Path

import numpy as np
from ase.calculators.calculator import CalculationFailed, all_changes

from .aimnet2_backend import _site_packages
from .uma_config import PRESETS, TASKS, UMAConfig, UMAConfigError, structure_warnings
from .worker_process import WorkerCalculator, _stop

WORKER = Path(__file__).with_name("uma_worker.py")
DEFAULT_TASK = "omol"
__all__ = ["PRESETS", "TASKS", "UMACalculator", "UMAConfigError", "find_fairchem"]


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


# One worker process per fairchem Python, shared by every UMACalculator using it, so a
# new panel run or bridge job does not reload the checkpoint.
_SHARED: dict[str, dict] = {}
_SHARED_LOCK = threading.Lock()


def shutdown_workers() -> None:
    """Stop every shared UMA worker (the next calculation starts a new one)."""
    with _SHARED_LOCK:
        for shared in _SHARED.values():
            _stop(shared["process"])
        _SHARED.clear()


atexit.register(shutdown_workers)


class UMACalculator(WorkerCalculator):
    """Energies and forces from UMA in a worker process.

    ``preset`` picks the checkpoint and its rules (:data:`~.uma_config.PRESETS`);
    ``model`` overrides the checkpoint (a .pt path of the same checkpoint, or any
    checkpoint with ``preset="custom"``); ``task`` a UMA task head; ``dispersion``
    ``none``, ``pbe-d3bj`` or ``r2scan-d3bj``; ``inference`` ``nitro`` or
    ``high-precision``; ``charge`` and ``multiplicity`` are the molecule's (``omol``
    only); ``magnetic=True`` requests a spin-polarized description (rejected by the
    nonmagnetic r²SCAN preset). ``device`` is ``cpu`` or ``cuda``. Single atoms (a
    free F⁻) need the isolated-atom references: ``atom_refs``, or
    ``iso_atom_elem_refs.yaml`` beside the checkpoint or in ``references`` beside it.

    ``on_message`` receives backend events (the effective inference mode, fallback
    warnings, predictor rebuilds); they are also kept in :attr:`messages`.
    """

    label_name = "UMA"
    periodic = True  # the materials tasks take cells

    def __init__(
        self,
        python: str | Path,
        *,
        preset: str | None = None,
        model: str | None = None,
        task: str | None = None,
        dispersion: str | None = None,
        inference: str = "nitro",
        charge: int = 0,
        multiplicity: int = 1,
        magnetic: bool | None = None,
        device: str = "cpu",
        atom_refs: str | Path | None = None,
        timeout: float = 900.0,
        on_message=None,
        **kwargs,
    ):
        super().__init__(python, timeout=timeout, **kwargs)
        if atom_refs is not None and not Path(atom_refs).expanduser().is_file():
            raise ValueError(f"UMA atom references file does not exist: {atom_refs}")
        self.atom_refs = str(Path(atom_refs).expanduser()) if atom_refs is not None else None
        model = (model or "").strip() or None
        if preset is None:  # the settings of older panels and jobs: a checkpoint, no preset
            preset = "custom" if model else "standard"
        if model and model.lower().endswith(".pt"):
            if not Path(model).expanduser().is_file():
                raise ValueError(f"UMA checkpoint does not exist: {model}")
            model = str(Path(model).expanduser())
        if preset == "custom" and not model:
            raise ValueError(
                "Choose a UMA checkpoint: a fairchem name such as uma-s-1p2p1, or a .pt "
                "file such as uma-s-1p2p1.pt"
            )
        self.config = UMAConfig(
            preset=preset, model=model, task=task or (DEFAULT_TASK if preset in
                                                        ("standard", "custom") else None),
            dispersion=dispersion, inference=inference, charge=int(charge),
            multiplicity=int(multiplicity), magnetic=magnetic,
        ).validated()  # UMAConfigError is a ValueError
        self.device = device
        self.supported_elements = None  # UMA covers nearly the whole periodic table
        self.on_message = on_message
        self.messages: list[str] = list(self.config.notes)
        self.backend_info: dict | None = None
        self.worker_info: dict = {}
        self.timing: dict = {}
        self._checked: tuple | None = None
        self._lock = threading.Lock()

    # the attributes older code reads
    @property
    def model(self) -> str:
        return self.config.model

    @property
    def task(self) -> str:
        return self.config.task

    @property
    def charge(self) -> int:
        return self.config.charge

    @property
    def multiplicity(self) -> int:
        return self.config.multiplicity

    def _message(self, text: str) -> None:
        self.messages.append(text)
        if self.on_message is not None:
            self.on_message(text)

    def worker_script(self) -> Path:
        return WORKER

    # --- shared worker ----------------------------------------------------------------

    def _start(self) -> None:
        key = str(self.python.resolve() if self.python.exists() else self.python)
        with _SHARED_LOCK:
            shared = _SHARED.get(key)
            if shared is not None and shared["process"].poll() is None:
                self._process, self._replies = shared["process"], shared["replies"]
                self.version, self.worker_info = shared["version"], shared["info"]
                self._lock = shared["lock"]
                return
            super()._start()
            self._cleanup.detach()  # the pool, not this calculator, owns the process
            self._cleanup = None
            self._lock = threading.Lock()
            _SHARED[key] = {"process": self._process, "replies": self._replies,
                            "version": self.version, "info": self.worker_info,
                            "lock": self._lock}

    def _ready(self, message: dict) -> None:
        self.worker_info = {"versions": message.get("versions", {}),
                            "gpu": message.get("gpu", {})}

    def close(self) -> None:
        """Detach from the shared worker (:func:`shutdown_workers` stops it)."""
        self._process = None

    def _reply(self, timeout: float) -> dict:
        process = self._process
        try:
            return super()._reply(timeout)
        except CalculationFailed:
            # A late answer would reach the next request: stop the shared worker.
            with _SHARED_LOCK:
                for key in [k for k, v in _SHARED.items() if v["process"] is process]:
                    del _SHARED[key]
            if process is not None:
                _stop(process)
            raise

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        structure = (tuple(int(z) for z in atoms.numbers),) if atoms is not None else None
        if atoms is not None and structure != self._checked:
            magmoms = (atoms.get_initial_magnetic_moments()
                       if atoms.has("initial_magmoms") else None)
            try:
                for warning in structure_warnings(self.config, atoms.numbers, magmoms):
                    self._message("Warning: " + warning)
            except UMAConfigError as exc:
                raise CalculationFailed(self.describe_error(str(exc))) from exc
            self._checked = structure
        if self._process is None or self._process.poll() is not None:
            self._start()
        with self._lock:
            super().calculate(atoms, properties, system_changes)

    def request(self, atoms) -> dict:
        periodic = bool(np.any(atoms.pbc))
        return {
            "config": self.config.as_dict(),
            "atom_refs": self.atom_refs,
            "device": self.device,
            "numbers": [int(z) for z in atoms.numbers],
            "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist() if periodic else None,
            "pbc": [bool(p) for p in atoms.pbc],
        }

    def results_from(self, answer: dict, atoms) -> dict:
        for event in answer.get("events", ()):
            self._message(event)
        if "backend" in answer:
            self.backend_info = answer["backend"]
        self.timing = answer.get("timing", {})
        energy = float(answer["energy"])
        forces = np.asarray(answer["forces"], float).reshape(len(atoms), 3)
        return {"energy": energy, "free_energy": energy, "forces": forces}

    def describe_error(self, error: str) -> str:
        c = self.config
        return (f"UMA ({c.checkpoint}, {c.task}, {c.inference}, charge {c.charge}, "
                f"multiplicity {c.multiplicity}): {error}")

    def run_metadata(self) -> dict[str, str]:
        """What ran, for provenance: checkpoint, task, dispersion, inference (requested
        and effective, with fallback reasons), device, and the worker's versions."""
        info = dict(self.config.describe())
        info["uma_device"] = self.device
        gpu = self.worker_info.get("gpu", {})
        if self.device.startswith("cuda") and gpu.get("name"):
            info["uma_gpu"] = f"{gpu['name']} (sm {gpu.get('capability')})"
        backend = self.backend_info or {}
        if backend:
            info["uma_inference_effective"] = backend["effective"]
            info["uma_tf32"] = str(backend["tf32"]).lower()
            if backend.get("warnings"):
                info["uma_fallback"] = " | ".join(backend["warnings"])
        for name, value in self.worker_info.get("versions", {}).items():
            info[f"uma_version_{name}"] = value
        return info

