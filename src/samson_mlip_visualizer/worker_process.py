"""An ASE calculator whose work runs in a long-lived Python worker process.

Programs that cannot live in SAMSON's Python (numpy 1.24) run in an
environment of their own. The worker script imports nothing from this package;
it reads one JSON request per line from stdin and answers each with one line on
stdout that starts with :data:`PREFIX` (anything else a library prints is
ignored). The first line it writes is ``{"ready": true, "version": ...}`` or
``{"error": ...}``. The worker stays up between calculations, so only the first
one pays for imports and model loading.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import weakref
from pathlib import Path

from ase.calculators.calculator import CalculationFailed, Calculator, all_changes

PREFIX = "@@SAMSON "
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# SAMSON points these at its embedded Python; inherited, they make another
# environment's interpreter load SAMSON's standard library and die at start-up.
_HOST_PYTHON_VARIABLES = ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "PYTHONSTARTUP")


def worker_environment() -> dict[str, str]:
    """This process's environment without the host interpreter's Python settings."""
    return {k: v for k, v in os.environ.items() if k.upper() not in _HOST_PYTHON_VARIABLES}


class WorkerCalculator(Calculator):
    """Base class: subclasses name the worker script, build requests, read answers."""

    implemented_properties = ["energy", "free_energy", "forces"]
    label_name = "worker"  # for messages, e.g. "Psi4"
    periodic = False  # whether the worker takes periodic cells (sent in the request)

    def __init__(self, python: str | Path, *, timeout: float = 3600.0, **kwargs):
        super().__init__(**kwargs)
        self.python = Path(python)
        self.timeout = timeout
        self.version: str | None = None
        self._process: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self._cleanup = None

    # --- for subclasses ---------------------------------------------------------------

    def worker_script(self) -> Path:
        raise NotImplementedError

    def request(self, atoms) -> dict:
        raise NotImplementedError

    def results_from(self, answer: dict, atoms) -> dict:
        raise NotImplementedError

    def describe_error(self, error: str) -> str:
        return f"{self.label_name}: {error}"

    # --- worker process ---------------------------------------------------------------

    def _start(self) -> None:
        try:
            process = subprocess.Popen(
                [str(self.python), "-u", str(self.worker_script())],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=worker_environment(),
                creationflags=_NO_WINDOW,
            )
        except OSError as exc:
            raise CalculationFailed(
                f"Could not start {self.label_name} with {self.python}: {exc}"
            ) from exc
        self._process = process
        self._replies = queue.Queue()
        threading.Thread(target=self._read, args=(process, self._replies), daemon=True).start()
        self._cleanup = weakref.finalize(self, _stop, process)
        ready = self._reply(timeout=300)
        if not ready.get("ready"):
            self.close()
            raise CalculationFailed(
                ready.get("error", f"The {self.label_name} worker did not start")
            )
        self.version = ready.get("version")

    def _read(self, process, replies) -> None:
        for line in process.stdout:
            if line.startswith(PREFIX):
                replies.put(json.loads(line[len(PREFIX) :]))
        replies.put({"error": f"The {self.label_name} worker exited (code {process.wait()})"})

    def _reply(self, timeout: float) -> dict:
        try:
            return self._replies.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise CalculationFailed(
                f"{self.label_name} did not answer within {timeout:g} s"
            ) from None

    def close(self) -> None:
        """Stop the worker process (a later calculation starts a new one)."""
        if self._cleanup is not None:
            self._cleanup()
        self._process = None

    # --- ASE --------------------------------------------------------------------------

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        if atoms.pbc.any() and not self.periodic:
            raise CalculationFailed(
                f"The {self.label_name} backend handles molecules only, not periodic cells"
            )
        if self._process is None or self._process.poll() is not None:
            self._start()
        try:
            self._process.stdin.write(json.dumps(self.request(atoms)) + "\n")
            self._process.stdin.flush()
        except OSError as exc:
            self.close()
            raise CalculationFailed(f"The {self.label_name} worker is gone: {exc}") from exc
        answer = self._reply(self.timeout)
        if "error" in answer:
            raise CalculationFailed(self.describe_error(answer["error"]))
        self.results = self.results_from(answer, atoms)


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            process.stdin.close()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
