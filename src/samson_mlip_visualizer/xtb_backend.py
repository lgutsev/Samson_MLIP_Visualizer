"""GFN-xTB through the ``xtb`` executable, as an ASE calculator.

A cheap, independent cross-check next to an MLIP: semi-empirical tight binding
has no training set to fall out of, so a barrier or geometry that agrees between
MACE and GFN2-xTB is more believable than either alone.

The executable runs in a subprocess rather than through ``tblite`` in-process:
it keeps xtb's own numpy and Fortran runtime out of SAMSON's Python (which pins
numpy 1.24), and the conda-forge ``tblite-python`` build crashes on Windows. One
call costs ~0.1 s of process start-up on top of the calculation; the SCF
restarts from the previous call's wavefunction.

Install xtb once, in an environment of its own::

    micromamba create -n xtb -c conda-forge xtb

then choose the ``xtb`` (``xtb.exe``) executable as the model file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import weakref
from pathlib import Path

import numpy as np
from ase.calculators.calculator import CalculationFailed, Calculator, all_changes
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree

METHODS = {
    "gfn2": ["--gfn", "2"],
    "gfn1": ["--gfn", "1"],
    "gfnff": ["--gfnff"],
}
# GFN1/GFN2-xTB and GFN-FF are parametrized for H through Rn.
SUPPORTED_ELEMENTS = frozenset(chemical_symbols[1:87])
_INPUT = "samson_xtb.xyz"
# Keep SAMSON's GUI from flashing a console window on every call.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _generic_rotation(axis=(1.0, 2.0, 3.0), degrees: float = 37.0) -> np.ndarray:
    axis = np.asarray(axis) / np.linalg.norm(axis)
    angle = np.radians(degrees)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


# xtb 6.7.1's GFN1/GFN2 analytic gradient is wrong when a molecule lies in a
# coordinate plane with a bond along an axis (bent HCN built by ASE: ~1 eV/Å off,
# with a net torque); the same molecule rotated is correct. Structures from a
# builder are often axis-aligned, so xtb always sees them in this fixed, generic
# orientation. Energies are unchanged; forces are rotated back.
_ROTATION = _generic_rotation()


def normalize_method(method: str) -> str:
    key = method.lower().replace("-", "").replace("xtb", "").replace(" ", "")
    key = {"2": "gfn2", "1": "gfn1", "ff": "gfnff"}.get(key, key)
    if key not in METHODS:
        raise ValueError(f"Unknown xTB method {method!r}; choose GFN2, GFN1, or GFN-FF")
    return key


def find_xtb() -> Path | None:
    """The xtb executable: ``$XTB_EXE``, then ``PATH``, then common conda envs."""
    configured = os.environ.get("XTB_EXE")
    if configured and Path(configured).is_file():
        return Path(configured)
    found = shutil.which("xtb")
    if found:
        return Path(found)
    home = Path.home()
    name = "xtb.exe" if sys.platform == "win32" else "xtb"
    tail = ("Library", "bin", name) if sys.platform == "win32" else ("bin", name)
    for root in ("micromamba", "miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        for env in sorted((home / root / "envs").glob("*")):
            candidate = env.joinpath(*tail)
            if candidate.is_file():
                return candidate
    return None


def is_xtb_executable(path: str | Path) -> bool:
    return Path(path).name.lower() in ("xtb", "xtb.exe")


def xtb_version(executable: str | Path) -> str | None:
    """The version xtb reports, e.g. ``6.7.1``, or ``None`` if it cannot be read."""
    try:
        output = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        if "xtb version" in line:
            return line.split("xtb version", 1)[1].split()[0]
    return None


def read_engrad(path: str | Path, natoms: int) -> tuple[float, np.ndarray]:
    """Energy (Eh) and gradient (Eh/bohr, ``(natoms, 3)``) from an ORCA-style .engrad."""
    values = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if int(values[0]) != natoms:
        raise CalculationFailed(f"xtb wrote {values[0]} atoms, expected {natoms}")
    energy = float(values[1])
    gradient = np.array([float(value) for value in values[2 : 2 + 3 * natoms]])
    return energy, gradient.reshape(natoms, 3)


class XTBCalculator(Calculator):
    """Energies and forces from ``xtb <geometry> --grad``.

    ``method`` is GFN2 (default), GFN1, or GFN-FF; ``multiplicity`` sets the
    number of unpaired electrons (``--uhf``); ``solvent`` enables ALPB implicit
    solvation (e.g. ``"water"``). Molecules only: periodic structures are refused.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        executable: str | Path,
        *,
        method: str = "gfn2",
        charge: int = 0,
        multiplicity: int = 1,
        solvent: str | None = None,
        accuracy: float = 1.0,
        timeout: float = 3600.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.executable = Path(executable)
        self.method = normalize_method(method)
        if multiplicity < 1:
            raise ValueError("The multiplicity must be at least 1")
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.solvent = solvent or None
        self.accuracy = float(accuracy)
        self.timeout = timeout
        self.supported_elements = SUPPORTED_ELEMENTS
        self._workdir = Path(tempfile.mkdtemp(prefix="samson-xtb-"))
        self._cleanup = weakref.finalize(self, shutil.rmtree, self._workdir, True)
        self._restart_numbers: tuple[int, ...] | None = None

    def command(self) -> list[str]:
        command = [str(self.executable), _INPUT, "--grad", *METHODS[self.method]]
        command += ["--chrg", str(self.charge), "--uhf", str(self.multiplicity - 1)]
        command += ["--acc", f"{self.accuracy:g}"]
        if self.solvent:
            command += ["--alpb", self.solvent]
        return command

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        if atoms.pbc.any():
            raise CalculationFailed("The xTB backend handles molecules only, not periodic cells")
        numbers = tuple(int(z) for z in atoms.numbers)
        restart = self._workdir / "xtbrestart"
        if numbers != self._restart_numbers and restart.exists():
            restart.unlink()  # a wavefunction from another system would mislead the SCF
        center = atoms.positions.mean(axis=0)
        positions = (atoms.positions - center) @ _ROTATION.T
        lines = [str(len(atoms)), "SAMSON MLIP Visualizer"]
        lines += [
            f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
            for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), positions, strict=True)
        ]
        (self._workdir / _INPUT).write_text("\n".join(lines) + "\n")
        engrad = self._workdir / (Path(_INPUT).stem + ".engrad")
        engrad.unlink(missing_ok=True)
        try:
            run = subprocess.run(
                self.command(),
                cwd=self._workdir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CalculationFailed(f"Could not run {self.executable}: {exc}") from exc
        if run.returncode != 0 or not engrad.is_file():
            tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-8:])
            raise CalculationFailed(f"xtb failed (exit code {run.returncode}):\n{tail}")
        energy, gradient = read_engrad(engrad, len(atoms))
        self._restart_numbers = numbers
        self.results = {
            "energy": energy * Hartree,
            "free_energy": energy * Hartree,
            "forces": -(gradient @ _ROTATION) * Hartree / Bohr,
        }
