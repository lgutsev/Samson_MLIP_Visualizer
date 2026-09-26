"""Psi4 backend: the worker protocol with a fake worker, and (if installed) real Psi4."""

import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import CalculationFailed
from ase.units import Bohr, Hartree

from samson_mlip_visualizer import psi4_backend
from samson_mlip_visualizer.calculators import (
    CalculatorLoadError,
    create_calculator,
    program_options,
)
from samson_mlip_visualizer.psi4_backend import (
    Psi4Calculator,
    find_psi4,
    has_psi4,
    psi4_version,
)

# A stand-in worker: E = Σ|r|² (Eh, bohr), so the gradient is 2r; it errors on "fail".
FAKE_WORKER = '''
import json, sys
print("noise the parent must ignore")
sys.stdout.write("@@SAMSON " + json.dumps({"ready": True, "version": "0.0-fake"}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    request = json.loads(line)
    if request["method"] == "fail":
        reply = {"error": "SCF did not converge"}
    else:
        bohr = [[x / 0.52917721 for x in p] for p in request["positions"]]
        reply = {"energy": sum(x * x for p in bohr for x in p),
                 "gradient": [[2 * x for x in p] for p in bohr]}
    sys.stdout.write("@@SAMSON " + json.dumps(reply) + "\\n")
    sys.stdout.flush()
'''


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(psi4_backend, "WORKER", worker)
    return sys.executable


def test_worker_protocol_energy_and_forces(fake_worker):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.8]])
    atoms.calc = calc = Psi4Calculator(fake_worker)
    bohr = atoms.positions / Bohr
    assert atoms.get_potential_energy() == pytest.approx((bohr**2).sum() * Hartree)
    assert np.allclose(atoms.get_forces(), -2 * bohr * Hartree / Bohr)
    assert calc.version == "0.0-fake"
    first = calc._process
    atoms.positions[1, 2] += 0.1
    atoms.get_potential_energy()
    assert calc._process is first  # one worker for all calculations
    calc.close()
    assert first.poll() is not None


def test_worker_errors_and_periodic_cells(fake_worker):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    atoms.calc = Psi4Calculator(fake_worker, method="fail")
    with pytest.raises(CalculationFailed, match="SCF did not converge"):
        atoms.get_potential_energy()
    crystal = molecule("H2", vacuum=3.0)
    crystal.pbc = True
    crystal.calc = Psi4Calculator(fake_worker)
    with pytest.raises(CalculationFailed, match="molecules only"):
        crystal.get_potential_energy()


def test_psi4_environment_detection(tmp_path, monkeypatch):
    python = tmp_path / "env" / "python.exe"
    python.parent.mkdir()
    python.write_text("")
    assert not has_psi4(python)
    package = tmp_path / "env" / "Lib" / "site-packages" / "psi4"
    package.mkdir(parents=True)
    (package / "metadata.py").write_text("__version__ = '1.11'\n")
    assert has_psi4(python) and psi4_version(python) == "1.11"
    monkeypatch.setenv("PSI4_PYTHON", str(python))
    assert find_psi4() == python
    calc = create_calculator("psi4", python, options={"method": "PBE0", "basis": "def2-SVP"})
    assert (calc.method, calc.basis) == ("pbe0", "def2-svp")
    assert "Rn" in calc.supported_elements
    with pytest.raises(CalculatorLoadError, match="python executable"):
        create_calculator("psi4", package / "metadata.py")


def test_program_options_defaults():
    assert program_options("psi4") == {
        "method": "pbe", "basis": "def2-tzvp", "charge": 0, "multiplicity": 1
    }
    assert program_options("xtb", solvent="")["solvent"] is None
    assert program_options("mace") is None


@pytest.mark.skipif(find_psi4() is None, reason="Psi4 is not installed")
def test_real_psi4_forces_match_finite_differences():
    water = molecule("H2O")
    water.rattle(0.05, seed=3)
    calc = Psi4Calculator(find_psi4(), method="pbe", basis="def2-svp", threads=2)
    water.calc = calc
    forces = water.get_forces()
    step, index = 1e-3, (1, 0)
    energies = []
    for sign in (1, -1):
        displaced = water.copy()
        displaced.positions[index] += sign * step
        displaced.calc = calc
        energies.append(displaced.get_potential_energy())
    assert forces[index] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=5e-3)
    calc.close()
