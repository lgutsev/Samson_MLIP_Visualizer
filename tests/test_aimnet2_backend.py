"""AIMNet2 backend: the worker protocol with a fake worker, and (if installed) real AIMNet2."""

import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import CalculationFailed

from samson_mlip_visualizer import aimnet2_backend
from samson_mlip_visualizer.aimnet2_backend import (
    AIMNet2Calculator,
    aimnet_version,
    find_aimnet,
    has_aimnet,
)
from samson_mlip_visualizer.calculators import (
    CalculatorLoadError,
    create_calculator,
    is_program,
    program_options,
)
from samson_mlip_visualizer.provenance import collect_provenance

# A stand-in worker: E = charge + Σ|r|² (eV, Å), so F = -2r; it errors on model "fail".
FAKE_WORKER = '''
import json, sys
print("noise the parent must ignore")
sys.stdout.write("@@SAMSON " + json.dumps({"ready": True, "version": "0.0-fake"}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    request = json.loads(line)
    if request["model"] == "fail":
        reply = {"error": "species not implemented"}
    else:
        positions = request["positions"]
        reply = {"energy": request["charge"] + sum(x * x for p in positions for x in p),
                 "forces": [[-2 * x for x in p] for p in positions]}
    sys.stdout.write("@@SAMSON " + json.dumps(reply) + "\\n")
    sys.stdout.flush()
'''


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(aimnet2_backend, "WORKER", worker)
    return sys.executable


def test_worker_protocol_passes_charge(fake_worker):
    atoms = Atoms("ClCH3Cl", positions=np.arange(18, dtype=float).reshape(6, 3) / 10)
    atoms.calc = calc = AIMNet2Calculator(fake_worker, charge=-1)
    assert atoms.get_potential_energy() == pytest.approx(-1 + (atoms.positions**2).sum())
    assert np.allclose(atoms.get_forces(), -2 * atoms.positions)
    assert calc.version == "0.0-fake"
    assert "Cl" in calc.supported_elements and "Na" not in calc.supported_elements
    first = calc._process
    atoms.positions[0, 0] += 0.1
    atoms.get_potential_energy()
    assert calc._process is first  # one worker for all calculations
    calc.close()
    assert first.poll() is not None


def test_worker_ignores_host_python_settings(fake_worker, tmp_path, monkeypatch):
    # SAMSON sets these for its embedded Python; a worker in another env must not see them.
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "samson-python"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "samson-python"))
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    atoms.calc = calc = AIMNet2Calculator(fake_worker)
    assert atoms.get_potential_energy() == pytest.approx(0.49)
    calc.close()


def test_worker_errors_and_bad_settings(fake_worker, tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    atoms.calc = AIMNet2Calculator(fake_worker, model="fail")
    with pytest.raises(CalculationFailed, match="species not implemented"):
        atoms.get_potential_energy()
    with pytest.raises(ValueError, match="does not exist"):
        AIMNet2Calculator(fake_worker, model=str(tmp_path / "missing.pt"))
    with pytest.raises(ValueError, match="multiplicity"):
        AIMNet2Calculator(fake_worker, multiplicity=0)
    # A model the element table does not know leaves the check to the worker.
    assert AIMNet2Calculator(fake_worker, model="aimnet2-pd").supported_elements is None


def test_aimnet_environment_detection(tmp_path, monkeypatch):
    python = tmp_path / "env" / "python.exe"
    python.parent.mkdir()
    python.write_text("")
    assert not has_aimnet(python) and not is_program("aimnet2", python)
    site = tmp_path / "env" / "Lib" / "site-packages"
    (site / "aimnet").mkdir(parents=True)
    (site / "aimnet-0.2.0.dist-info").mkdir()
    assert has_aimnet(python) and aimnet_version(python) == "0.2.0"
    assert is_program("aimnet2", python) and not is_program("psi4", python)
    monkeypatch.setenv("AIMNET_PYTHON", str(python))
    assert find_aimnet() == python
    options = program_options("aimnet2", charge=-1)
    assert options == {"model": "aimnet2", "charge": -1, "multiplicity": 1}
    calc = create_calculator("aimnet2", python, device="cuda", options=options)
    assert (calc.charge, calc.device) == (-1, "cpu")  # the MACE device does not carry over
    provenance = collect_provenance(
        backend="aimnet2", model_path=python, device="cuda", dtype="float64", settings=options
    )
    assert (provenance.device, provenance.dtype) == ("cpu", "float32")
    assert provenance.versions["aimnet"] == "0.2.0"
    weights = tmp_path / "aimnet2_wb97m_d3_0.pt"
    weights.write_bytes(b"")
    with pytest.raises(CalculatorLoadError, match="python executable"):
        create_calculator("aimnet2", weights)  # the weights are an option, not the model


@pytest.mark.skipif(find_aimnet() is None, reason="aimnet is not installed (set AIMNET_PYTHON)")
def test_real_aimnet2_anion_forces_match_finite_differences():
    # The Cl- + CH3Cl SN2 transition-state region: an anion only a charge-aware model handles.
    atoms = Atoms(
        "CClClH3",
        positions=[[0, 0, 0.05], [0, 0, 2.3], [0, 0, -2.35],
                   [1.07, 0, 0], [-0.535, 0.9267, 0.02], [-0.535, -0.9267, 0]],
    )
    calc = AIMNet2Calculator(find_aimnet(), charge=-1)
    atoms.calc = calc
    forces = atoms.get_forces()
    step, index = 5e-3, (0, 2)
    energies = []
    for sign in (1, -1):
        displaced = atoms.copy()
        displaced.positions[index] += sign * step
        displaced.calc = calc
        energies.append(displaced.get_potential_energy())
    assert forces[index] == pytest.approx(-(energies[0] - energies[1]) / (2 * step), abs=5e-3)
    neutral = AIMNet2Calculator(find_aimnet(), charge=0)
    shifted = atoms.copy()
    shifted.calc = neutral
    assert abs(shifted.get_potential_energy() - atoms.get_potential_energy()) > 0.5
    calc.close()
    neutral.close()
