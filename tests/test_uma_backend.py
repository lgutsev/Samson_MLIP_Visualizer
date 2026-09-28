"""UMA backend: the worker protocol with a fake worker, and (if configured) real UMA.

The real-model test needs ``FAIRCHEM_PYTHON`` (a Python with fairchem-core) and
``UMA_CHECKPOINT`` (a UMA .pt file, e.g. uma-s-1p1.pt).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import CalculationFailed

from samson_mlip_visualizer import uma_backend
from samson_mlip_visualizer.calculators import (
    CalculatorLoadError,
    create_calculator,
    is_program,
    program_options,
)
from samson_mlip_visualizer.provenance import collect_provenance
from samson_mlip_visualizer.remote.qt_server import _job_options
from samson_mlip_visualizer.uma_backend import (
    UMACalculator,
    fairchem_version,
    find_fairchem,
    has_fairchem,
)

# A stand-in worker that echoes the request back: E = charge + Σ|r|², F = −2r, and
# the last request is written next to it so the test can read what was sent.
FAKE_WORKER = '''
import json, sys
from pathlib import Path
sys.stdout.write("@@SAMSON " + json.dumps({"ready": True, "version": "0.0-fake"}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    request = json.loads(line)
    Path(__file__).with_name("last_request.json").write_text(line)
    if request["model"] == "fail":
        reply = {"error": "unsupported element"}
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
    monkeypatch.setattr(uma_backend, "WORKER", worker)
    return sys.executable, tmp_path / "last_request.json"


def test_worker_protocol_passes_task_charge_and_spin(fake_worker, tmp_path):
    python, sent = fake_worker
    checkpoint = tmp_path / "uma-s-1p1.pt"
    checkpoint.write_bytes(b"")
    atoms = Atoms("CH3ClF", positions=np.arange(18, dtype=float).reshape(6, 3) / 10)
    atoms.calc = calc = UMACalculator(python, model=str(checkpoint), charge=-1, multiplicity=1)
    assert atoms.get_potential_energy() == pytest.approx(-1 + (atoms.positions**2).sum())
    assert np.allclose(atoms.get_forces(), -2 * atoms.positions)
    request = json.loads(sent.read_text())
    assert (request["task"], request["charge"], request["multiplicity"]) == ("omol", -1, 1)
    assert request["model"] == str(checkpoint) and request["cell"] is None
    assert calc.version == "0.0-fake" and calc.supported_elements is None
    calc.close()


def test_periodic_cells_reach_the_worker(fake_worker):
    python, sent = fake_worker
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=[5.43] * 3, pbc=True)
    atoms.calc = calc = UMACalculator(python, model="uma-s-1p1", task="omat")
    atoms.get_potential_energy()
    request = json.loads(sent.read_text())
    assert request["task"] == "omat" and request["pbc"] == [True, True, True]
    assert np.allclose(request["cell"], np.diag([5.43] * 3))
    calc.close()


def test_worker_errors_and_bad_settings(fake_worker, tmp_path):
    python, _ = fake_worker
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    atoms.calc = UMACalculator(python, model="fail")
    with pytest.raises(CalculationFailed, match="unsupported element"):
        atoms.get_potential_energy()
    with pytest.raises(ValueError, match="Choose a UMA checkpoint"):
        UMACalculator(python)
    with pytest.raises(ValueError, match="does not exist"):
        UMACalculator(python, model=str(tmp_path / "missing.pt"))
    with pytest.raises(ValueError, match="task must be one of"):
        UMACalculator(python, model="uma-s-1p1", task="omolecule")
    with pytest.raises(ValueError, match="apply to the omol task"):
        UMACalculator(python, model="uma-s-1p1", task="omat", charge=-1)
    with pytest.raises(ValueError, match="atom references file does not exist"):
        UMACalculator(python, model="uma-s-1p1", atom_refs=tmp_path / "refs.yaml")


def test_fairchem_environment_detection(tmp_path, monkeypatch):
    python = tmp_path / "env" / "python.exe"
    python.parent.mkdir()
    python.write_text("")
    assert not has_fairchem(python) and not is_program("uma", python)
    site = tmp_path / "env" / "Lib" / "site-packages"
    (site / "fairchem" / "core").mkdir(parents=True)
    (site / "fairchem_core-2.23.0.dist-info").mkdir()
    assert has_fairchem(python) and fairchem_version(python) == "2.23.0"
    assert is_program("uma", python) and not is_program("aimnet2", python)
    monkeypatch.setenv("FAIRCHEM_PYTHON", str(python))
    assert find_fairchem() == python
    checkpoint = tmp_path / "uma-s-1p1.pt"
    checkpoint.write_bytes(b"")
    options = program_options("uma", method=str(checkpoint), charge=-1)
    assert options == {"model": str(checkpoint), "task": "omol", "charge": -1, "multiplicity": 1}
    calc = create_calculator("uma", python, device="cuda", options=options)
    assert (calc.charge, calc.task, calc.device) == (-1, "omol", "cpu")
    provenance = collect_provenance(backend="uma", model_path=python, device="cuda",
                                    dtype="float64", settings=options)
    assert (provenance.device, provenance.dtype) == ("cpu", "float32")
    assert provenance.versions["fairchem-core"] == "2.23.0"
    with pytest.raises(CalculatorLoadError, match="python executable"):
        create_calculator("uma", checkpoint)  # the checkpoint is an option, not the model
    with pytest.raises(CalculatorLoadError, match="Choose a UMA checkpoint"):
        create_calculator("uma", python, options=program_options("uma"))


def test_panel_options_become_job_parameters():
    # A job's "model" is the program; the panel's network must not overwrite it.
    uma = program_options("uma", method="uma-s-1p1.pt", task="omol", charge=-1)
    assert _job_options("uma", uma) == {"uma_model": "uma-s-1p1.pt", "uma_task": "omol",
                                        "charge": -1, "multiplicity": 1}
    assert _job_options("aimnet2", program_options("aimnet2"))["aimnet_model"] == "aimnet2"
    xtb = _job_options("xtb", program_options("xtb", charge=1))
    assert xtb == {"xtb_method": "gfn2", "charge": 1, "multiplicity": 1}  # no solvent: unset
    assert _job_options("psi4", program_options("psi4"))["psi4_method"] == "pbe"


REAL = os.environ.get("FAIRCHEM_PYTHON"), os.environ.get("UMA_CHECKPOINT")


@pytest.mark.skipif(not (REAL[0] and REAL[1] and Path(REAL[1]).is_file()),
                    reason="set FAIRCHEM_PYTHON and UMA_CHECKPOINT to run real UMA")
def test_real_uma_anion_forces_match_finite_differences():
    # The Cl- + CH3Cl SN2 transition-state region: an anion, so charge must matter.
    atoms = Atoms(
        "CClClH3",
        positions=[[0, 0, 0.05], [0, 0, 2.3], [0, 0, -2.35],
                   [1.07, 0, 0], [-0.535, 0.9267, 0.02], [-0.535, -0.9267, 0]],
    )
    calc = UMACalculator(REAL[0], model=REAL[1], charge=-1)
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
    neutral = UMACalculator(REAL[0], model=REAL[1], charge=0, multiplicity=2)
    shifted = atoms.copy()
    shifted.calc = neutral
    assert abs(shifted.get_potential_energy() - atoms.get_potential_energy()) > 0.5
    calc.close()
    neutral.close()
