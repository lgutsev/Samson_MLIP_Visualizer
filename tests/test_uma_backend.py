"""UMA backend: presets and their rules, the worker protocol with a fake worker, and
(if configured) real UMA.

The real-model tests need ``FAIRCHEM_PYTHON`` (a Python with fairchem-core) and
``UMA_CHECKPOINT`` (a UMA .pt file or fairchem name, e.g. uma-s-1p2p1);
``UMA_DEVICE`` (default cpu) picks the device.
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
from samson_mlip_visualizer.uma_config import (
    PRESETS,
    UMAConfig,
    UMAConfigError,
    structure_warnings,
)

# A stand-in worker that echoes the request back: E = charge + Σ|r|², F = −2r, and
# the last request is written next to it so the test can read what was sent. It
# reports a Nitro fallback the way the real worker does.
FAKE_WORKER = '''
import json, os, sys
from pathlib import Path
sys.stdout.write("@@SAMSON " + json.dumps({"ready": True, "version": "0.0-fake",
    "versions": {"fairchem-core": "0.0-fake"}, "gpu": {"cuda": False}}) + "\\n")
sys.stdout.flush()
first = True
for line in sys.stdin:
    request = json.loads(line)
    config = request["config"]
    Path(__file__).with_name("last_request.json").write_text(line)
    if config["model"] == "fail":
        reply = {"error": "unsupported element"}
    else:
        positions = request["positions"]
        reply = {"energy": config["charge"] + sum(x * x for p in positions for x in p),
                 "forces": [[-2 * x for x in p] for p in positions],
                 "backend": {"requested": config["inference"], "effective": "fallback",
                             "nitro": False, "tf32": False, "merge_mole": False,
                             "compile": False, "dispersion": config["dispersion"],
                             "warnings": ["Nitro needs a GPU; using multigpu settings"]},
                 "events": ["UMA loaded on cpu: requested nitro, effective fallback"]
                           if first else [],
                 "timing": {"load_s": 0.0, "eval_s": 0.0, "pid": os.getpid()}}
        first = False
    sys.stdout.write("@@SAMSON " + json.dumps(reply) + "\\n")
    sys.stdout.flush()
'''


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(uma_backend, "WORKER", worker)
    uma_backend.shutdown_workers()
    yield sys.executable, tmp_path / "last_request.json"
    uma_backend.shutdown_workers()


# --- presets ------------------------------------------------------------------------


def test_presets_fill_in_their_checkpoint_task_and_dispersion():
    standard = UMAConfig().validated()
    assert (standard.model, standard.task, standard.dispersion, standard.inference) == (
        "uma-s-1p2p1", "omol", "none", "nitro")
    d4rev = UMAConfig("d4rev", charge=-1, multiplicity=2).validated()
    assert (d4rev.model, d4rev.task, d4rev.dispersion) == ("uma-s-1p2p1-ft-d4rev", "omol",
                                                          "none")
    assert (d4rev.charge, d4rev.multiplicity) == (-1, 2)  # omol keeps charge and spin
    r2scan = UMAConfig("r2scan").validated()
    assert (r2scan.model, r2scan.task, r2scan.dispersion) == ("uma-s-1p2p1-ft-r2scan", "omat",
                                                             "r2scan-d3bj")
    assert any("not spin-polarized" in note for note in r2scan.notes)
    described = r2scan.describe()
    assert described["uma_spin_polarized"] == "false"
    assert "a1=0.49484001" in described["uma_d3_settings"]
    assert "cutoff_angstrom=15.0" in described["uma_d3_settings"]
    assert UMAConfig("d4rev").validated().describe()["uma_dispersion"] == (
        "included in the labels")


def test_invalid_task_combinations_are_rejected_before_a_run():
    with pytest.raises(UMAConfigError, match="only the omol task"):
        UMAConfig("d4rev", task="omat").validated()
    with pytest.raises(UMAConfigError, match="only the omat task"):
        UMAConfig("r2scan", task="omol").validated()
    with pytest.raises(UMAConfigError, match="task must be one of"):
        UMAConfig(task="omolecule").validated()
    with pytest.raises(UMAConfigError, match="runs uma-s-1p2p1, not uma-s-1p1"):
        UMAConfig(model="uma-s-1p1").validated()
    with pytest.raises(UMAConfigError, match="apply to the omol task"):
        UMAConfig(task="omat", charge=1).validated()
    with pytest.raises(UMAConfigError, match="preset must be one of"):
        UMAConfig("nitro").validated()
    # a fine-tune named by path in the custom preset still gets its preset's rules
    by_path = UMAConfig("custom", model="D:/UMA/uma-s-1p2p1-ft-d4rev.pt").validated()
    assert by_path.preset == "d4rev" and by_path.model.endswith(".pt")
    with pytest.raises(UMAConfigError, match="already includes dispersion"):
        UMAConfig("custom", model="uma-s-1p2p1-ft-d4rev", dispersion="pbe-d3bj").validated()


def test_double_dispersion_is_prevented():
    with pytest.raises(UMAConfigError, match="already includes dispersion"):
        UMAConfig("d4rev", dispersion="pbe-d3bj").validated()
    with pytest.raises(UMAConfigError, match="already includes dispersion"):
        UMAConfig("d4rev", dispersion="r2scan-d3bj").validated()
    for task in ("omol", "omc", "odac", "oc25"):  # their labels include dispersion
        with pytest.raises(UMAConfigError, match="count it twice"):
            UMAConfig(task=task, dispersion="pbe-d3bj").validated()
    with pytest.raises(UMAConfigError, match="RPBE"):
        UMAConfig(task="oc20", dispersion="pbe-d3bj").validated()
    with pytest.raises(UMAConfigError, match="takes dispersion pbe-d3bj"):
        UMAConfig(task="omat", dispersion="r2scan-d3bj").validated()
    with pytest.raises(UMAConfigError, match="takes dispersion r2scan-d3bj"):
        UMAConfig("r2scan", dispersion="pbe-d3bj").validated()
    for task in ("omat", "oc22"):
        config = UMAConfig(task=task, dispersion="pbe-d3bj").validated()
        assert "not equivalent to every VASP" in " ".join(config.notes)
        assert "a1=0.4289" in config.describe()["uma_d3_settings"]
    assert UMAConfig("r2scan", dispersion="none").validated().dispersion == "none"


def test_magnetic_use_of_the_r2scan_preset_is_rejected():
    with pytest.raises(UMAConfigError, match="magnetic calculation was requested"):
        UMAConfig("r2scan", magnetic=True).validated()
    with pytest.raises(UMAConfigError, match="not spin-polarized"):
        UMAConfig("r2scan", multiplicity=3).validated()
    assert UMAConfig(task="omat", magnetic=True).validated().preset == "standard"
    r2scan = UMAConfig("r2scan").validated()
    with pytest.raises(UMAConfigError, match="initial magnetic moments"):
        structure_warnings(r2scan, [28, 8], initial_magmoms=[1.5, 0.0])
    # elements and single atoms warn; neither decides whether a system is magnetic
    assert any("often magnetic" in w for w in structure_warnings(r2scan, [26] * 13))
    assert any("cohesive" in w for w in structure_warnings(r2scan, [14]))
    assert structure_warnings(r2scan, [14, 14]) == []
    assert structure_warnings(UMAConfig().validated(), [26] * 13) == []


def test_turbo_maps_to_nitro_and_high_precision_stays_selectable():
    turbo = UMAConfig(inference="turbo").validated()
    assert turbo.inference == "nitro"
    assert any("removed the 'turbo'" in note for note in turbo.notes)
    assert UMAConfig(inference="high-precision").validated().inference == "high-precision"
    with pytest.raises(UMAConfigError, match="inference mode"):
        UMAConfig(inference="fast").validated()
    assert set(PRESETS) == {"standard", "d4rev", "r2scan", "custom"}


# --- worker protocol ------------------------------------------------------------------


def test_worker_protocol_passes_the_config(fake_worker, tmp_path):
    python, sent = fake_worker
    checkpoint = tmp_path / "uma-s-1p1.pt"
    checkpoint.write_bytes(b"")
    atoms = Atoms("CH3ClF", positions=np.arange(18, dtype=float).reshape(6, 3) / 10)
    events = []
    atoms.calc = calc = UMACalculator(python, model=str(checkpoint), charge=-1,
                                      multiplicity=1, on_message=events.append)
    assert calc.config.preset == "custom"  # a checkpoint without a preset: older settings
    assert atoms.get_potential_energy() == pytest.approx(-1 + (atoms.positions**2).sum())
    assert np.allclose(atoms.get_forces(), -2 * atoms.positions)
    request = json.loads(sent.read_text())
    config = request["config"]
    assert (config["task"], config["charge"], config["multiplicity"]) == ("omol", -1, 1)
    assert config["model"] == str(checkpoint) and request["cell"] is None
    assert calc.version == "0.0-fake" and calc.supported_elements is None
    assert events == ["UMA loaded on cpu: requested nitro, effective fallback"]
    metadata = calc.run_metadata()
    assert metadata["uma_inference_requested"] == "nitro"
    assert metadata["uma_inference_effective"] == "fallback"
    assert "Nitro needs a GPU" in metadata["uma_fallback"]
    assert metadata["uma_version_fairchem-core"] == "0.0-fake"


def test_periodic_cells_and_dispersion_reach_the_worker(fake_worker):
    python, sent = fake_worker
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=[5.43] * 3, pbc=True)
    atoms.calc = UMACalculator(python, preset="r2scan", device="cuda")
    atoms.get_potential_energy()
    request = json.loads(sent.read_text())
    assert request["config"]["task"] == "omat" and request["pbc"] == [True, True, True]
    assert request["config"]["dispersion"] == "r2scan-d3bj" and request["device"] == "cuda"
    assert np.allclose(request["cell"], np.diag([5.43] * 3))


def test_calculators_share_one_worker(fake_worker):
    python, _ = fake_worker
    pids = []
    for preset in ("standard", "d4rev"):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calc = UMACalculator(python, preset=preset)
        atoms.get_potential_energy()
        pids.append(calc.timing["pid"])
        calc.close()  # detaches; the worker keeps its loaded predictors
    assert pids[0] == pids[1]


def test_magnetic_structures_are_refused_by_the_r2scan_preset(fake_worker):
    python, sent = fake_worker
    atoms = Atoms("NiO", positions=[[0, 0, 0], [2.1, 0, 0]], cell=[4.2] * 3, pbc=True)
    atoms.set_initial_magnetic_moments([1.7, 0.0])
    atoms.calc = UMACalculator(python, preset="r2scan")
    with pytest.raises(CalculationFailed, match="initial magnetic moments"):
        atoms.get_potential_energy()
    assert not sent.exists()  # rejected before anything reached the worker


def test_worker_errors_and_bad_settings(fake_worker, tmp_path):
    python, _ = fake_worker
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    atoms.calc = UMACalculator(python, preset="custom", model="fail")
    with pytest.raises(CalculationFailed, match="unsupported element"):
        atoms.get_potential_energy()
    with pytest.raises(ValueError, match="Choose a UMA checkpoint"):
        UMACalculator(python, preset="custom")
    with pytest.raises(ValueError, match="does not exist"):
        UMACalculator(python, model=str(tmp_path / "missing.pt"))
    with pytest.raises(ValueError, match="task must be one of"):
        UMACalculator(python, task="omolecule")
    with pytest.raises(ValueError, match="apply to the omol task"):
        UMACalculator(python, task="omat", charge=-1)
    with pytest.raises(ValueError, match="atom references file does not exist"):
        UMACalculator(python, atom_refs=tmp_path / "refs.yaml")
    with pytest.raises(ValueError, match="count it twice"):
        UMACalculator(python, preset="d4rev", dispersion="pbe-d3bj")


def test_fairchem_environment_detection(tmp_path, monkeypatch):
    python = tmp_path / "env" / "python.exe"
    python.parent.mkdir()
    python.write_text("")
    assert not has_fairchem(python) and not is_program("uma", python)
    site = tmp_path / "env" / "Lib" / "site-packages"
    (site / "fairchem" / "core").mkdir(parents=True)
    (site / "fairchem_core-2.24.0.dist-info").mkdir()
    assert has_fairchem(python) and fairchem_version(python) == "2.24.0"
    assert is_program("uma", python) and not is_program("aimnet2", python)
    monkeypatch.setenv("FAIRCHEM_PYTHON", str(python))
    assert find_fairchem() == python
    checkpoint = tmp_path / "uma-s-1p1.pt"
    checkpoint.write_bytes(b"")
    options = program_options("uma", method=str(checkpoint), charge=-1)
    assert options == {"preset": "custom", "model": str(checkpoint), "task": "omol",
                       "dispersion": None, "inference": "nitro", "charge": -1,
                       "multiplicity": 1}
    calc = create_calculator("uma", python, device="cuda", options=options)
    assert (calc.charge, calc.task, calc.device) == (-1, "omol", "cuda")
    assert program_options("uma", preset="r2scan")["task"] == "omat"
    provenance = collect_provenance(backend="uma", model_path=python, device="cuda",
                                    dtype="float64", settings=options)
    assert (provenance.device, provenance.dtype) == ("cuda", "float32")
    assert provenance.versions["fairchem-core"] == "2.24.0"
    with pytest.raises(CalculatorLoadError, match="python executable"):
        create_calculator("uma", checkpoint)  # the checkpoint is an option, not the model
    with pytest.raises(CalculatorLoadError, match="Choose a UMA checkpoint"):
        create_calculator("uma", python, options=program_options("uma", preset="custom"))
    with pytest.raises(CalculatorLoadError, match="magnetic"):
        create_calculator("uma", python,
                          options=program_options("uma", preset="r2scan", magnetic=True))


def test_panel_options_become_job_parameters():
    # A job's "model" is the program; the panel's network must not overwrite it.
    uma = program_options("uma", preset="d4rev", charge=-1, inference="high-precision")
    assert _job_options("uma", uma) == {"uma_preset": "d4rev", "uma_task": "omol",
                                        "uma_inference": "high-precision",
                                        "charge": -1, "multiplicity": 1}
    assert _job_options("aimnet2", program_options("aimnet2"))["aimnet_model"] == "aimnet2"
    xtb = _job_options("xtb", program_options("xtb", charge=1))
    assert xtb == {"xtb_method": "gfn2", "charge": 1, "multiplicity": 1}  # no solvent: unset
    assert _job_options("psi4", program_options("psi4"))["psi4_method"] == "pbe"


# --- real UMA ------------------------------------------------------------------------

REAL = os.environ.get("FAIRCHEM_PYTHON"), os.environ.get("UMA_CHECKPOINT")
DEVICE = os.environ.get("UMA_DEVICE", "cpu")
needs_real = pytest.mark.skipif(
    not (REAL[0] and REAL[1] and (Path(REAL[1]).is_file() or not REAL[1].endswith(".pt"))),
    reason="set FAIRCHEM_PYTHON and UMA_CHECKPOINT to run real UMA",
)


@needs_real
def test_real_uma_anion_forces_match_finite_differences():
    # The Cl- + CH3Cl SN2 transition-state region: an anion, so charge must matter.
    atoms = Atoms(
        "CClClH3",
        positions=[[0, 0, 0.05], [0, 0, 2.3], [0, 0, -2.35],
                   [1.07, 0, 0], [-0.535, 0.9267, 0.02], [-0.535, -0.9267, 0]],
    )
    calc = UMACalculator(REAL[0], preset="custom", model=REAL[1], charge=-1, device=DEVICE,
                         inference="high-precision")
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
    neutral = UMACalculator(REAL[0], preset="custom", model=REAL[1], charge=0,
                            multiplicity=2, device=DEVICE, inference="high-precision")
    shifted = atoms.copy()
    shifted.calc = neutral
    assert abs(shifted.get_potential_energy() - atoms.get_potential_energy()) > 0.5
    # the change of charge and spin rebuilt the predictor instead of a silent fallback
    assert any("Rebuilt" in m or "loaded" in m for m in neutral.messages)


@needs_real
def test_real_uma_pbe_d3_adds_dispersion_to_a_periodic_cell():
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43)
    plain = UMACalculator(REAL[0], preset="custom", model=REAL[1], task="omat", device=DEVICE)
    d3 = UMACalculator(REAL[0], preset="custom", model=REAL[1], task="omat",
                       dispersion="pbe-d3bj", device=DEVICE)
    energies = []
    for calc in (plain, d3):
        copy = atoms.copy()
        copy.calc = calc
        energies.append(copy.get_potential_energy())
        # symmetric, so no forces; nitro and its multigpu fallback use TF32 (~1 meV/Å noise)
        assert np.abs(copy.get_forces()).max() < 5e-3
    assert energies[1] < energies[0]  # D3 is attractive
    assert d3.run_metadata()["uma_dispersion"] == "pbe-d3bj"
