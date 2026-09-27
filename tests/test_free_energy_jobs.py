"""Slow growth, blue moon, and metadynamics as bridge jobs, on the three-atom spring model."""

import numpy as np
import pytest
from samson_fakes import Atom, FakeSamson, Model, install_samson_module
from test_free_energy import KT, Bonds, exact_free_energy, three_atoms
from test_remote_dispatcher import TOKEN, result

from samson_mlip_visualizer import free_energy as fe
from samson_mlip_visualizer.remote import jobs as jobs_module
from samson_mlip_visualizer.remote.dispatcher import Dispatcher

COORDINATE = "0-1, 0-2:-1"
MD = {"temperature_k": 300.0, "timestep_fs": 0.5, "andersen_probability": 0.1,
      "report_interval": 50}


@pytest.fixture
def springs(monkeypatch, tmp_path):
    """Atom 0 with atoms 1 and 2 on springs, at ξ = d01 − d02 = 0, in a fake SAMSON."""
    atoms = three_atoms(0.0)
    model = Model([Atom(symbol, position) for symbol, position in
                   zip(atoms.get_chemical_symbols(), atoms.positions, strict=True)],
                  selected=True, cell=None, name="springs")
    samson = FakeSamson([model])
    install_samson_module(monkeypatch, samson)
    monkeypatch.setattr(jobs_module, "create_calculator",
                        lambda *a, **k: Bonds([(0, 1), (0, 2)]))
    path = tmp_path / "model.model"
    path.write_bytes(b"x")
    return samson, Dispatcher(TOKEN, samson=samson), str(path)


def run(dispatcher, kind, model, **options):
    job = result(dispatcher, "job.start", kind=kind, model=model, **options)
    return result(dispatcher, "job.status", id=job["id"])


def test_parse_coordinate_and_integration_helpers():
    coordinate = fe.parse_coordinate("0-4, 0-5:-1")
    assert coordinate.terms == [(0, 4, 1.0), (0, 5, -1.0)]
    assert fe.parse_coordinate("1-2:0.5; 3-4").terms == [(1, 2, 0.5), (3, 4, 1.0)]
    for bad in ("", "0-", "a-b", "0-1:x"):
        with pytest.raises(ValueError):
            fe.parse_coordinate(bad)
    xi, gradient = [-1.0, 0.0, 1.0, 2.0], [-1.0, 1.0, -1.0, -2.0]
    assert fe.zero_crossing(xi, gradient, up=True) == pytest.approx(-0.5)
    assert fe.zero_crossing(xi, gradient, up=False) == pytest.approx(0.5)
    assert fe.zero_crossing(xi, [1.0, 2.0, 3.0, 4.0], up=True) is None
    error = fe.integration_error([0.0, 1.0, 2.0], [0.2, 0.2, 0.2])
    assert error == pytest.approx([0.0, np.sqrt(0.02), np.sqrt(0.04)])


def test_constrained_md_and_metadynamics_stop_when_asked():
    coordinate = fe.DistanceCombination([(0, 1, 1.0), (0, 2, -1.0)])
    calls = []
    record = fe.constrained_md(three_atoms(0.0), coordinate, steps=100, timestep_fs=0.5,
                               should_stop=lambda: calls.append(1) or len(calls) >= 5)
    assert len(record.lam) == 5
    atoms = three_atoms(0.0)
    bias = fe.MetadynamicsCalculator(atoms.calc, coordinate, height=0.01, sigma=0.1,
                                     temperature_k=300.0)
    calls.clear()
    out = fe.metadynamics(atoms, bias, steps=100, pace=2, timestep_fs=0.5,
                          should_stop=lambda: calls.append(1) or len(calls) >= 7)
    assert len(out["centers"]) == 4  # steps 0, 2, 4, 6


def test_blue_moon_job_integrates_the_exact_profile(springs, tmp_path):
    samson, dispatcher, model = springs
    output = tmp_path / "windows.npz"
    status = run(dispatcher, "blue_moon", model, coordinate=COORDINATE,
                 values=[0.4, 0.0, -0.4], steps=8000, skip=500, increment=2e-3,
                 output=str(output), **MD)
    assert status["state"] == "finished", status["error"]
    blue = status["result"]
    assert [w["xi"] for w in blue["windows"]] == [0.4, 0.0, -0.4]  # run order
    assert blue["xi"] == [-0.4, 0.0, 0.4]  # integrated in increasing ξ
    exact = exact_free_energy(np.array(blue["xi"]))
    assert blue["free_energy_ev"] == pytest.approx(exact - exact[0], abs=2 * KT)
    assert blue["xi_min"] == pytest.approx(0.0, abs=0.15)  # the springs' symmetric minimum
    assert blue["xi_star"] is None and "barrier_ev" not in blue
    assert all(e > 0 for e in blue["free_energy_error_ev"][1:])
    assert np.load(output)["lam_2"].shape == (8000,)
    assert blue["moved_atoms"] and blue["publish_errors"]  # the fake has no SBPath
    atoms = samson.models[0].atoms
    d01 = np.linalg.norm(np.subtract(atoms[1].position, atoms[0].position))
    d02 = np.linalg.norm(np.subtract(atoms[2].position, atoms[0].position))
    assert d01 - d02 == pytest.approx(-0.4, abs=1e-6)  # left at the last window


def test_slow_growth_job_moves_to_the_start_and_across(springs):
    samson, dispatcher, model = springs
    status = run(dispatcher, "slow_growth", model, coordinate=COORDINATE, start=-0.3,
                 end=0.3, increment=5e-4, equilibration_steps=200, **MD)
    assert status["state"] == "finished", status["error"]
    growth = status["result"]
    assert growth["steps"] == 1200 and not growth["stopped"]
    assert growth["xi"][0] == pytest.approx(-0.3 + 5e-4) and growth["xi"][-1] == pytest.approx(0.3)
    assert len(growth["xi"]) == len(growth["free_energy_ev"]) <= 201
    assert abs(growth["end_ev"]) < 0.1  # a symmetric profile, from one noisy fast run
    assert "hysteresis" in growth["note"]


def test_metadynamics_job_respects_the_walls(springs):
    samson, dispatcher, model = springs
    status = run(dispatcher, "metadynamics", model, coordinate=COORDINATE, steps=600, pace=20,
                 height=0.01, sigma=0.1, bias_factor=5.0, lower_wall=-0.3, upper_wall=0.3,
                 wall_k=50.0, max_distances="0-1:2.5", **MD)
    assert status["state"] == "finished", status["error"]
    meta = status["result"]
    assert meta["hills"] == 30 and meta["last_hill_height_ev"] < 0.01
    assert -0.3 - 1e-9 <= meta["xi"][0] and meta["xi"][-1] <= 0.3 + 1e-9
    assert min(meta["free_energy_ev"]) == 0.0


def test_free_energy_jobs_check_their_inputs(springs):
    samson, dispatcher, model = springs
    cases = [
        ({"kind": "blue_moon", "values": [0.0, 0.1]}, "coordinate"),
        ({"kind": "slow_growth", "coordinate": "0-7", "end": 1.0}, "has 3 atoms"),
        ({"kind": "blue_moon", "coordinate": COORDINATE, "values": [0.1]}, "two or more"),
        ({"kind": "metadynamics", "coordinate": COORDINATE, "max_distances": "0-1"}, "limits"),
        ({"kind": "metadynamics", "coordinate": COORDINATE, "bias_factor": 1.0}, "above 1"),
    ]
    for options, message in cases:
        status = run(dispatcher, model=model, **options)
        assert status["state"] == "failed" and message in status["error"], (options, status)
