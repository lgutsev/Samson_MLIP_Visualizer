"""IRC and QST through the CLI and the bridge's job runner."""

import pytest
from ase import Atoms
from ase.io import read, write
from samson_fakes import Atom, FakeSamson, Model, install_samson_module
from test_reaction_path import BondWell
from test_remote_dispatcher import TOKEN, result

from samson_mlip_visualizer import cli
from samson_mlip_visualizer.remote import jobs as jobs_module
from samson_mlip_visualizer.remote.dispatcher import Dispatcher

pytest.importorskip("sella")


def write_diatomic(path, r):
    write(path, Atoms("H2", positions=[[0, 0, 0], [r, 0, 0]]))


@pytest.fixture
def bondwell_cli(monkeypatch, tmp_path):
    model = tmp_path / "model.model"
    model.write_bytes(b"x")
    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: BondWell())
    return tmp_path, model


def test_cli_prfo_is_the_default_ts_method(bondwell_cli, capsys):
    tmp_path, model = bondwell_cli
    structure = tmp_path / "guess.xyz"
    write_diatomic(structure, 1.85)
    assert cli.main([str(structure), str(model), "--ts", "--fmax", "0.001"]) == 0
    out = capsys.readouterr().out
    assert "P-RFO (Sella)" in out and "Finished (converged)" in out


def test_cli_irc_writes_every_frame(bondwell_cli, capsys):
    tmp_path, model = bondwell_cli
    structure = tmp_path / "ts.xyz"
    trajectory = tmp_path / "irc.extxyz"
    write_diatomic(structure, 2.0)
    code = cli.main(
        [str(structure), str(model), "--irc", "--irc-step", "0.05", "--fmax", "0.001",
         "--trajectory", str(trajectory)]
    )
    assert code == 0
    out = capsys.readouterr().out
    frames = read(trajectory, index=":")
    assert f"Wrote all {len(frames)} IRC frames" in out
    bonds = sorted([frames[0].get_distance(0, 1), frames[-1].get_distance(0, 1)])
    assert bonds == pytest.approx([1.5, 2.5], abs=0.05)
    assert max(frame.info["energy_ev"] for frame in frames) == pytest.approx(1.0)


def test_cli_qst2_and_qst3(bondwell_cli, capsys):
    tmp_path, model = bondwell_cli
    reactant, product, guess = (tmp_path / name for name in ("r.xyz", "p.xyz", "g.xyz"))
    write_diatomic(reactant, 1.5)
    write_diatomic(product, 2.5)
    write_diatomic(guess, 2.1)
    band = tmp_path / "band.extxyz"
    output = tmp_path / "ts.xyz"
    assert cli.main(
        [str(reactant), str(model), "--qst", str(product), "--images", "5",
         "--trajectory", str(band), "-o", str(output)]
    ) == 0
    assert "barrier 1.0000" in capsys.readouterr().out
    assert len(read(band, index=":")) == 7
    assert read(output).get_distance(0, 1) == pytest.approx(2.0, abs=1e-3)
    assert cli.main(
        [str(reactant), str(model), "--qst", str(product), "--qst-guess", str(guess)]
    ) == 0
    assert "QST3" in capsys.readouterr().out


# --- bridge jobs -------------------------------------------------------------------------


def diatomic_model(r, name, selected=True):
    atoms = [Atom("H", [0, 0, 0]), Atom("H", [r, 0, 0])]
    return Model(atoms, selected=selected, cell=None, name=name)


@pytest.fixture
def bondwell_bridge(monkeypatch, tmp_path):
    def setup(*models):
        samson = FakeSamson(list(models))
        install_samson_module(monkeypatch, samson)
        monkeypatch.setattr(jobs_module, "create_calculator", lambda *a, **k: BondWell())
        model = tmp_path / "model.model"
        model.write_bytes(b"x")
        return samson, Dispatcher(TOKEN, samson=samson), str(model)

    return setup


def test_irc_job_returns_all_frames(bondwell_bridge, tmp_path):
    samson, dispatcher, model = bondwell_bridge(diatomic_model(2.0, "TS"))
    trajectory = tmp_path / "irc.extxyz"
    job = result(
        dispatcher, "job.start", kind="irc", model=model, step=0.05, fmax=0.001,
        trajectory=str(trajectory), return_positions=True,
    )
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["state"] == "finished", status["error"]
    irc = status["result"]
    assert irc["frames"] == len(irc["energies_ev"]) == len(irc["positions"])
    assert irc["energies_ev"][irc["ts_frame"]] == pytest.approx(1.0)
    assert len(read(trajectory, index=":")) == irc["frames"]
    # The stand-in has no SBPath, so publishing the path is reported, not fatal.
    assert irc["publish_errors"]
    atoms = samson.models[0].atoms
    assert abs(atoms[1].position[0] - atoms[0].position[0]) == pytest.approx(2.0)  # back at TS


def test_qst_job_uses_selected_models(bondwell_bridge):
    samson, dispatcher, model = bondwell_bridge(
        diatomic_model(1.5, "reactant"), diatomic_model(2.5, "product")
    )
    job = result(dispatcher, "job.start", kind="qst", model=model, images=5, fmax=0.01)
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["state"] == "finished", status["error"]
    qst = status["result"]
    assert qst["method"] == "QST2" and qst["neb_converged"]
    assert qst["barrier_forward_ev"] == pytest.approx(1.0, abs=1e-4)
    assert qst["ts"]["converged"] and qst["ts"]["frequencies"]["n_imaginary"] == 1


def test_qst_job_needs_two_or_three_models(bondwell_bridge):
    samson, dispatcher, model = bondwell_bridge(diatomic_model(1.5, "only"))
    job = result(dispatcher, "job.start", kind="qst", model=model)
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["state"] == "failed" and "2 (QST2) or 3 (QST3)" in status["error"]


def test_ts_job_defaults_to_prfo(bondwell_bridge):
    samson, dispatcher, model = bondwell_bridge(diatomic_model(1.85, "guess"))
    job = result(dispatcher, "job.start", kind="ts", model=model, fmax=0.001)
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["state"] == "finished", status["error"]
    assert status["result"]["method"] == "prfo" and status["result"]["converged"]
