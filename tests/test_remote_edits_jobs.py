import pytest
from ase.calculators.emt import EMT
from samson_fakes import Atom, FakeSamson, Model, install_samson_module, water_document
from test_remote_dispatcher import TOKEN, call, error, result

from samson_mlip_visualizer.remote import jobs as jobs_module
from samson_mlip_visualizer.remote.dispatcher import Dispatcher
from samson_mlip_visualizer.remote.protocol import BUSY, INVALID_PARAMS


@pytest.fixture
def samson(monkeypatch):
    facade = water_document()
    install_samson_module(monkeypatch, facade)
    return facade


@pytest.fixture
def dispatcher(samson):
    return Dispatcher(TOKEN, samson=samson)


def test_summary_reports_effective_selection(dispatcher):
    summary = result(dispatcher, "document.summary")
    assert summary["selected_atoms"] == 1
    assert summary["effectively_selected_atoms"] == 3


def test_nsl_selection(dispatcher, samson):
    reply = result(dispatcher, "selection.select", nsl="node.type atom and atom.symbol O")
    assert reply["nsl"] == "node.type atom and atom.symbol O"
    assert samson.selections == ["node.type atom and atom.symbol O"]
    assert error(dispatcher, "selection.select", nsl="invalid")["code"] == INVALID_PARAMS


def test_view_capture(dispatcher, samson, tmp_path):
    path = tmp_path / "view.png"
    reply = result(dispatcher, "view.capture", path=str(path), width=640, height=480)
    assert reply == {"path": str(path), "width": 640, "height": 480}
    assert path.read_bytes().startswith(b"\x89PNG")
    assert samson.captures == [(640, 480, False, False, False)]
    assert error(dispatcher, "view.capture", path=str(path), width=5)["code"] == INVALID_PARAMS


def test_add_delete_and_edit_atoms(dispatcher, samson):
    atoms = [{"symbol": "Na", "position": [5, 5, 5]}, {"symbol": "Cl", "position": [7.5, 5, 5]}]
    reply = result(dispatcher, "atoms.add", atoms=atoms)  # the selected model
    assert reply == {"added": 2, "model": "water 1", "atom_count": 8}
    first = samson.models[0].atoms
    assert [atom.elementSymbol for atom in first] == ["O", "H", "H", "Na", "Cl"]
    assert first[3].created and first[4].position == [7.5, 5, 5]

    result(dispatcher, "atoms.set_elements", atoms=[3, 4], symbols=["K", "Br"])
    assert [atom.elementSymbol for atom in first[3:]] == ["K", "Br"]
    result(dispatcher, "atoms.set_fixed", atoms=[0, 1])
    assert first[0].fixedFlag and first[1].fixedFlag
    result(dispatcher, "atoms.set_fixed", atoms=[1], fixed=False)
    assert not first[1].fixedFlag

    assert result(dispatcher, "atoms.delete", atoms=[3, 4]) == {"deleted": 2, "atom_count": 6}
    assert samson.holds == [
        "Remote: add atoms",
        "Remote: change elements",
        "Remote: set fixed atoms",
        "Remote: set fixed atoms",
        "Remote: delete atoms",
    ]


def test_edit_validation_changes_nothing(dispatcher, samson):
    bad_symbol = [{"symbol": "Xx", "position": [0, 0, 0]}]
    assert error(dispatcher, "atoms.add", atoms=bad_symbol)["code"] == -32000
    bad_position = [{"symbol": "C", "position": [0, 0]}]
    assert error(dispatcher, "atoms.add", atoms=bad_position)["code"] == INVALID_PARAMS
    assert error(dispatcher, "atoms.delete", atoms=[99])["code"] == INVALID_PARAMS
    assert error(dispatcher, "atoms.set_elements", atoms=[0, 1], symbols=["C"])["code"] == (
        INVALID_PARAMS
    )
    assert len(samson.models[0].atoms) == 3 and samson.holds == []


def test_add_needs_model_when_several_selected(samson, dispatcher):
    samson.models[1].selectionFlag = True
    atoms = [{"symbol": "C", "position": [0, 0, 5]}]
    assert error(dispatcher, "atoms.add", atoms=atoms)["code"] == INVALID_PARAMS
    assert result(dispatcher, "atoms.add", atoms=atoms, model=1)["model"] == "water 2"


def test_undo_redo(dispatcher, samson):
    result(dispatcher, "history.undo")
    result(dispatcher, "history.redo")
    assert samson.history == ["undo", "redo"]


# --- jobs ------------------------------------------------------------------------------


def copper_document():
    positions = [[0, 0, 0], [2.6, 0, 0], [1.3, 2.2, 0], [1.3, 0.7, 2.1]]
    return FakeSamson([Model([Atom("Cu", p) for p in positions], cell=None, name="Cu4")])


@pytest.fixture
def jobs_setup(monkeypatch, tmp_path):
    samson = copper_document()
    install_samson_module(monkeypatch, samson)
    monkeypatch.setattr(jobs_module, "create_calculator", lambda *a, **k: EMT())
    model = tmp_path / "model.model"
    model.write_bytes(b"fake")
    scheduled = []
    dispatcher = Dispatcher(TOKEN, samson=samson, schedule=scheduled.append)
    return samson, dispatcher, scheduled, str(model)


def test_relax_job_runs_after_reply_and_blocks_edits(jobs_setup):
    samson, dispatcher, scheduled, model = jobs_setup
    job = result(dispatcher, "job.start", kind="relax", model=model, fmax=0.01, optimizer="BFGS")
    assert job["state"] == "queued" and len(scheduled) == 1
    assert error(dispatcher, "atoms.set_fixed", atoms=[0])["code"] == BUSY
    assert error(dispatcher, "job.start", kind="single_point", model=model)["code"] == BUSY

    scheduled.pop()()  # the Qt server runs this from the event loop
    status = result(dispatcher, "job.status", id=job["id"], log_lines=3)
    assert status["state"] == "finished"
    assert status["result"]["converged"] is True
    assert status["result"]["provenance"]["mlip_backend"] == "mace"
    assert samson.holds == ["Remote MLIP relax"]
    assert samson.events > 0  # the job pumped the event loop
    assert dispatcher.busy is None
    assert len(status["log"]) == 3


def test_job_that_moves_nothing_leaves_no_undo_step(jobs_setup):
    samson, dispatcher, scheduled, model = jobs_setup
    job = result(dispatcher, "job.start", kind="relax", model=model, fmax=100.0)
    scheduled.pop()()
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["result"]["steps"] == 0 and status["result"]["moved_atoms"] is False
    assert samson.holds == []


def test_md_and_single_point_jobs(jobs_setup):
    samson, dispatcher, scheduled, model = jobs_setup
    job = result(
        dispatcher, "job.start", kind="md", model=model, steps=20, temperature_k=50, seed=1,
        timestep_fs=2.0, report_interval=5,
    )
    scheduled.pop()()
    md = result(dispatcher, "job.status", id=job["id"])
    assert md["state"] == "finished" and md["result"]["steps"] == 20

    point = result(dispatcher, "job.start", kind="single_point", model=model)
    scheduled.pop()()
    energy = result(dispatcher, "job.status", id=point["id"])["result"]["energy_ev"]
    assert isinstance(energy, float)
    assert [entry["kind"] for entry in result(dispatcher, "job.list")] == ["md", "single_point"]


def test_job_stop_and_failures(jobs_setup):
    samson, dispatcher, scheduled, model = jobs_setup
    job = result(dispatcher, "job.start", kind="md", model=model, steps=5000, timestep_fs=2.0)
    assert result(dispatcher, "job.stop", id=job["id"])["params"]["steps"] == 5000
    scheduled.pop()()
    assert result(dispatcher, "job.status", id=job["id"])["state"] == "stopped"

    bad = result(dispatcher, "job.start", kind="md", model=model, ensemble="NPT")
    scheduled.pop()()
    failed = result(dispatcher, "job.status", id=bad["id"])
    assert failed["state"] == "failed" and "ensemble" in failed["error"]
    assert dispatcher.busy is None

    assert error(dispatcher, "job.start", kind="dance", model=model)["code"] == INVALID_PARAMS
    assert error(dispatcher, "job.start", kind="relax")["code"] == INVALID_PARAMS  # no model
    assert error(dispatcher, "job.status", id=99)["code"] == INVALID_PARAMS


def test_job_defaults_supply_the_model(monkeypatch, tmp_path):
    samson = copper_document()
    install_samson_module(monkeypatch, samson)
    monkeypatch.setattr(jobs_module, "create_calculator", lambda *a, **k: EMT())
    model = tmp_path / "model.model"
    model.write_bytes(b"fake")
    dispatcher = Dispatcher(TOKEN, samson=samson, job_defaults=lambda: {"model": str(model)})
    job = result(dispatcher, "job.start", kind="frequencies")  # runs inline without a scheduler
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["state"] == "finished"
    assert status["result"]["rigid_body_modes_removed"] == 6
    assert call(dispatcher, "job.list")["result"][0]["params"]["model"] == str(model)
