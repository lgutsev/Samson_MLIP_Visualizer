import pytest
from ase.io import read
from samson_fakes import FakeSamson, install_samson_module, water_document

from samson_mlip_visualizer.remote.dispatcher import Dispatcher
from samson_mlip_visualizer.remote.protocol import (
    BUSY,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_ERROR,
    UNAUTHORIZED,
    UNAVAILABLE,
    decode,
    encode,
)

TOKEN = "test-token"


def call(dispatcher, method, token=TOKEN, **params):
    request = {"jsonrpc": "2.0", "id": 7, "method": method, "params": params}
    if token is not None:
        request["token"] = token
    response = decode(dispatcher.handle_line(encode(request)))
    assert response["id"] == 7
    return response


def result(dispatcher, method, **params):
    response = call(dispatcher, method, **params)
    assert "error" not in response, response.get("error")
    return response["result"]


def error(dispatcher, method, **params):
    return call(dispatcher, method, **params)["error"]


@pytest.fixture
def samson(monkeypatch):
    facade = water_document()
    install_samson_module(monkeypatch, facade)
    return facade


@pytest.fixture
def dispatcher(samson):
    return Dispatcher(TOKEN, samson=samson)


def test_requires_a_token():
    with pytest.raises(ValueError):
        Dispatcher("")


def test_rejects_missing_or_wrong_token(dispatcher):
    for token in (None, "wrong", "", "tést"):
        assert call(dispatcher, "bridge.ping", token=token)["error"]["code"] == UNAUTHORIZED
    assert dispatcher.requests == 0


def test_reports_malformed_requests(dispatcher):
    assert decode(dispatcher.handle_line(b"{oops"))["error"]["code"] == PARSE_ERROR
    assert error(dispatcher, "no.such.method")["code"] == METHOD_NOT_FOUND
    line = encode({"id": 7, "method": "bridge.ping", "params": [1], "token": TOKEN})
    assert decode(dispatcher.handle_line(line))["error"]["code"] == INVALID_PARAMS


def test_ping_and_info(dispatcher):
    assert result(dispatcher, "bridge.ping")["pong"] is True
    info = result(dispatcher, "bridge.info")
    assert info["allow_exec"] is False
    assert "structure.get" in info["methods"]
    assert "python.exec" not in info["methods"]
    assert info["requests"] == 2


def test_summary_lists_models_and_selection(dispatcher):
    summary = result(dispatcher, "document.summary")
    assert summary["atoms"] == 6
    assert summary["selected_atoms"] == 1
    assert [model["name"] for model in summary["models"]] == ["water 1", "water 2"]
    assert [model["selected"] for model in summary["models"]] == [True, False]
    assert summary["models"][0]["unit_cell"] is False


def test_structure_get_follows_selection_or_all(dispatcher):
    auto = result(dispatcher, "structure.get")
    assert auto["symbols"] == ["O", "H", "H"]
    assert auto["models"] == ["water 1"]
    assert auto["selected_atoms"] == [0]
    assert auto["cell"] is None
    assert auto["pbc"] == [False, False, False]

    everything = result(dispatcher, "structure.get", models="all")
    assert len(everything["symbols"]) == 6
    assert everything["fixed"] == [3]
    assert everything["positions"][3] == [3.0, 0.0, 0.0]
    assert error(dispatcher, "structure.get", models="some")["code"] == INVALID_PARAMS


def test_set_positions_writes_one_undo_step(dispatcher, samson):
    positions = [[0, 0, 0.1], [1, 0, 0], [0, 1, 0]]
    reply = result(
        dispatcher,
        "structure.set_positions",
        positions=positions,
        symbols=["O", "H", "H"],
        label="Nudge",
    )
    assert reply == {"atoms": 3}
    assert samson.models[0].atoms[0].position == pytest.approx([0, 0, 0.1])
    assert samson.holds == ["Nudge"]
    assert samson.events == 0  # no event-loop re-entry inside a request


def test_set_positions_validates_input(dispatcher, samson):
    def code(**params):
        return error(dispatcher, "structure.set_positions", **params)["code"]

    assert code(positions=[[0, 0, 0]]) == INVALID_PARAMS
    assert code(positions=[[0, 0, 0]] * 3, symbols=["H", "H", "H"]) == INVALID_PARAMS
    assert code(positions=[[0, 0, "x"]] * 3) == INVALID_PARAMS
    raw = (
        b'{"id":7,"method":"structure.set_positions","token":"test-token",'
        b'"params":{"positions":[[Infinity,0,0],[0,0,0],[0,0,0]]}}'
    )
    infinite = decode(dispatcher.handle_line(raw))["error"]
    assert infinite["code"] == INVALID_PARAMS and "finite" in infinite["message"]
    assert samson.holds == []


def test_busy_blocks_document_changes_but_not_reads(dispatcher):
    dispatcher.busy = "MD run"
    failure = error(dispatcher, "structure.set_positions", positions=[[0, 0, 0]] * 3)
    assert failure["code"] == BUSY
    assert "MD run" in failure["message"]
    assert result(dispatcher, "structure.get")["symbols"] == ["O", "H", "H"]


def test_selection_roundtrip(dispatcher):
    assert result(dispatcher, "selection.get") == {"models": [0], "atoms": [0], "atom_count": 6}
    assert result(dispatcher, "selection.set", atoms=[4, 5]) == {"models": [], "atoms": [4, 5]}
    assert result(dispatcher, "selection.get") == {"models": [], "atoms": [4, 5], "atom_count": 6}
    result(dispatcher, "selection.set", models=[1], clear=False)
    assert result(dispatcher, "selection.get") == {"models": [1], "atoms": [4, 5], "atom_count": 6}
    assert error(dispatcher, "selection.set", atoms=[6])["code"] == INVALID_PARAMS
    assert error(dispatcher, "selection.set", atoms=[True])["code"] == INVALID_PARAMS


def test_import_and_command_use_samson(dispatcher, samson, tmp_path):
    path = tmp_path / "in.xyz"
    path.write_text("1\n\nH 0 0 0\n")
    assert result(dispatcher, "file.import", path=str(path))["path"] == str(path)
    assert samson.imports == [str(path)]
    missing = error(dispatcher, "file.import", path=str(tmp_path / "missing.xyz"))
    assert missing["code"] == INVALID_PARAMS
    assert result(dispatcher, "command.run", name="Center") == {"name": "Center", "result": True}
    assert samson.commands == ["Center"]


def test_missing_samson_functions_are_reported(tmp_path):
    class Minimal:
        def getNodes(self, query):
            return []

    dispatcher = Dispatcher(TOKEN, samson=Minimal())
    path = tmp_path / "in.xyz"
    path.write_text("1\n\nH 0 0 0\n")
    assert error(dispatcher, "file.import", path=str(path))["code"] == UNAVAILABLE
    assert error(dispatcher, "command.run", name="Center")["code"] == UNAVAILABLE


def test_export_writes_structure(dispatcher, tmp_path):
    path = tmp_path / "out.xyz"
    assert result(dispatcher, "file.export", path=str(path), models="all")["atoms"] == 6
    assert read(path).get_chemical_symbols() == ["O", "H", "H", "O", "H", "H"]


def test_operation_failures_become_server_errors(monkeypatch):
    empty = FakeSamson([])
    install_samson_module(monkeypatch, empty)
    failure = error(Dispatcher(TOKEN, samson=empty), "structure.get")
    assert failure["code"] == SERVER_ERROR
    assert "no structural model" in failure["message"]
    assert failure["data"]["type"] == "SamsonBridgeError"


def test_exec_is_off_by_default(dispatcher):
    failure = error(dispatcher, "python.exec", code="1 + 1")
    assert failure["code"] == METHOD_NOT_FOUND
    assert "allow_exec" in failure["message"]


def test_exec_runs_code_and_keeps_state(samson):
    dispatcher = Dispatcher(TOKEN, samson=samson, allow_exec=True)
    code = "x = len(SAMSON.getNodes('node.type structuralModel'))\nprint('models', x)"
    assert result(dispatcher, "python.exec", code=code) == {
        "stdout": "models 2\n",
        "stderr": "",
        "result": None,
    }
    assert result(dispatcher, "python.exec", code="x * 10")["result"] == "20"

    failure = error(dispatcher, "python.exec", code="print('before')\n1 / 0")
    assert failure["code"] == SERVER_ERROR
    assert "ZeroDivisionError" in failure["message"]
    assert failure["data"]["stdout"] == "before\n"
    assert "Traceback" in failure["data"]["traceback"]
    assert error(dispatcher, "python.exec", code="raise SystemExit(3)")["code"] == SERVER_ERROR
    assert error(dispatcher, "python.exec", code="def (")["code"] == INVALID_PARAMS


def test_request_callback_sees_authorized_methods_only(samson):
    seen = []
    dispatcher = Dispatcher(TOKEN, samson=samson, on_request=seen.append)
    result(dispatcher, "bridge.ping")
    call(dispatcher, "bridge.ping", token="wrong")
    assert seen == ["bridge.ping"]
