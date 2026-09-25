import json
import socket
import socketserver
import threading

import pytest
from ase import Atoms
from ase.io import read, write
from samson_fakes import install_samson_module, water_document

from samson_mlip_visualizer.remote.client import SamsonClient, main
from samson_mlip_visualizer.remote.dispatcher import Dispatcher
from samson_mlip_visualizer.remote.protocol import (
    CONNECTION_ENV,
    INVALID_PARAMS,
    UNAUTHORIZED,
    UNAVAILABLE,
    BridgeError,
    ConnectionInfo,
    LineBuffer,
    write_connection_file,
)

TOKEN = "client-test-token"


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        buffer = LineBuffer()
        while chunk := self.request.recv(1 << 16):
            for line in buffer.feed(chunk):
                self.request.sendall(self.server.dispatcher.handle_line(line))


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    """A pure-Python stand-in for the Qt transport, serving a fake document."""
    samson = water_document()
    install_samson_module(monkeypatch, samson)
    server = _Server(("127.0.0.1", 0), _Handler)
    server.dispatcher = Dispatcher(TOKEN, samson=samson, allow_exec=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    path = write_connection_file(
        ConnectionInfo("127.0.0.1", port, TOKEN, 1, "0.1.0", True, "now"), tmp_path / "bridge.json"
    )
    monkeypatch.setenv(CONNECTION_ENV, str(path))
    yield samson, port
    server.shutdown()
    server.server_close()


def test_client_roundtrip(bridge):
    samson, _ = bridge
    client = SamsonClient.from_connection_file()
    assert client.ping()["pong"] is True
    assert client.summary()["atoms"] == 6

    atoms = client.get_structure(models="all")
    assert isinstance(atoms, Atoms) and len(atoms) == 6
    assert atoms.constraints[0].get_indices().tolist() == [3]
    atoms.positions[0] += [0, 0, 0.5]
    assert client.set_positions(atoms, models="all", label="Remote test") == {"atoms": 6}
    assert samson.models[0].atoms[0].position == pytest.approx([0, 0, 0.5])
    assert samson.holds == ["Remote test"]

    assert client.get_selection()["atoms"] == [0]
    client.set_selection(atoms=[1, 2])
    assert client.get_selection()["atoms"] == [1, 2]
    assert client.run_command("Center")["result"] is True
    assert client.execute("1 + 2")["result"] == "3"


def test_client_raises_bridge_errors(bridge):
    _, port = bridge
    with pytest.raises(BridgeError) as invalid:
        SamsonClient.from_connection_file().call("structure.get", models="bad")
    assert invalid.value.code == INVALID_PARAMS
    with pytest.raises(BridgeError) as denied:
        SamsonClient("127.0.0.1", port, "wrong-token").ping()
    assert denied.value.code == UNAUTHORIZED


def test_client_explains_when_nothing_listens():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(BridgeError, match="not running") as error:
        SamsonClient("127.0.0.1", port, "token", timeout=10).ping()
    assert error.value.code == UNAVAILABLE


def test_cli_summary_get_and_exec(bridge, tmp_path, capsys):
    assert main(["summary"]) == 0
    assert json.loads(capsys.readouterr().out)["atoms"] == 6

    output = tmp_path / "document.xyz"
    assert main(["get", "--all", "-o", str(output)]) == 0
    capsys.readouterr()
    assert read(output).get_chemical_symbols() == ["O", "H", "H", "O", "H", "H"]

    assert main(["exec", "print('hi'); 6 * 7"]) == 0
    assert capsys.readouterr().out == "hi\n42\n"


def test_cli_set_positions_from_file(bridge, tmp_path):
    samson, _ = bridge
    atoms = SamsonClient.from_connection_file().get_structure()
    atoms.positions += 1.0
    path = tmp_path / "moved.xyz"
    write(path, atoms)
    assert main(["set-positions", str(path)]) == 0
    assert samson.models[0].atoms[1].position == pytest.approx([1.96, 1, 1])


def test_cli_reports_errors(bridge, capsys):
    assert main(["exec", "1 / 0"]) == 1
    err = capsys.readouterr().err
    assert "ZeroDivisionError" in err and "Traceback" in err
    assert main(["call", "structure.get", "--params", '{"models": "bad"}']) == 1


def test_cli_without_bridge(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(CONNECTION_ENV, str(tmp_path / "absent.json"))
    assert main(["ping"]) == 1
    assert "serve()" in capsys.readouterr().err
