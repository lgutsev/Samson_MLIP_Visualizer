"""The Qt transport with a real QTcpServer; skipped where PySide6 is not installed."""

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("PySide6.QtNetwork")

from samson_fakes import install_samson_module, water_document  # noqa: E402

from samson_mlip_visualizer.remote import qt_server  # noqa: E402
from samson_mlip_visualizer.remote.client import SamsonClient  # noqa: E402
from samson_mlip_visualizer.remote.protocol import BUSY, BridgeError  # noqa: E402


@pytest.fixture
def app():
    # A full QApplication, so widget tests can share the process.
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def call_while_spinning(function):
    """Run a blocking client call in a thread while this thread runs the Qt loop."""
    outcome = {}

    def target():
        try:
            outcome["value"] = function()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            outcome["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    deadline = time.monotonic() + 20
    while thread.is_alive() and time.monotonic() < deadline:
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
    thread.join(timeout=1)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def test_qt_bridge_serves_requests(app, monkeypatch, tmp_path):
    samson = water_document()
    install_samson_module(monkeypatch, samson)
    log = []
    path = tmp_path / "bridge.json"
    server = qt_server.serve(log=log.append, samson=samson, connection_path=path)
    try:
        assert path.exists()
        assert qt_server.serve(log=log.append, samson=samson, connection_path=path) is server
        client = SamsonClient.from_connection_file(path, timeout=15)

        assert call_while_spinning(client.ping)["pong"] is True
        atoms = call_while_spinning(client.get_structure)
        atoms.positions[0] += [0.2, 0, 0]
        call_while_spinning(lambda: client.set_positions(atoms))
        assert samson.models[0].atoms[0].position[0] == pytest.approx(0.2)

        qt_server.set_busy("test job")
        with pytest.raises(BridgeError) as busy:
            call_while_spinning(lambda: client.set_positions(atoms))
        assert busy.value.code == BUSY
        qt_server.set_busy(None)

        status = qt_server.status()
        assert status["port"] == server.port
        assert status["requests"] == 3
        assert "[bridge] bridge.ping" in log
    finally:
        qt_server.stop()
    assert not path.exists()
    assert qt_server.status() is None
