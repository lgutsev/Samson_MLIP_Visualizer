"""Qt transport for the SAMSON bridge: a loopback TCP server on SAMSON's event loop.

Requests are read and answered in Qt slots, i.e. on SAMSON's main thread, which
is where SAMSON's Python API must be called.
"""

from __future__ import annotations

import atexit
import os
import secrets
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from .dispatcher import Dispatcher
from .protocol import (
    BridgeError,
    ConnectionInfo,
    LineBuffer,
    encode,
    remove_connection_file,
    write_connection_file,
)

_SERVER: BridgeServer | None = None


class BridgeServer:
    """Accept loopback connections and feed request lines to a dispatcher."""

    def __init__(
        self,
        dispatcher: Dispatcher,
        token: str,
        *,
        port: int = 0,
        connection_path: Path | None = None,
    ):
        from PySide6 import QtNetwork

        self.dispatcher = dispatcher
        self._token = token
        self._buffers: dict[Any, LineBuffer] = {}
        self._server = QtNetwork.QTcpServer()
        self._server.newConnection.connect(self._accept)
        address = QtNetwork.QHostAddress(QtNetwork.QHostAddress.SpecialAddress.LocalHost)
        if not self._server.listen(address, port):
            raise RuntimeError(
                f"The bridge could not listen on 127.0.0.1:{port}: {self._server.errorString()}"
            )
        self.port = int(self._server.serverPort())
        info = ConnectionInfo(
            host="127.0.0.1",
            port=self.port,
            token=token,
            pid=os.getpid(),
            version=__version__,
            allow_exec=dispatcher.allow_exec,
            started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        try:
            self.connection_path = write_connection_file(info, connection_path)
        except BaseException:
            self._server.close()
            raise

    def _accept(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if not socket.peerAddress().isLoopback():
                socket.abort()
                continue
            self._buffers[socket] = LineBuffer()
            socket.readyRead.connect(lambda s=socket: self._read(s))
            socket.disconnected.connect(lambda s=socket: self._drop(s))

    def _read(self, socket) -> None:
        buffer = self._buffers.get(socket)
        if buffer is None:
            return
        try:
            lines = buffer.feed(socket.readAll().data())
        except BridgeError as exc:
            socket.write(encode({"jsonrpc": "2.0", "id": None, "error": exc.to_dict()}))
            socket.disconnectFromHost()
            return
        for line in lines:
            socket.write(self.dispatcher.handle_line(line))
        socket.flush()

    def _drop(self, socket) -> None:
        self._buffers.pop(socket, None)
        socket.deleteLater()

    def close(self) -> None:
        for socket in list(self._buffers):
            socket.disconnectFromHost()
        self._buffers.clear()
        self._server.close()
        remove_connection_file(self.connection_path, token=self._token)


def panel_settings() -> dict[str, Any]:
    """Model settings from the open MLIP panel, used when a job omits them."""
    try:
        from .. import samson_app

        window = samson_app._WINDOW
        if window is None:
            return {}
        files = window._model_files()
        settings = {
            "model": files,
            "backend": window.backend.currentText().lower(),
            "device": window.device.currentText(),
            "dtype": window.dtype.currentText(),
        }
        options = window._program_options(settings["backend"])
        if options is not None:
            settings[f"{settings['backend']}_method"] = options.pop("method")
            settings.update(options)  # basis (Psi4), charge, multiplicity, solvent (xTB)
        return settings
    except Exception:  # noqa: BLE001 - no panel, or no model chosen yet
        return {}


def serve(
    port: int = 0,
    *,
    allow_exec: bool = False,
    log: Callable[[str], None] | None = print,
    samson: Any = None,
    connection_path: Path | None = None,
) -> BridgeServer:
    """Start the bridge inside SAMSON and return the server.

    Calling it again returns the running server, or restarts it when
    ``allow_exec`` changes. ``allow_exec=True`` lets any program that can read
    the connection file run Python inside SAMSON; leave it off unless needed.
    ``log`` receives one line per request (default: print to the console).
    """
    global _SERVER
    if _SERVER is not None:
        if _SERVER.dispatcher.allow_exec == allow_exec:
            return _SERVER
        stop()

    def on_request(method: str) -> None:
        if log is not None:
            log(f"[bridge] {method}")

    def schedule(function: Callable[[], None]) -> None:
        from PySide6.QtCore import QTimer

        # Run jobs from the event loop, after the reply that started them is sent.
        QTimer.singleShot(0, function)

    token = secrets.token_urlsafe(32)
    dispatcher = Dispatcher(
        token,
        samson=samson,
        allow_exec=allow_exec,
        on_request=on_request,
        schedule=schedule,
        job_defaults=panel_settings,
    )
    _SERVER = BridgeServer(dispatcher, token, port=port, connection_path=connection_path)
    if log is not None:
        mode = "Python execution ENABLED" if allow_exec else "fixed operations only"
        log(
            f"SAMSON bridge listening on 127.0.0.1:{_SERVER.port} ({mode}); "
            f"connection file {_SERVER.connection_path}"
        )
    return _SERVER


def stop() -> None:
    """Stop the bridge and remove its connection file."""
    global _SERVER
    if _SERVER is not None:
        _SERVER.close()
        _SERVER = None


def status() -> dict[str, Any] | None:
    """Port, options, and request count of the running bridge, or ``None``."""
    if _SERVER is None:
        return None
    dispatcher = _SERVER.dispatcher
    return {
        "port": _SERVER.port,
        "allow_exec": dispatcher.allow_exec,
        "requests": dispatcher.requests,
        "busy": dispatcher.busy,
        "connection_file": str(_SERVER.connection_path),
    }


def set_busy(reason: str | None) -> None:
    """Refuse document-changing requests while ``reason`` is set (e.g. a panel job)."""
    if _SERVER is not None:
        _SERVER.dispatcher.busy = reason


@atexit.register
def _remove_stale_connection_file() -> None:
    if _SERVER is not None:
        remove_connection_file(_SERVER.connection_path, token=_SERVER._token)
