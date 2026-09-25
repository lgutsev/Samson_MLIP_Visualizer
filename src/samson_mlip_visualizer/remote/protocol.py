"""Wire protocol shared by the in-SAMSON bridge server and its clients.

Messages are JSON-RPC 2.0 objects, one per line (UTF-8, ``\\n``-terminated). Every
request carries the session token as a top-level ``"token"`` member. The server
writes its address and token to a connection file that only the current user can
read, the same arrangement Jupyter uses for kernels.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
CONNECTION_ENV = "SAMSON_BRIDGE_CONNECTION"

# JSON-RPC 2.0 error codes, plus implementation-defined ones in -32000..-32099.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SERVER_ERROR = -32000
UNAUTHORIZED = -32001
UNAVAILABLE = -32002
BUSY = -32003


class BridgeError(RuntimeError):
    """A JSON-RPC error returned by the bridge, or a failure to reach it."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def encode(message: dict[str, Any]) -> bytes:
    """Serialize one message as a single ``\\n``-terminated line."""
    text = json.dumps(message, default=_default, separators=(",", ":"), allow_nan=False)
    return text.encode("utf-8") + b"\n"


def decode(line: bytes) -> dict[str, Any]:
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError(PARSE_ERROR, f"Invalid JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise BridgeError(INVALID_REQUEST, "A message must be a JSON object")
    return message


class LineBuffer:
    """Accumulate socket bytes and return complete, non-empty lines."""

    def __init__(self, max_bytes: int = MAX_MESSAGE_BYTES):
        self._buffer = bytearray()
        self._max_bytes = max_bytes

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        lines = []
        while True:
            end = self._buffer.find(b"\n")
            if end < 0:
                break
            line = bytes(self._buffer[:end]).strip()
            del self._buffer[: end + 1]
            if line:
                lines.append(line)
        if len(self._buffer) > self._max_bytes:
            self._buffer.clear()
            raise BridgeError(INVALID_REQUEST, "Message exceeds the size limit")
        return lines


@dataclass(frozen=True)
class ConnectionInfo:
    host: str
    port: int
    token: str
    pid: int
    version: str
    allow_exec: bool
    started: str


def data_dir() -> Path:
    """Per-user directory for the connection file."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "samson-mlip-visualizer"


def connection_file() -> Path:
    override = os.environ.get(CONNECTION_ENV)
    return Path(override) if override else data_dir() / "bridge.json"


def write_connection_file(info: ConnectionInfo, path: Path | None = None) -> Path:
    """Write the connection file atomically, readable by the current user only."""
    path = Path(path) if path else connection_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".bridge-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(asdict(info), stream, indent=2)
        if sys.platform != "win32":
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def read_connection_file(path: Path | None = None) -> ConnectionInfo:
    path = Path(path) if path else connection_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BridgeError(
            UNAVAILABLE,
            f"No bridge connection file at {path}. Start the bridge in SAMSON's Python "
            "console: from samson_mlip_visualizer.remote import serve; serve()",
        ) from exc
    return ConnectionInfo(**data)


def remove_connection_file(path: Path | None = None, *, token: str | None = None) -> None:
    """Delete the connection file, but only if it still belongs to ``token``."""
    path = Path(path) if path else connection_file()
    try:
        if token is not None and read_connection_file(path).token != token:
            return
        path.unlink()
    except (BridgeError, OSError, TypeError, ValueError):
        pass
