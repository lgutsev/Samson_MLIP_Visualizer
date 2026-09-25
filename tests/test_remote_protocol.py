import os
import sys

import numpy as np
import pytest

from samson_mlip_visualizer.remote.protocol import (
    CONNECTION_ENV,
    INVALID_REQUEST,
    PARSE_ERROR,
    UNAVAILABLE,
    BridgeError,
    ConnectionInfo,
    LineBuffer,
    connection_file,
    decode,
    encode,
    read_connection_file,
    remove_connection_file,
    write_connection_file,
)


def info(token="secret"):
    return ConnectionInfo(
        host="127.0.0.1",
        port=1234,
        token=token,
        pid=42,
        version="0.1.0",
        allow_exec=False,
        started="2026-09-25T00:00:00+00:00",
    )


def test_encode_is_one_line_and_handles_numpy():
    line = encode({"result": {"positions": np.eye(2), "count": np.int64(3), "pbc": (True,)}})
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    expected = {"positions": [[1.0, 0.0], [0.0, 1.0]], "count": 3, "pbc": [True]}
    assert decode(line) == {"result": expected}


def test_encode_rejects_nan():
    with pytest.raises(ValueError):
        encode({"result": float("nan")})


def test_decode_errors():
    with pytest.raises(BridgeError) as parse:
        decode(b"{not json")
    assert parse.value.code == PARSE_ERROR
    with pytest.raises(BridgeError) as shape:
        decode(b"[1, 2]")
    assert shape.value.code == INVALID_REQUEST


def test_line_buffer_splits_and_keeps_partial_lines():
    buffer = LineBuffer()
    assert buffer.feed(b'{"a":1}\n\n{"b"') == [b'{"a":1}']
    assert buffer.feed(b":2}\n") == [b'{"b":2}']
    assert buffer.feed(b"") == []


def test_line_buffer_limits_message_size():
    buffer = LineBuffer(max_bytes=8)
    with pytest.raises(BridgeError) as error:
        buffer.feed(b"0123456789")
    assert error.value.code == INVALID_REQUEST
    assert buffer.feed(b"ok\n") == [b"ok"]


def test_connection_file_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "bridge.json"
    monkeypatch.setenv(CONNECTION_ENV, str(path))
    assert connection_file() == path

    written = write_connection_file(info())
    assert written == path
    assert read_connection_file() == info()
    assert not list(path.parent.glob(".bridge-*"))
    if sys.platform != "win32":
        assert oct(os.stat(path).st_mode & 0o777) == oct(0o600)


def test_remove_connection_file_respects_owner(tmp_path):
    path = tmp_path / "bridge.json"
    write_connection_file(info("mine"), path)
    remove_connection_file(path, token="someone else")
    assert path.exists()
    remove_connection_file(path, token="mine")
    assert not path.exists()
    remove_connection_file(path, token="mine")  # already gone: no error


def test_missing_connection_file_explains_how_to_start(tmp_path):
    with pytest.raises(BridgeError, match="serve") as error:
        read_connection_file(tmp_path / "absent.json")
    assert error.value.code == UNAVAILABLE
