"""Local bridge that lets other programs on this computer drive a running SAMSON.

Inside SAMSON's Python console::

    from samson_mlip_visualizer.remote import serve
    serve()                   # fixed operations only
    serve(allow_exec=True)    # also accept Python code (only if you need it)

Anywhere else on the same machine::

    from samson_mlip_visualizer.remote import SamsonClient
    client = SamsonClient.from_connection_file()
    atoms = client.get_structure()

or ``samson-remote summary`` on the command line. The server listens on
127.0.0.1 only and every request must carry the token from a connection file
that only the current user can read.
"""

from .client import SamsonClient
from .protocol import BridgeError


def serve(*args, **kwargs):
    """Start the bridge inside SAMSON; see :func:`qt_server.serve`."""
    from .qt_server import serve as _serve

    return _serve(*args, **kwargs)


def stop() -> None:
    """Stop the bridge and remove its connection file."""
    from .qt_server import stop as _stop

    _stop()


def status():
    """Port, options, and request count of the running bridge, or ``None``."""
    from .qt_server import status as _status

    return _status()


__all__ = ["BridgeError", "SamsonClient", "serve", "status", "stop"]
