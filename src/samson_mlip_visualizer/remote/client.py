"""Client for the SAMSON bridge, and the ``samson-remote`` command line tool.

Runs outside SAMSON: it needs only the connection file that ``serve()`` writes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import (
    UNAVAILABLE,
    BridgeError,
    LineBuffer,
    decode,
    encode,
    read_connection_file,
)

_START_HINT = (
    "Start the bridge in SAMSON's Python console: "
    "from samson_mlip_visualizer.remote import serve; serve()"
)


class SamsonClient:
    """Call bridge methods on a running SAMSON. One TCP connection per call."""

    def __init__(self, host: str, port: int, token: str, *, timeout: float = 60.0):
        self._host = host
        self._port = port
        self._token = token
        self.timeout = timeout
        self._ids = itertools.count(1)

    @classmethod
    def from_connection_file(
        cls, path: str | Path | None = None, *, timeout: float = 60.0
    ) -> SamsonClient:
        info = read_connection_file(Path(path) if path else None)
        return cls(info.host, info.port, info.token, timeout=timeout)

    def call(self, method: str, **params: Any) -> Any:
        """Call ``method`` and return its result; raise :class:`BridgeError` on failure."""
        request = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": method,
            "params": params,
            "token": self._token,
        }
        try:
            with socket.create_connection((self._host, self._port), self.timeout) as connection:
                connection.sendall(encode(request))
                buffer = LineBuffer()
                lines: list[bytes] = []
                while not lines:
                    chunk = connection.recv(1 << 16)
                    if not chunk:
                        raise BridgeError(UNAVAILABLE, "The bridge closed the connection")
                    lines = buffer.feed(chunk)
        except ConnectionRefusedError as exc:
            raise BridgeError(
                UNAVAILABLE,
                f"Nothing is listening on {self._host}:{self._port}, so the bridge is not "
                f"running (the connection file may be stale). {_START_HINT}",
            ) from exc
        except TimeoutError as exc:
            raise BridgeError(
                UNAVAILABLE, f"No reply within {self.timeout:g} s; SAMSON may be busy"
            ) from exc
        except OSError as exc:
            raise BridgeError(UNAVAILABLE, f"Could not reach the bridge: {exc}") from exc
        response = decode(lines[0])
        error = response.get("error")
        if error:
            raise BridgeError(
                error.get("code", UNAVAILABLE),
                error.get("message", "Unknown error"),
                error.get("data"),
            )
        return response.get("result")

    # --- conveniences ----------------------------------------------------------------

    def ping(self) -> dict[str, Any]:
        return self.call("bridge.ping")

    def info(self) -> dict[str, Any]:
        return self.call("bridge.info")

    def summary(self) -> dict[str, Any]:
        return self.call("document.summary")

    def get_structure(self, models: str = "auto"):
        """The structure SAMSON would evaluate, as ASE ``Atoms`` (with FixAtoms)."""
        from ase import Atoms
        from ase.constraints import FixAtoms

        data = self.call("structure.get", models=models)
        atoms = Atoms(
            symbols=data["symbols"],
            positions=data["positions"],
            cell=data["cell"],
            pbc=data["pbc"],
        )
        if data["fixed"]:
            atoms.set_constraint(FixAtoms(indices=data["fixed"]))
        return atoms

    def set_positions(self, atoms_or_positions: Any, *, models: str = "auto", label: str = ""):
        """Write positions back to SAMSON (one undo step).

        Passing ASE ``Atoms`` also checks that the element order still matches.
        """
        params: dict[str, Any] = {"models": models}
        if hasattr(atoms_or_positions, "get_positions"):
            params["symbols"] = atoms_or_positions.get_chemical_symbols()
            positions = atoms_or_positions.get_positions()
        else:
            positions = atoms_or_positions
        params["positions"] = np.asarray(positions, dtype=float).tolist()
        if label:
            params["label"] = label
        return self.call("structure.set_positions", **params)

    def get_selection(self) -> dict[str, Any]:
        return self.call("selection.get")

    def set_selection(
        self, *, atoms: Sequence[int] = (), models: Sequence[int] = (), clear: bool = True
    ) -> dict[str, Any]:
        return self.call(
            "selection.set", atoms=list(atoms), models=list(models), clear=clear
        )

    def import_file(self, path: str | Path) -> dict[str, Any]:
        return self.call("file.import", path=str(Path(path).expanduser().resolve()))

    def export(
        self, path: str | Path, *, models: str = "auto", format: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"path": str(Path(path).expanduser().resolve()), "models": models}
        if format:
            params["format"] = format
        return self.call("file.export", **params)

    def run_command(self, name: str) -> dict[str, Any]:
        return self.call("command.run", name=name)

    def execute(self, code: str) -> dict[str, Any]:
        """Run Python inside SAMSON (only if the bridge allows it)."""
        return self.call("python.exec", code=code)

    def select(self, nsl: str) -> dict[str, Any]:
        """Select with a SAMSON NSL expression; returns the new selection."""
        return self.call("selection.select", nsl=nsl)

    def capture(self, path: str | Path, width: int = 1200, height: int = 800) -> dict[str, Any]:
        """Save the 3D viewport as an image (format from the extension)."""
        return self.call(
            "view.capture", path=str(Path(path).expanduser().resolve()), width=width, height=height
        )

    def add_atoms(self, atoms: Sequence[tuple[str, Sequence[float]]], model: int | None = None):
        """Add ``[(symbol, (x, y, z)), ...]`` (Å) as one undo step."""
        params: dict[str, Any] = {
            "atoms": [
                {"symbol": symbol, "position": [float(v) for v in position]}
                for symbol, position in atoms
            ]
        }
        if model is not None:
            params["model"] = model
        return self.call("atoms.add", **params)

    def delete_atoms(self, indices: Sequence[int]) -> dict[str, Any]:
        return self.call("atoms.delete", atoms=list(indices))

    def set_elements(self, indices: Sequence[int], symbols: str | Sequence[str]):
        symbols = symbols if isinstance(symbols, str) else list(symbols)
        return self.call("atoms.set_elements", atoms=list(indices), symbols=symbols)

    def set_fixed(self, indices: Sequence[int], fixed: bool = True) -> dict[str, Any]:
        return self.call("atoms.set_fixed", atoms=list(indices), fixed=fixed)

    def undo(self) -> dict[str, Any]:
        return self.call("history.undo")

    def redo(self) -> dict[str, Any]:
        return self.call("history.redo")

    def start_job(self, kind: str, **options: Any) -> dict[str, Any]:
        """Start an MLIP job (single_point, relax, md, ts, frequencies); returns at once."""
        return self.call("job.start", kind=kind, **options)

    def job_status(self, job_id: int, log_lines: int = 20) -> dict[str, Any]:
        return self.call("job.status", id=job_id, log_lines=log_lines)

    def stop_job(self, job_id: int) -> dict[str, Any]:
        return self.call("job.stop", id=job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.call("job.list")

    def wait_job(
        self, job_id: int, *, poll: float = 1.0, timeout: float | None = None, log_lines: int = 20
    ) -> dict[str, Any]:
        """Poll until the job finishes, stops, or fails; return its final status."""
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.job_status(job_id, log_lines=log_lines)
            if status["state"] not in ("queued", "running"):
                return status
            if deadline is not None and time.monotonic() > deadline:
                return status
            time.sleep(poll)


# --- command line ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samson-remote", description="Talk to the bridge running inside SAMSON."
    )
    parser.add_argument("--connection", type=Path, default=None, help="Connection file")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ping", help="Check that SAMSON answers")
    commands.add_parser("info", help="Versions, options, and methods")
    commands.add_parser("summary", help="Structural models and selection")
    get = commands.add_parser("get", help="Print or save the structure")
    get.add_argument("-o", "--output", type=Path, help="Write with ASE instead of printing")
    get.add_argument("--all", action="store_true", help="All models, not the selection")
    put = commands.add_parser("set-positions", help="Copy positions from a structure file")
    put.add_argument("structure", type=Path)
    put.add_argument("--all", action="store_true", help="All models, not the selection")
    select = commands.add_parser("select", help="Select atoms/models by index")
    select.add_argument("--atoms", type=int, nargs="*", default=[])
    select.add_argument("--models", type=int, nargs="*", default=[])
    select.add_argument("--add", action="store_true", help="Keep the current selection")
    commands.add_parser("selection", help="Show the current selection")
    load = commands.add_parser("import", help="Import a file into the document")
    load.add_argument("path", type=Path)
    save = commands.add_parser("export", help="Write the structure with ASE")
    save.add_argument("path", type=Path)
    save.add_argument("--format", default=None)
    save.add_argument("--all", action="store_true", help="All models, not the selection")
    command = commands.add_parser("command", help="Run a SAMSON command by name")
    command.add_argument("name")
    run = commands.add_parser("exec", help="Run Python in SAMSON (if allowed)")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("code", nargs="?")
    source.add_argument("-f", "--file", type=Path)
    nsl = commands.add_parser("nsl", help="Select with a SAMSON NSL expression")
    nsl.add_argument("expression")
    shot = commands.add_parser("capture", help="Save the 3D viewport as an image")
    shot.add_argument("path", type=Path)
    shot.add_argument("--width", type=int, default=1200)
    shot.add_argument("--height", type=int, default=800)
    job = commands.add_parser("job", help="Start or follow MLIP jobs")
    job.add_argument("action", choices=["start", "status", "stop", "wait", "list"])
    job.add_argument("target", nargs="?", help="Job kind for start, job id otherwise")
    job.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Job option, e.g. --set fmax=0.01 --set optimizer=LBFGS (values parsed as JSON)",
    )
    call = commands.add_parser("call", help="Call any method with JSON params")
    call.add_argument("method")
    call.add_argument("--params", default="{}", help="JSON object")
    return parser


def _parse_options(pairs: Sequence[str]) -> dict[str, Any]:
    options = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise SystemExit(f"--set expects KEY=VALUE, not {pair!r}")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value  # bare strings such as optimizer=LBFGS
    return options


def _run_job_command(args: argparse.Namespace, client: SamsonClient) -> Any:
    if args.action == "list":
        return client.list_jobs()
    if args.target is None:
        raise SystemExit(f"job {args.action} needs a {'kind' if args.action == 'start' else 'id'}")
    if args.action == "start":
        return client.start_job(args.target, **_parse_options(args.set))
    job_id = int(args.target)
    if args.action == "status":
        return client.job_status(job_id)
    if args.action == "stop":
        return client.stop_job(job_id)
    return client.wait_job(job_id)


def _run(args: argparse.Namespace, client: SamsonClient) -> Any:
    models = "all" if getattr(args, "all", False) else "auto"
    if args.command == "ping":
        return client.ping()
    if args.command == "info":
        return client.info()
    if args.command == "summary":
        return client.summary()
    if args.command == "get":
        if args.output is None:
            return client.call("structure.get", models=models)
        from ase.io import write

        atoms = client.get_structure(models)
        write(args.output, atoms)
        return {"written": str(args.output), "atoms": len(atoms)}
    if args.command == "set-positions":
        from ase.io import read

        return client.set_positions(read(args.structure), models=models)
    if args.command == "select":
        return client.set_selection(atoms=args.atoms, models=args.models, clear=not args.add)
    if args.command == "selection":
        return client.get_selection()
    if args.command == "import":
        return client.import_file(args.path)
    if args.command == "export":
        return client.export(args.path, models=models, format=args.format)
    if args.command == "command":
        return client.run_command(args.name)
    if args.command == "exec":
        code = args.file.read_text(encoding="utf-8") if args.file else args.code
        return client.execute(code)
    if args.command == "nsl":
        return client.select(args.expression)
    if args.command == "capture":
        return client.capture(args.path, args.width, args.height)
    if args.command == "job":
        return _run_job_command(args, client)
    params = json.loads(args.params)
    if not isinstance(params, dict):
        raise SystemExit("--params must be a JSON object")
    return client.call(args.method, **params)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        client = SamsonClient.from_connection_file(args.connection, timeout=args.timeout)
        result = _run(args, client)
    except BridgeError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if isinstance(exc.data, dict):
            for key in ("stdout", "stderr", "traceback"):
                if exc.data.get(key):
                    print(exc.data[key], file=sys.stderr, end="")
        return 1
    if args.command == "exec":
        sys.stdout.write(result["stdout"])
        sys.stderr.write(result["stderr"])
        if result["result"] is not None:
            print(result["result"])
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
