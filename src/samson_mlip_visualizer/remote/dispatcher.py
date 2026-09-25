"""Request handling for the SAMSON bridge: authentication, routing, operations.

The dispatcher is transport-independent: the Qt server inside SAMSON (and the
tests) hand it one request line at a time and send back the reply line. The
operations call SAMSON's Python API, so they must run on SAMSON's main thread,
which the Qt server guarantees.
"""

from __future__ import annotations

import ast
import contextlib
import hmac
import io
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ase.constraints import FixAtoms

from .. import __version__
from ..samson_bridge import (
    SamsonBridgeError,
    _is_selected,
    _unit_cell,
    extract_structure,
    node_name,
    selected_atom_indices,
    set_selection_flag,
    sync_positions,
)
from .protocol import (
    BUSY,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    SERVER_ERROR,
    UNAUTHORIZED,
    UNAVAILABLE,
    BridgeError,
    decode,
    encode,
)

# Methods that change the document; refused while a panel job is running.
MUTATING = frozenset(
    {"structure.set_positions", "selection.set", "file.import", "command.run", "python.exec"}
)
_MISSING = object()


def _param(params: dict[str, Any], name: str, kind: type, default: Any = _MISSING) -> Any:
    if name not in params:
        if default is _MISSING:
            raise BridgeError(INVALID_PARAMS, f"Missing parameter {name!r}")
        return default
    value = params[name]
    if not isinstance(value, kind):
        raise BridgeError(INVALID_PARAMS, f"Parameter {name!r} must be of type {kind.__name__}")
    return value


def _indices(params: dict[str, Any], name: str, size: int) -> list[int]:
    values = _param(params, name, list, [])
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
            raise BridgeError(INVALID_PARAMS, f"{name!r} must hold indices in [0, {size})")
    return values


def _plain(value: Any) -> Any:
    """Pass JSON scalars through; describe anything else."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


class Dispatcher:
    """Authenticate requests and route them to operations on SAMSON."""

    def __init__(
        self,
        token: str,
        *,
        samson: Any = None,
        allow_exec: bool = False,
        on_request: Callable[[str], None] | None = None,
    ):
        if not token:
            raise ValueError("A non-empty token is required")
        self._token = token.encode("utf-8")
        self._samson = samson
        self.allow_exec = allow_exec
        self.on_request = on_request
        self.busy: str | None = None
        self.requests = 0
        self._namespace: dict[str, Any] | None = None
        self._methods: dict[str, tuple[Callable[[dict[str, Any]], Any], str]] = {
            "bridge.ping": (self._ping, "Liveness check and package version."),
            "bridge.info": (self._info, "Versions, options, request count, and methods."),
            "document.summary": (self._summary, "Structural models, atom counts, selection."),
            "structure.get": (
                self._structure_get,
                "Symbols, positions (Å), cell, PBC, fixed atoms. params: models='auto'|'all'.",
            ),
            "structure.set_positions": (
                self._structure_set_positions,
                "Write positions (Å) as one undo step. params: positions, symbols?, models?, "
                "label?.",
            ),
            "selection.get": (
                self._selection_get,
                "Selected models, and selected atoms as indices over all models.",
            ),
            "selection.set": (
                self._selection_set,
                "Select by index. params: atoms?, models?, clear=True.",
            ),
            "file.import": (self._file_import, "Import a file into the document. params: path."),
            "file.export": (
                self._file_export,
                "Write the structure with ASE. params: path, format?, models?.",
            ),
            "command.run": (
                self._command_run,
                "Run a SAMSON command by its interface name. params: name.",
            ),
        }
        if allow_exec:
            self._methods["python.exec"] = (
                self._python_exec,
                "Run Python in SAMSON; the last expression is returned. params: code.",
            )

    @property
    def samson(self) -> Any:
        if self._samson is None:
            from samson import SAMSON

            self._samson = SAMSON
        return self._samson

    # --- request handling ------------------------------------------------------------

    def handle_line(self, line: bytes) -> bytes:
        """Answer one request line with one response line."""
        request_id = None
        try:
            request = decode(line)
            request_id = request.get("id")
            response = {"jsonrpc": "2.0", "id": request_id, "result": self._dispatch(request)}
        except BridgeError as exc:
            response = {"jsonrpc": "2.0", "id": request_id, "error": exc.to_dict()}
        except Exception as exc:  # noqa: BLE001 - report operation failures to the caller
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": SERVER_ERROR,
                    "message": str(exc) or type(exc).__name__,
                    "data": {"type": type(exc).__name__, "traceback": traceback.format_exc()},
                },
            }
        try:
            return encode(response)
        except (TypeError, ValueError) as exc:
            error = {"code": INTERNAL_ERROR, "message": f"The result could not be encoded: {exc}"}
            return encode({"jsonrpc": "2.0", "id": request_id, "error": error})

    def _dispatch(self, request: dict[str, Any]) -> Any:
        token = request.get("token")
        if not isinstance(token, str) or not hmac.compare_digest(
            token.encode("utf-8"), self._token
        ):
            raise BridgeError(UNAUTHORIZED, "Missing or invalid bridge token")
        method = request.get("method")
        if not isinstance(method, str):
            raise BridgeError(INVALID_REQUEST, "The request has no method name")
        entry = self._methods.get(method)
        if entry is None:
            hint = ""
            if method == "python.exec":
                hint = " Python execution is off; start the bridge with allow_exec=True."
            raise BridgeError(METHOD_NOT_FOUND, f"Unknown method {method!r}.{hint}")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            raise BridgeError(INVALID_PARAMS, "params must be a JSON object")
        if method in MUTATING and self.busy:
            raise BridgeError(BUSY, f"SAMSON is busy ({self.busy}); try again when it finishes")
        self.requests += 1
        if self.on_request is not None:
            with contextlib.suppress(Exception):
                self.on_request(method)
        return entry[0](params)

    # --- helpers ---------------------------------------------------------------------

    def _holding(self, label: str):
        holding = getattr(self.samson, "holding", None)
        return holding(label) if callable(holding) else contextlib.nullcontext()

    def _models(self) -> list[Any]:
        return list(self.samson.getNodes("node.type structuralModel"))

    def _structure(self, params: dict[str, Any]):
        which = _param(params, "models", str, "auto")
        if which == "auto":
            return extract_structure(self.samson)
        if which == "all":
            models = self._models()
            if not models:
                raise SamsonBridgeError("The active SAMSON document contains no structural model")
            return extract_structure(self.samson, models=models)
        raise BridgeError(INVALID_PARAMS, "models must be 'auto' or 'all'")

    # --- operations ------------------------------------------------------------------

    def _ping(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True, "version": __version__}

    def _info(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": __version__,
            "pid": os.getpid(),
            "allow_exec": self.allow_exec,
            "busy": self.busy,
            "requests": self.requests,
            "methods": {name: doc for name, (_, doc) in sorted(self._methods.items())},
        }

    def _summary(self, params: dict[str, Any]) -> dict[str, Any]:
        entries = []
        for index, model in enumerate(self._models()):
            atoms = list(model.getNodes("node.type atom"))
            entries.append(
                {
                    "index": index,
                    "name": node_name(model),
                    "atoms": len(atoms),
                    "selected": _is_selected(model),
                    "selected_atoms": sum(1 for atom in atoms if _is_selected(atom)),
                    "unit_cell": _unit_cell(model) is not None,
                }
            )
        return {
            "models": entries,
            "atoms": sum(entry["atoms"] for entry in entries),
            "selected_atoms": sum(entry["selected_atoms"] for entry in entries),
        }

    def _structure_get(self, params: dict[str, Any]) -> dict[str, Any]:
        structure = self._structure(params)
        atoms = structure.ase_atoms
        fixed = sorted(
            {
                int(index)
                for constraint in atoms.constraints
                if isinstance(constraint, FixAtoms)
                for index in constraint.get_indices()
            }
        )
        return {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions(),
            "cell": atoms.cell.array if atoms.cell.rank else None,
            "pbc": atoms.pbc,
            "fixed": fixed,
            "models": [node_name(model) for model in structure.models],
            "selected_atoms": selected_atom_indices(structure),
        }

    def _structure_set_positions(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            positions = np.asarray(_param(params, "positions", list), dtype=float)
        except ValueError as exc:
            raise BridgeError(INVALID_PARAMS, f"positions must be [[x, y, z], ...]: {exc}") from exc
        structure = self._structure(params)
        count = len(structure.samson_atoms)
        if positions.shape != (count, 3):
            raise BridgeError(
                INVALID_PARAMS, f"positions must have shape ({count}, 3), not {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise BridgeError(INVALID_PARAMS, "positions must be finite")
        symbols = _param(params, "symbols", list, None)
        if symbols is not None and symbols != structure.ase_atoms.get_chemical_symbols():
            raise BridgeError(
                INVALID_PARAMS,
                "symbols do not match the structure in SAMSON; the selection or the "
                "document changed since the structure was read",
            )
        with self._holding(_param(params, "label", str, "Remote: set positions")):
            sync_positions(structure, positions, samson=self.samson, process_events=False)
        return {"atoms": count}

    def _selection_get(self, params: dict[str, Any]) -> dict[str, Any]:
        selected_models, selected_atoms, offset = [], [], 0
        for index, model in enumerate(self._models()):
            atoms = list(model.getNodes("node.type atom"))
            if _is_selected(model):
                selected_models.append(index)
            selected_atoms.extend(offset + i for i, atom in enumerate(atoms) if _is_selected(atom))
            offset += len(atoms)
        return {"models": selected_models, "atoms": selected_atoms, "atom_count": offset}

    def _selection_set(self, params: dict[str, Any]) -> dict[str, Any]:
        models = self._models()
        atoms = [atom for model in models for atom in model.getNodes("node.type atom")]
        model_indices = _indices(params, "models", len(models))
        atom_indices = _indices(params, "atoms", len(atoms))
        if _param(params, "clear", bool, True):
            for node in (*models, *atoms):
                if _is_selected(node):
                    set_selection_flag(node, False)
        for index in model_indices:
            set_selection_flag(models[index], True)
        for index in atom_indices:
            set_selection_flag(atoms[index], True)
        return {"models": model_indices, "atoms": atom_indices}

    def _file_import(self, params: dict[str, Any]) -> dict[str, Any]:
        path = Path(_param(params, "path", str)).expanduser()
        if not path.is_file():
            raise BridgeError(INVALID_PARAMS, f"No such file: {path}")
        importer = getattr(self.samson, "importFromFile", None)
        if not callable(importer):
            raise BridgeError(UNAVAILABLE, "This SAMSON build exposes no SAMSON.importFromFile")
        return {"path": str(path), "result": _plain(importer(str(path)))}

    def _file_export(self, params: dict[str, Any]) -> dict[str, Any]:
        from ase.io import write

        path = Path(_param(params, "path", str)).expanduser()
        structure = self._structure(params)
        write(path, structure.ase_atoms, format=_param(params, "format", str, None))
        return {"path": str(path), "atoms": len(structure.ase_atoms)}

    def _command_run(self, params: dict[str, Any]) -> dict[str, Any]:
        name = _param(params, "name", str)
        runner = getattr(self.samson, "runCommand", None)
        if not callable(runner):
            raise BridgeError(UNAVAILABLE, "This SAMSON build exposes no SAMSON.runCommand")
        return {"name": name, "result": _plain(runner(name))}

    def _python_exec(self, params: dict[str, Any]) -> dict[str, Any]:
        code = _param(params, "code", str)
        if self._namespace is None:
            self._namespace = {"__name__": "__samson_bridge__", "SAMSON": self.samson, "np": np}
            with contextlib.suppress(ImportError):
                import samson

                self._namespace["samson"] = samson
        try:
            tree = ast.parse(code, "<bridge>", "exec")
        except SyntaxError as exc:
            raise BridgeError(INVALID_PARAMS, f"SyntaxError: {exc}") from exc
        final = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            final = ast.Expression(tree.body.pop().value)
        stdout, stderr = io.StringIO(), io.StringIO()
        value = None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(tree, "<bridge>", "exec"), self._namespace)
                if final is not None:
                    value = eval(compile(final, "<bridge>", "eval"), self._namespace)
        except (Exception, SystemExit) as exc:
            raise BridgeError(
                SERVER_ERROR,
                f"{type(exc).__name__}: {exc}",
                {
                    "traceback": traceback.format_exc(),
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                },
            ) from exc
        return {
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": None if value is None else repr(value),
        }
