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
    add_atom,
    atom_parent,
    choose_structural_models,
    element_type,
    extract_structure,
    is_effectively_selected,
    node_name,
    selected_atom_indices,
    set_selection_flag,
    sync_positions,
)
from .jobs import JobManager, JobSpecError
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

# Methods that change the document; refused while a panel or bridge job is running.
MUTATING = frozenset(
    {
        "structure.set_positions",
        "selection.set",
        "selection.select",
        "atoms.add",
        "atoms.delete",
        "atoms.set_elements",
        "atoms.set_fixed",
        "history.undo",
        "history.redo",
        "file.import",
        "command.run",
        "job.start",
        "python.exec",
    }
)
_MISSING = object()
_EFFECTIVE_LIST_LIMIT = 5000


def _int(params: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
    value = _param(params, name, int, default)
    if isinstance(value, bool) or not low <= value <= high:
        raise BridgeError(INVALID_PARAMS, f"{name!r} must be an integer in [{low}, {high}]")
    return value


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
        schedule: Callable[[Callable[[], None]], None] | None = None,
        job_defaults: Callable[[], dict[str, Any]] = dict,
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
        self.jobs = JobManager(
            lambda: self.samson,
            schedule=schedule,
            set_busy=lambda reason: setattr(self, "busy", reason),
            defaults=job_defaults,
        )
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
            "selection.select": (
                self._selection_select,
                "Select with a SAMSON NSL expression, e.g. 'node.type atom and atom.symbol O'. "
                "params: nsl.",
            ),
            "view.capture": (
                self._view_capture,
                "Save the 3D viewport as an image. params: path, width=1200, height=800, "
                "transparent=False.",
            ),
            "atoms.add": (
                self._atoms_add,
                "Add atoms (one undo step). params: atoms=[{symbol, position}], model?.",
            ),
            "atoms.delete": (
                self._atoms_delete,
                "Delete atoms by index over all models (one undo step). params: atoms.",
            ),
            "atoms.set_elements": (
                self._atoms_set_elements,
                "Change elements (one undo step). params: atoms, symbols (one or one per atom).",
            ),
            "atoms.set_fixed": (
                self._atoms_set_fixed,
                "Set or clear the fixed-atom flag (one undo step). params: atoms, fixed=True.",
            ),
            "history.undo": (self._history_undo, "Undo the last operation in SAMSON."),
            "history.redo": (self._history_redo, "Redo the last undone operation in SAMSON."),
            "job.start": (
                self._job_start,
                "Start an MLIP job; returns at once. params: kind (single_point|relax|md|ts|"
                "frequencies|irc|qst|scan), model?, backend?, device?, dtype?, models?, and kind "
                "options.",
            ),
            "qm.export": (
                self._qm_export,
                "Write a Gaussian (.gjf/.com) or ORCA (.inp) input from the selected models. "
                "params: path, job='auto' (1 model: ts, 2: qst2, 3: qst3) | ts | opt | irc, "
                "level?, charge=0, multiplicity=1.",
            ),
            "job.status": (
                self._job_status,
                "Progress, log, and result. params: id, log_lines=20.",
            ),
            "job.stop": (self._job_stop, "Ask a job to stop after its current step. params: id."),
            "job.list": (self._job_list, "All jobs of this bridge session."),
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
                    "effectively_selected_atoms": sum(
                        1 for atom in atoms if is_effectively_selected(atom)
                    ),
                    "unit_cell": _unit_cell(model) is not None,
                }
            )
        return {
            "models": entries,
            "atoms": sum(entry["atoms"] for entry in entries),
            "selected_atoms": sum(entry["selected_atoms"] for entry in entries),
            "effectively_selected_atoms": sum(
                entry["effectively_selected_atoms"] for entry in entries
            ),
            "note": "selected_atoms counts atoms picked individually; "
            "effectively_selected_atoms also counts atoms inside selected models.",
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
        selected_models, selected_atoms, effective, offset = [], [], [], 0
        for index, model in enumerate(self._models()):
            atoms = list(model.getNodes("node.type atom"))
            if _is_selected(model):
                selected_models.append(index)
            for i, atom in enumerate(atoms):
                if _is_selected(atom):
                    selected_atoms.append(offset + i)
                if is_effectively_selected(atom):
                    effective.append(offset + i)
            offset += len(atoms)
        reply = {
            "models": selected_models,
            "atoms": selected_atoms,
            "effective_atom_count": len(effective),
            "atom_count": offset,
        }
        if len(effective) <= _EFFECTIVE_LIST_LIMIT:
            reply["effective_atoms"] = effective
        return reply

    def _selection_select(self, params: dict[str, Any]) -> dict[str, Any]:
        nsl = _param(params, "nsl", str)
        select = getattr(self.samson, "select", None)
        if not callable(select):
            raise BridgeError(UNAVAILABLE, "This SAMSON build exposes no SAMSON.select")
        valid = bool(select(nsl))
        if not valid:
            raise BridgeError(INVALID_PARAMS, f"SAMSON rejected the NSL expression {nsl!r}")
        return {"nsl": nsl, **self._selection_get({})}

    def _view_capture(self, params: dict[str, Any]) -> dict[str, Any]:
        path = Path(_param(params, "path", str)).expanduser()
        width = _int(params, "width", 1200, 16, 8192)
        height = _int(params, "height", 800, 16, 8192)
        transparent = _param(params, "transparent", bool, False)
        capture = getattr(self.samson, "captureViewportToFile", None)
        if not callable(capture):
            raise BridgeError(UNAVAILABLE, "This SAMSON build exposes no captureViewportToFile")
        # No path tracing, no progress bar: a quick, non-interactive capture.
        capture(str(path), width, height, transparent, False, False)
        return {"path": str(path), "width": width, "height": height}

    def _all_atoms(self) -> list[Any]:
        return [atom for model in self._models() for atom in model.getNodes("node.type atom")]

    def _atoms_add(self, params: dict[str, Any]) -> dict[str, Any]:
        entries = _param(params, "atoms", list)
        if not entries:
            raise BridgeError(INVALID_PARAMS, "atoms must list at least one atom")
        parsed = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("symbol"), str):
                raise BridgeError(INVALID_PARAMS, "each atom needs {'symbol': str, 'position'}")
            try:
                position = np.asarray(entry.get("position"), dtype=float)
            except (TypeError, ValueError) as exc:
                raise BridgeError(INVALID_PARAMS, f"bad position: {exc}") from exc
            if position.shape != (3,) or not np.isfinite(position).all():
                raise BridgeError(INVALID_PARAMS, "position must be three finite numbers (Å)")
            element_type(entry["symbol"])  # validate before changing anything
            parsed.append((entry["symbol"], position))
        models = self._models()
        if "model" in params:
            model = models[_int(params, "model", 0, 0, max(len(models) - 1, 0))]
        else:
            chosen = choose_structural_models(self.samson)
            if len(chosen) != 1:
                raise BridgeError(INVALID_PARAMS, "Several models are selected; pass 'model'")
            model = chosen[0]
        parent = atom_parent(model)
        with self._holding(_param(params, "label", str, "Remote: add atoms")):
            for symbol, position in parsed:
                add_atom(parent, symbol, position)
        return {
            "added": len(parsed),
            "model": node_name(model),
            "atom_count": len(self._all_atoms()),
        }

    def _atoms_delete(self, params: dict[str, Any]) -> dict[str, Any]:
        atoms = self._all_atoms()
        indices = _indices(params, "atoms", len(atoms))
        if not indices:
            raise BridgeError(INVALID_PARAMS, "atoms must list at least one index")
        with self._holding(_param(params, "label", str, "Remote: delete atoms")):
            for index in sorted(set(indices)):
                atoms[index].erase()
        return {"deleted": len(set(indices)), "atom_count": len(self._all_atoms())}

    def _atoms_set_elements(self, params: dict[str, Any]) -> dict[str, Any]:
        atoms = self._all_atoms()
        indices = _indices(params, "atoms", len(atoms))
        symbols = params.get("symbols")
        if isinstance(symbols, str):
            symbols = [symbols] * len(indices)
        if not isinstance(symbols, list) or len(symbols) != len(indices):
            raise BridgeError(INVALID_PARAMS, "symbols must be one symbol or one per atom")
        types = [element_type(symbol) for symbol in symbols]
        with self._holding(_param(params, "label", str, "Remote: change elements")):
            for index, kind in zip(indices, types, strict=True):
                atoms[index].elementType = kind
        return {"changed": len(indices)}

    def _atoms_set_fixed(self, params: dict[str, Any]) -> dict[str, Any]:
        atoms = self._all_atoms()
        indices = _indices(params, "atoms", len(atoms))
        fixed = _param(params, "fixed", bool, True)
        with self._holding(_param(params, "label", str, "Remote: set fixed atoms")):
            for index in indices:
                atoms[index].fixedFlag = fixed
        return {"changed": len(indices), "fixed": fixed}

    def _history(self, name: str) -> dict[str, Any]:
        action = getattr(self.samson, name, None)
        if not callable(action):
            raise BridgeError(UNAVAILABLE, f"This SAMSON build exposes no SAMSON.{name}")
        action()
        return {name: True}

    def _history_undo(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._history("undo")

    def _history_redo(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._history("redo")

    def _qm_export(self, params: dict[str, Any]) -> dict[str, Any]:
        from ..qm_export import write_qm_input

        path = Path(_param(params, "path", str)).expanduser()
        job = _param(params, "job", str, "auto")
        models = choose_structural_models(self.samson)
        if job == "auto":
            jobs = {1: "ts", 2: "qst2", 3: "qst3"}
            if len(models) not in jobs:
                raise BridgeError(INVALID_PARAMS, "Select 1, 2, or 3 structural models")
            job = jobs[len(models)]
            structures = [extract_structure(self.samson, models=[m]).ase_atoms for m in models]
        else:
            structures = [extract_structure(self.samson, models=models).ase_atoms]
        try:
            written = write_qm_input(
                path,
                structures,
                job=job,
                level=_param(params, "level", str, None),
                charge=_int(params, "charge", 0, -50, 50),
                multiplicity=_int(params, "multiplicity", 1, 1, 50),
            )
        except ValueError as exc:
            raise BridgeError(INVALID_PARAMS, str(exc)) from exc
        return {"job": job, "files": [str(file) for file in written]}

    def _job_start(self, params: dict[str, Any]) -> dict[str, Any]:
        kind = _param(params, "kind", str)
        options = {key: value for key, value in params.items() if key != "kind"}
        try:
            return self.jobs.start(kind, options).to_dict()
        except JobSpecError as exc:
            raise BridgeError(INVALID_PARAMS, str(exc)) from exc

    def _job(self, params: dict[str, Any]):
        try:
            return self.jobs.get(_int(params, "id", 0, 1, 1 << 30))
        except JobSpecError as exc:
            raise BridgeError(INVALID_PARAMS, str(exc)) from exc

    def _job_status(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._job(params).to_dict(_int(params, "log_lines", 20, 0, 500))

    def _job_stop(self, params: dict[str, Any]) -> dict[str, Any]:
        job = self._job(params)
        return self.jobs.stop(job.id).to_dict()

    def _job_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [job.to_dict(log_lines=0) for job in self.jobs.jobs()]

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
