"""MCP server that exposes the SAMSON bridge as tools for coding assistants.

Speaks the Model Context Protocol over stdio (one JSON-RPC message per line) and
forwards each tool call to the bridge running inside SAMSON. It holds no state:
every call reads the bridge's connection file, so restarting the bridge (new
port and token) needs no restart here. Run with
``python -m samson_mlip_visualizer.remote.mcp_server`` or ``samson-mcp``.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .. import __version__
from .client import SamsonClient
from .protocol import BridgeError

PROTOCOL_VERSION = "2025-06-18"
INSTRUCTIONS = (
    "Tools act on the document open in the user's running SAMSON (molecular modeling). "
    "Positions are in Å. Atom indices in selection and atom-editing tools count over all "
    "structural models in document order. Edits are single undo steps; MLIP jobs run "
    "asynchronously: start one, then poll samson_job_status. "
    "Transition states, in this order: (1) relax the end points (float64, tight fmax); "
    "(2) find the TS: ts (P-RFO/dimer) from a guess, qst from two minima, or scan from a "
    "forming/breaking bond; (3) characterize it with frequencies: exactly one imaginary "
    "mode, moving the right atoms; (4) confirm connectivity with irc (or check_irc=true), "
    "relaxing both ends to the intended minima; (5) optionally samson_export_qm. A TS is "
    "not confirmed until an IRC reaches both minima: QST's final P-RFO can slide to "
    "another saddle, and a scan's frames are constrained snapshots (check its 'jumps') "
    "whose highest point can be an artifact."
)

_INDICES = {"type": "array", "items": {"type": "integer", "minimum": 0}}
_MODELS = {
    "type": "string",
    "enum": ["auto", "all"],
    "description": "'auto': the selected models (or the only one); 'all': every model",
}


def _schema(properties: dict[str, Any] | None = None, required: list[str] | None = None):
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return schema


# name -> (bridge method, description, input schema)
_FORWARDED: dict[str, tuple[str, str, dict[str, Any]]] = {
    "samson_status": ("bridge.info", "Bridge version, options, busy state, methods.", _schema()),
    "samson_summary": (
        "document.summary",
        "Structural models in the open document with atom counts and selection.",
        _schema(),
    ),
    "samson_get_structure": (
        "structure.get",
        "Element symbols, positions (Å), cell, periodicity, and fixed atoms.",
        _schema({"models": _MODELS}),
    ),
    "samson_set_positions": (
        "structure.set_positions",
        "Replace all positions (Å) of the structure, as one undo step. Pass the symbols "
        "from samson_get_structure so a changed document is detected.",
        _schema(
            {
                "positions": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "symbols": {"type": "array", "items": {"type": "string"}},
                "models": _MODELS,
                "label": {"type": "string"},
            },
            ["positions"],
        ),
    ),
    "samson_get_selection": (
        "selection.get",
        "Selected models, individually selected atoms, and effectively selected atoms.",
        _schema(),
    ),
    "samson_select": (
        "selection.select",
        "Select with a SAMSON NSL expression, e.g. 'node.type atom and atom.symbol O'.",
        _schema({"nsl": {"type": "string"}}, ["nsl"]),
    ),
    "samson_set_selection": (
        "selection.set",
        "Select models and atoms by index.",
        _schema(
            {"atoms": _INDICES, "models": _INDICES, "clear": {"type": "boolean", "default": True}}
        ),
    ),
    "samson_add_atoms": (
        "atoms.add",
        "Add atoms to a structural model (one undo step).",
        _schema(
            {
                "atoms": {
                    "type": "array",
                    "items": _schema(
                        {
                            "symbol": {"type": "string"},
                            "position": {"type": "array", "items": {"type": "number"}},
                        },
                        ["symbol", "position"],
                    ),
                },
                "model": {"type": "integer", "minimum": 0},
            },
            ["atoms"],
        ),
    ),
    "samson_delete_atoms": (
        "atoms.delete",
        "Delete atoms by index (one undo step).",
        _schema({"atoms": _INDICES}, ["atoms"]),
    ),
    "samson_set_elements": (
        "atoms.set_elements",
        "Change the element of atoms (one undo step).",
        _schema(
            {
                "atoms": _INDICES,
                "symbols": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
            },
            ["atoms", "symbols"],
        ),
    ),
    "samson_set_fixed": (
        "atoms.set_fixed",
        "Set or clear SAMSON's fixed-atom flag (one undo step).",
        _schema({"atoms": _INDICES, "fixed": {"type": "boolean", "default": True}}, ["atoms"]),
    ),
    "samson_undo": ("history.undo", "Undo the last operation in SAMSON.", _schema()),
    "samson_redo": ("history.redo", "Redo the last undone operation in SAMSON.", _schema()),
    "samson_import_file": (
        "file.import",
        "Import a structure file into the document with SAMSON's importer.",
        _schema({"path": {"type": "string"}}, ["path"]),
    ),
    "samson_export_file": (
        "file.export",
        "Write the structure to a file with ASE (format from the extension).",
        _schema({"path": {"type": "string"}, "format": {"type": "string"}, "models": _MODELS},
                ["path"]),
    ),
    "samson_run_command": (
        "command.run",
        "Run a SAMSON command by its interface name; returns false if none matched.",
        _schema({"name": {"type": "string"}}, ["name"]),
    ),
    "samson_export_qm": (
        "qm.export",
        "Write a Gaussian (.gjf) or ORCA (.inp) input from the selected models for a "
        "quantum-chemistry follow-up: job 'auto' (1 model: Opt=TS with CalcFC; 2: QST2; "
        "3: QST3; ORCA uses NEB-TS), or 'ts', 'opt', 'irc'. Review level/charge/spin.",
        _schema(
            {
                "path": {"type": "string"},
                "job": {"type": "string", "enum": ["auto", "ts", "opt", "irc"]},
                "level": {"type": "string"},
                "charge": {"type": "integer"},
                "multiplicity": {"type": "integer", "minimum": 1},
            },
            ["path"],
        ),
    ),
    "samson_job_status": (
        "job.status",
        "State, step, recent log lines, and result of an MLIP job.",
        _schema(
            {"id": {"type": "integer", "minimum": 1}, "log_lines": {"type": "integer"}}, ["id"]
        ),
    ),
    "samson_stop_job": (
        "job.stop",
        "Ask an MLIP job to stop after its current step.",
        _schema({"id": {"type": "integer", "minimum": 1}}, ["id"]),
    ),
    "samson_list_jobs": ("job.list", "All MLIP jobs of this bridge session.", _schema()),
    "samson_exec": (
        "python.exec",
        "Run Python inside SAMSON (only when the user enabled it); the value of the last "
        "expression is returned. SAMSON, samson and np are predefined.",
        _schema({"code": {"type": "string"}}, ["code"]),
    ),
}

_JOB_START_SCHEMA = _schema(
    {
        "kind": {
            "type": "string",
            "enum": ["single_point", "relax", "md", "ts", "frequencies", "irc", "qst", "scan"],
        },
        "options": {
            "type": "object",
            "description": (
                "Optional. model/backend/device/dtype default to the MLIP panel's settings; "
                "backend 'xtb' (model = the xtb executable) takes xtb_method (gfn2|gfn1|gfnff), "
                "charge, multiplicity, solvent (ALPB). models ('auto'|'all'). relax: fmax, "
                "max_steps, optimizer. md: ensemble, "
                "temperature_k, timestep_fs, steps, friction_per_fs, tdamp_fs, seed, "
                "report_interval, fixed_distances ('0-3, 5-9:1.2'), trajectory, "
                "max_temperature_k. ts: method (prfo|dimer; prfo = Sella P-RFO, like "
                "Gaussian Opt=TS), fmax, max_steps, check_frequencies, exact_hessian "
                "(CalcFC), recompute_every (RecalcFC=N), check_irc (after the frequency "
                "check, with one imaginary mode: IRC both ways, report where the relaxed ends "
                "land, add the IRC path); dimer also start "
                "(hessian|pair|random), pair ('4-7'), displacement. irc (from a TS): step "
                "(Å·amu½), max_steps per side, fmax, relax_ends, trajectory (file for all "
                "frames), return_positions; adds an 'IRC path' with every frame. qst "
                "(QST2/QST3: select reactant, [guess,] product models in document order): "
                "images, fmax, max_steps, refine, check_frequencies, check_irc (matches the "
                "ends to reactant and product: 'connects'); adds the path and a TS model. "
                "scan (hard cases: ModRedundant scan then TS): pair ('4-7'), stop (Å), "
                "start?, points, relax_fmax, refine, exact_hessian, fmax, check_irc (reports "
                "the pair distance at each end); adds the scan path (constrained snapshots, "
                "not a trajectory; 'jumps' lists points where the geometry snapped). A TS "
                "from ts, qst, or scan is not confirmed until an IRC reaches both minima."
            ),
        },
    },
    ["kind"],
)
_VIEW_SCHEMA = _schema(
    {
        "width": {"type": "integer", "minimum": 64, "maximum": 4096, "default": 1200},
        "height": {"type": "integer", "minimum": 64, "maximum": 4096, "default": 800},
    }
)


class McpServer:
    """Translate MCP requests into bridge calls."""

    def __init__(self, client_factory: Callable[[], SamsonClient] | None = None):
        self._client_factory = client_factory or (
            lambda: SamsonClient.from_connection_file(timeout=120)
        )

    def tools(self) -> list[dict[str, Any]]:
        tools = [
            {"name": name, "description": description, "inputSchema": schema}
            for name, (_, description, schema) in _FORWARDED.items()
        ]
        tools.append(
            {
                "name": "samson_start_job",
                "description": "Start an MLIP job (single point, relax, MD, TS search, "
                "frequencies, IRC, QST2/QST3 path search, bond scan) on the open structure. "
                "Returns at once; poll samson_job_status. A TS is not confirmed until an IRC "
                "reaches both intended minima (irc job, or check_irc=true).",
                "inputSchema": _JOB_START_SCHEMA,
            }
        )
        tools.append(
            {
                "name": "samson_view",
                "description": "Take a screenshot of SAMSON's 3D viewport.",
                "inputSchema": _VIEW_SCHEMA,
            }
        )
        return tools

    # --- tool execution ----------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            client = self._client_factory()
            if name == "samson_view":
                return self._view(client, arguments)
            if name == "samson_start_job":
                options = arguments.get("options") or {}
                result = client.call("job.start", kind=arguments.get("kind"), **options)
            elif name in _FORWARDED:
                result = client.call(_FORWARDED[name][0], **arguments)
            else:
                raise KeyError(name)
        except BridgeError as exc:
            detail = exc.message
            if isinstance(exc.data, dict) and exc.data.get("traceback"):
                detail += "\n" + exc.data["traceback"]
            return {"content": [{"type": "text", "text": detail}], "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(result, indent=1)}]}

    def _view(self, client: SamsonClient, arguments: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "viewport.png"
            width = int(arguments.get("width", 1200))
            height = int(arguments.get("height", 800))
            client.capture(path, width, height)
            data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"content": [{"type": "image", "data": data, "mimeType": "image/png"}]}

    # --- protocol ----------------------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Answer one JSON-RPC message; notifications get no reply."""
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            return None
        params = message.get("params") or {}
        if method == "initialize":
            result: Any = {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "samson-bridge", "version": __version__},
                "instructions": INSTRUCTIONS,
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.tools()}
        elif method == "tools/call":
            name = params.get("name")
            if name not in _FORWARDED and name not in ("samson_view", "samson_start_job"):
                return _error(request_id, -32602, f"Unknown tool {name!r}")
            result = self.call_tool(name, params.get("arguments") or {})
        else:
            return _error(request_id, -32601, f"Method not found: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                reply = _error(None, -32700, f"Parse error: {exc}")
            else:
                reply = self.handle(message) if isinstance(message, dict) else None
            if reply is not None:
                stdout.write(json.dumps(reply) + "\n")
                stdout.flush()


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> int:
    McpServer().serve()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
