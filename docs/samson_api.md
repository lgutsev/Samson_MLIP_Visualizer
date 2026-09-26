# A local API for SAMSON

**Status:** all four steps of the plan are implemented and have been used
against a running SAMSON 2026 R1 (11.0.1): the read-only probe, the bridge
server and client, MLIP jobs through the bridge, and an MCP server
([`samson_mlip_visualizer/remote/`](../src/samson_mlip_visualizer/remote/)).

The bridge gives programs on this computer a way to control a running SAMSON,
and optionally to run Python inside it. It never starts by itself: you start
it from the panel's **Start bridge** button or the Python console.

## Using it

Inside SAMSON:

```python
from samson_mlip_visualizer.remote import serve, stop
serve()                   # fixed operations only
serve(allow_exec=True)    # also accept Python code; only when you need it
stop()
```

From any Python on the same machine that has this package installed:

```python
from samson_mlip_visualizer.remote import SamsonClient

client = SamsonClient.from_connection_file()
print(client.summary())
atoms = client.get_structure()          # ASE Atoms, FixAtoms included
atoms.positions[0] += [0, 0, 0.1]
client.set_positions(atoms)             # one undo step in SAMSON
client.capture("view.png")              # screenshot of the 3D viewport
job = client.start_job("relax", fmax=0.01, optimizer="LBFGS")
print(client.wait_job(job["id"])["result"])
```

Jobs that omit `model`, `backend`, `device`, or `dtype` use the settings in the
open MLIP panel.

From the command line (`samson-remote` once the package is pip-installed, or
`python -m samson_mlip_visualizer.remote`):

```bash
samson-remote summary
samson-remote get --all -o document.extxyz
samson-remote set-positions relaxed.xyz
samson-remote nsl "node.type atom and atom.symbol H"
samson-remote capture view.png --width 1600 --height 1000
samson-remote job start md --set temperature_k=300 --set steps=2000
samson-remote job wait 1
samson-remote exec "print(len(SAMSON.getNodes('node.type atom')))"
```

The connection file is `%LOCALAPPDATA%\samson-mlip-visualizer\bridge.json` on
Windows (`~/.local/state/samson-mlip-visualizer/bridge.json` elsewhere); set
`SAMSON_BRIDGE_CONNECTION` to use another path.

### For coding assistants (MCP)

`samson-mcp` (or `python -m samson_mlip_visualizer.remote.mcp_server`) is a
stdio [Model Context Protocol](https://modelcontextprotocol.io) server with one
tool per operation below, plus `samson_view`, which returns the viewport as an
image. Register it with the Python that has this package; for Claude Code, a
project `.mcp.json`:

```json
{
  "mcpServers": {
    "samson": {
      "type": "stdio",
      "command": "C:/path/to/SAMSON-Application/11.0.1/Binaries/python.exe",
      "args": ["-W", "ignore", "-m", "samson_mlip_visualizer.remote.mcp_server"]
    }
  }
}
```

The MCP server is stateless: it reads the connection file on every call, so
restarting the bridge needs no restart of the assistant.

## Operations

Atom indices count over all structural models in document order, the same
order as `structure.get` with `models="all"`. Every edit is one undo step.

| Method | Purpose |
|---|---|
| `bridge.ping`, `bridge.info` | liveness, versions, options, busy state, method list |
| `document.summary` | models, atom counts, individually and effectively selected atoms, cells |
| `structure.get` | symbols, positions, cell, PBC, fixed atoms (`models`: `auto`/`all`) |
| `structure.set_positions` | write positions back (atom order checked via `symbols`) |
| `selection.get` / `selection.set` | read or change the selection by index |
| `selection.select` | select with an NSL expression, e.g. `node.type atom and atom.symbol O` |
| `view.capture` | save the 3D viewport as an image |
| `atoms.add` / `atoms.delete` | create atoms in a model / erase atoms |
| `atoms.set_elements` / `atoms.set_fixed` | change elements / SAMSON's fixed-atom flag |
| `history.undo` / `history.redo` | SAMSON's undo and redo |
| `file.import` / `file.export` | SAMSON's importer / write a structure with ASE |
| `command.run` | run a SAMSON command by its interface name (false if none matched) |
| `job.start` | (`backend: "xtb"` uses the xtb executable as `model`, with `xtb_method`, `charge`, `multiplicity`, `solvent`; `backend: "psi4"` uses the Psi4 environment's python, with `psi4_method`, `basis`, `charge`, `multiplicity`) start `single_point`, `relax`, `md`, `ts` (`method`: `prfo`/`dimer`), `frequencies`, `irc` (adds an IRC path with every frame; `trajectory` writes them too), `qst` (QST2/QST3 from the selected models; adds the path and a TS model), or `scan` (`pair='I-J'`, `stop`, `points`: bond scan, then P-RFO from the highest point; adds the scan path). `ts`/`scan` take `exact_hessian`, and `ts` takes `recompute_every` (Gaussian `CalcFC` / `RecalcFC=N`); returns at once |
| `qm.export` | write a Gaussian (`.gjf`/`.com`) or ORCA (`.inp`) input from the selected models: `path`, `job` (`auto`: 1 model → `ts`, 2 → `qst2`, 3 → `qst3`; or `ts`/`opt`/`irc`), `level`, `charge`, `multiplicity` |
| `job.status` / `job.stop` / `job.list` | progress, log and result / stop after the current step / all jobs |
| `python.exec` | **opt-in only**: run code in SAMSON's Python; returns stdout, stderr and the last expression |

**Transition-state workflow.** Clients (and assistants, through the MCP
instructions) should follow this order: relax the end points (float64, tight
`fmax`); find the TS (`ts` from a guess, `qst` from two minima, or `scan` from a
forming or breaking bond); check frequencies (exactly one imaginary mode, moving
the right atoms); then confirm connectivity with `irc`, or `check_irc: true` on
`ts`/`qst`/`scan`. That option reports each relaxed end's energy relative to the
TS, the scanned pair distance (`scan`), and the matched minimum plus `connects`
(`qst`). Only then use `qm.export`. A TS is not confirmed until an IRC reaches
both minima: QST's final P-RFO can slide to another saddle, and a scan whose
`jumps` list is not empty is not a reaction path.

**Selection semantics.** "Selected atoms" are atoms picked individually;
"effectively selected" also counts atoms inside a selected model (SAMSON's
`isSelected`). The panel's pair buttons use the first.

**Jobs** run on SAMSON's main thread, like panel jobs, but start after the
request that created them has been answered. Their progress callbacks keep
SAMSON's event loop turning, so SAMSON repaints and the bridge keeps answering
status, stop, and read requests. One job runs at a time; while one runs,
document changes (from the bridge or the panel) are refused. A job that moves
atoms is one undo step; one that moves nothing leaves no undo step.

## Security model

- Bound to `127.0.0.1`; non-loopback peers are dropped.
- Every request needs the session token (compared in constant time); a new
  random token is made at each start and lives only in the per-user connection
  file.
- Fixed operations by default; `python.exec` exists only when the server is
  started with `allow_exec=True`, because it is equivalent to typing into
  SAMSON's console. Turn it off when you are not using it.
- Each accepted request is logged (panel log, or the console for `serve()`);
  stopping the bridge removes the connection file.
- Messages are capped at 64 MB.
- Binding to loopback does not normally trigger a Windows Firewall prompt.

## Known limitations

- Unit cells cannot be set through the bridge yet (`SBStructuralModel.setUnitCell`
  exists; the `SBUnitCell` constructor has not been wired up).
- No change notifications: clients poll.
- `SAMSON.exportToFile` (SAMSON's own exporter) took ~12 s in testing, so
  `file.export` writes with ASE instead.
- Commands must be named exactly as in SAMSON's interface; there is no command
  listing.

## Background

When this was designed (September 2026), SAMSON had no external entry point:
with SAMSON running it listens on no network ports, the Python console's
Jupyter kernel is in-process (an outside Jupyter client cannot attach), and
SAMSON AI is an in-application assistant. Inside SAMSON, the SDK's Python
bindings, PySide6 with QtNetwork, and this package's engines provide everything
needed; SAMSON's API must be called on its main thread, which a Qt server
guarantees.

| Alternative | Verdict |
|---|---|
| Attach to the console's Jupyter kernel | Not possible: the kernel is in-process |
| Start a full Jupyter kernel inside SAMSON | Unrestricted code execution; its event loop competes with SAMSON's |
| Folder watcher executing job files | No sockets, but slow and awkward; fallback if local ports are blocked |
| Native C++ SAMSON extension | Right for an interactive MLIP interaction model, overkill for an API |

[`scripts/probe_samson_api.py`](../scripts/probe_samson_api.py) is a read-only
survey of SAMSON's Python API (it changes nothing and opens no connections);
run it in SAMSON's code editor to check names on another SAMSON version.
