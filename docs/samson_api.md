# A local API for SAMSON

**Status:** steps 1–2 of the plan are implemented: the read-only probe, and the
bridge server, client and core operations in
[`samson_mlip_visualizer/remote/`](../src/samson_mlip_visualizer/remote/). They
are tested against a SAMSON stand-in and with a real Qt server in SAMSON's
Python, but not yet inside a running SAMSON. MLIP jobs through the bridge
(step 3) and the MCP server (step 4) are next.

The bridge gives programs on this computer a way to control a running SAMSON,
and optionally to run Python inside it. It never starts by itself: you start
it from the console or the panel.

## Using it

Start it inside SAMSON, from the panel's **Start bridge** button or the Python
console:

```python
from samson_mlip_visualizer.remote import serve, stop
serve()                   # fixed operations only
serve(allow_exec=True)    # also accept Python code; only when you need it
stop()
```

Then, from any Python on the same machine that has this package installed:

```python
from samson_mlip_visualizer.remote import SamsonClient

client = SamsonClient.from_connection_file()
print(client.summary())
atoms = client.get_structure()          # ASE Atoms, FixAtoms included
atoms.positions[0] += [0, 0, 0.1]
client.set_positions(atoms)             # one undo step in SAMSON
```

or the command line (`samson-remote` once the package is pip-installed, or
`python -m samson_mlip_visualizer.remote.client`):

```bash
samson-remote summary
samson-remote get --all -o document.extxyz
samson-remote set-positions relaxed.xyz
samson-remote select --atoms 0 3
samson-remote command "Center"
samson-remote exec "print(len(SAMSON.getNodes('node.type atom')))"
```

The connection file is `%LOCALAPPDATA%\samson-mlip-visualizer\bridge.json` on
Windows (`~/.local/state/samson-mlip-visualizer/bridge.json` elsewhere); set
`SAMSON_BRIDGE_CONNECTION` to use another path. Atom indices in
`selection.get/set` count over all structural models in document order, the
same order as `structure.get` with `models="all"`. While the panel runs a
job, the bridge still answers reads but refuses changes to the document.

## Why

Driving SAMSON from outside the GUI would let notebooks, batch scripts, and
coding assistants load structures, run MLIP jobs, and read results without a
person copying output between windows. It would also let SAMSON act as a live
viewer for jobs started elsewhere.

## What exists today (SAMSON 11.0.1, checked September 2026)

- **No external entry point.** With SAMSON running, it listens on no network
  ports; its sockets are internal loopback pairs and HTTPS to SAMSON Connect.
- **The Python console's Jupyter kernel is in-process**, so an outside Jupyter
  client cannot attach to it.
- **SAMSON AI** is an in-application assistant, not an interface other
  programs can call (as far as the documentation shows).
- **Inside SAMSON, everything needed is available:** the Python bindings of the
  SDK (commands via `SAMSON.runCommand`, file import/export, node queries in
  the Node Specification Language, undo grouping with `SAMSON.holding`,
  animation paths), PySide6 6.10.2 with QtNetwork (`QTcpServer`,
  `QLocalServer`), and this package's structure bridge and MLIP engines.
- **Constraint:** SAMSON's API must be called on its main (GUI) thread.

## Design: a local bridge in three layers

1. **Server inside SAMSON**, started from the Python console or the panel.
   - Qt's own server class, so requests arrive on SAMSON's main thread and
     SAMSON calls need no cross-thread marshalling.
   - Bound to `127.0.0.1` only; peers must be loopback.
   - JSON-RPC 2.0, one message per line.
   - Every request carries a random session token. The token and port live in
     a connection file readable only by the current user, as Jupyter does for
     kernels.
   - A fixed list of operations (below). Arbitrary Python execution is a
     separate opt-in, off by default.
2. **Plain Python client** with no SAMSON dependency: usable from notebooks,
   scripts, the `samson-remote` CLI, and tests.
3. **MCP server** wrapping the client, so a coding assistant can drive SAMSON
   with typed tools instead of pasted output.

### Operations

| Method | Purpose |
|---|---|
| `bridge.ping`, `bridge.info` | liveness, versions, enabled options |
| `document.summary` | structural models, atom counts, selection, cells |
| `structure.get` | symbols, positions, cell, PBC, fixed atoms (selected or all models) |
| `structure.set_positions` | write positions back (one undo step; atom order checked) |
| `selection.get` / `selection.set` | read or change the SAMSON selection |
| `file.import` / `file.export` | load a file into the document / write a structure |
| `command.run` | run a SAMSON command by its interface name |
| `python.exec` | **opt-in only**: run code in SAMSON's Python; returns stdout, stderr and the last expression |
| `mlip.*` | *planned (step 3)*: single point, relax, MD, TS search, frequencies as jobs with progress and stop |

`bridge.info` lists the methods a running bridge offers.

### Security model

- Bound to `127.0.0.1`; non-loopback peers are dropped.
- Every request needs the session token (compared in constant time); a new
  random token is made at each start and lives only in the per-user connection
  file.
- Fixed operations by default; `python.exec` exists only when the server is
  started with `allow_exec=True`, because it is equivalent to typing into
  SAMSON's console.
- Each accepted request is logged (panel log, or the console for `serve()`);
  stopping the bridge removes the connection file.
- Messages are capped at 64 MB; document changes are refused while a panel job
  runs.
- Binding to loopback does not normally trigger a Windows Firewall prompt.

## Alternatives considered

| Option | Verdict |
|---|---|
| Attach to the console's Jupyter kernel | Not possible: the kernel is in-process |
| Start a full Jupyter kernel inside SAMSON | Unrestricted code execution; its event loop competes with SAMSON's |
| Folder watcher executing job files | No sockets, but slow and awkward; fallback if local ports are blocked |
| Native C++ SAMSON extension | Right for an interactive MLIP interaction model, overkill for an API |

## Plan

1. **Probe** (done) — `scripts/probe_samson_api.py`, read-only, records the
   exact API names still unverified: screenshot capture, SBPath creation,
   selection setters, import/export signatures.
2. **Server, client, core operations** (done) — tested against a SAMSON
   stand-in and with a real Qt server; next, a run inside SAMSON.
3. **MLIP jobs** through the bridge, with progress and stop.
4. **MCP server** and its configuration for coding assistants.

## Open questions

- Exact import/export and viewport-capture function names (probe).
- Whether selection flags on atoms are writable from Python (probe).
- Whether a server created from the console keeps running after the cell
  returns (expected, as the panel's widgets do, if a reference is kept).
