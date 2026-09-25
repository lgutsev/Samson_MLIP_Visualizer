# A local API for SAMSON: proposal

**Status: proposed, not implemented.** The bridge below would give programs on
this computer a way to control a running SAMSON, and optionally to run Python
inside it. That is a deliberate new entry point, so it waits for an explicit
decision to build it. Nothing in the package opens a network port today.

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
   scripts, the `samson-mlip` CLI, and tests.
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
| `mlip.*` | single point, relax, MD, TS search, frequencies as jobs with progress and stop |
| `python.exec` | **opt-in only**: run code in SAMSON's Python |

### Security model

- Loopback only, token-authenticated, per-user connection file.
- Fixed operations by default; `python.exec` requires starting the server with
  an explicit flag, because it is equivalent to typing into SAMSON's console.
- Each request is logged in the panel; stopping the bridge removes the
  connection file.
- Binding to loopback does not normally trigger a Windows Firewall prompt.

## Alternatives considered

| Option | Verdict |
|---|---|
| Attach to the console's Jupyter kernel | Not possible: the kernel is in-process |
| Start a full Jupyter kernel inside SAMSON | Unrestricted code execution; its event loop competes with SAMSON's |
| Folder watcher executing job files | No sockets, but slow and awkward; fallback if local ports are blocked |
| Native C++ SAMSON extension | Right for an interactive MLIP interaction model, overkill for an API |

## Plan

1. **Probe** — `scripts/probe_samson_api.py` (available now, read-only)
   records the exact API names still unverified: screenshot capture, SBPath
   creation, selection setters, import/export signatures.
2. **Server, client, core operations** — tested against a SAMSON stand-in, then
   in SAMSON.
3. **MLIP jobs** through the bridge, with progress and stop.
4. **MCP server** and its configuration for coding assistants.

## Open questions

- Exact import/export and viewport-capture function names (probe).
- Whether selection flags on atoms are writable from Python (probe).
- Whether a server created from the console keeps running after the cell
  returns (expected, as the panel's widgets do, if a reference is kept).
