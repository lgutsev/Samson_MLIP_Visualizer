"""Read-only survey of SAMSON's Python API, to plan the panel's next features.

Run this file in SAMSON's Python code editor. Select two atoms first if you want
to check the MD/TS "selected pair" buttons. The script changes nothing in the
document and opens no network connections. It writes ``samson_api_probe.json``
at the repository root (git-ignored) and prints a short summary.

It looks for the API behind open roadmap items: trajectory playback (SBPath),
screenshots, file import/export, selection, and per-atom flags.
"""

import json
import re
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

INTERESTING = re.compile(
    r"(?i)import|export|command|capture|screenshot|image|render|snapshot|hold|undo|path|"
    r"select|document|camera|frame|anim|version|property|node"
)
NODE_ATTRIBUTES = re.compile(r"(?i)select|fixed|flag|name|element|index|visib|highlight")


def _doc(obj, limit=400):
    return (getattr(obj, "__doc__", None) or "").strip()[:limit]


def _public(obj):
    return sorted(name for name in dir(obj) if not name.startswith("_"))


def _describe(owner, names):
    described = {}
    for name in names:
        try:
            value = getattr(owner, name)
        except Exception as exc:  # noqa: BLE001 - report whatever the binding raises
            described[name] = {"error": repr(exc)}
            continue
        described[name] = {"type": type(value).__name__, "doc": _doc(value)}
    return described


def _node_attributes(node):
    """Flag-like attributes of a node: kind, writability, and (for data) value.

    Methods are never called, so nothing in the document can change.
    """
    cls = type(node)
    attributes = {}
    for name in _public(node):
        if not NODE_ATTRIBUTES.search(name):
            continue
        descriptor = getattr(cls, name, None)
        entry = {"class_attribute": type(descriptor).__name__}
        if isinstance(descriptor, property):
            entry["writable"] = descriptor.fset is not None
        try:
            value = getattr(node, name)
            entry["value"] = "<callable>" if callable(value) else repr(value)[:200]
        except Exception as exc:  # noqa: BLE001
            entry["value_error"] = repr(exc)
        attributes[name] = entry
    return attributes


def _repository_root():
    import samson_mlip_visualizer

    root = Path(samson_mlip_visualizer.__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() else Path.home()


def probe():
    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "executable": sys.executable,
        "thread": {
            "name": threading.current_thread().name,
            "is_main": threading.current_thread() is threading.main_thread(),
        },
    }

    try:
        import PySide6
        from PySide6 import QtCore, QtWidgets

        app = QtWidgets.QApplication.instance()
        report["qt"] = {
            "pyside6": PySide6.__version__,
            "qt": QtCore.qVersion(),
            "application": type(app).__name__ if app else None,
            "on_gui_thread": bool(app) and QtCore.QThread.currentThread() == app.thread(),
        }
    except Exception:  # noqa: BLE001
        report["qt"] = {"error": traceback.format_exc(limit=2)}

    import samson
    from samson import SAMSON

    facade_names = _public(SAMSON)
    module_names = _public(samson)
    report["samson_facade"] = {
        "names": facade_names,
        "candidates": _describe(SAMSON, [n for n in facade_names if INTERESTING.search(n)]),
    }
    report["samson_module"] = {
        "names": module_names,
        "candidates": _describe(
            samson, [n for n in module_names if re.search(r"(?i)path|anim|camera|image", n)]
        ),
    }

    versions = {}
    for name in facade_names:
        if re.fullmatch(r"get\w*Version\w*", name):
            try:
                versions[name] = repr(getattr(SAMSON, name)())
            except Exception as exc:  # noqa: BLE001
                versions[name] = f"error: {exc!r}"
    report["samson_versions"] = versions

    models = list(SAMSON.getNodes("node.type structuralModel"))
    report["structural_models"] = {"count": len(models)}
    if models:
        model = models[0]
        report["structural_models"]["first_model"] = _node_attributes(model)
        atoms = list(model.getNodes("node.type atom"))
        if atoms:
            report["first_atom"] = _node_attributes(atoms[0])

    from samson_mlip_visualizer.samson_bridge import extract_structure, selected_atom_indices

    try:
        structure = extract_structure()
        report["selection_check"] = {
            "atoms_in_structure": len(structure.samson_atoms),
            "selected_atom_indices": selected_atom_indices(structure),
        }
    except Exception as exc:  # noqa: BLE001
        report["selection_check"] = {"error": str(exc)}

    path = _repository_root() / "samson_api_probe.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    candidates = report["samson_facade"]["candidates"]
    print(f"SAMSON API probe written to {path}")
    print(f"  SAMSON facade: {len(facade_names)} names, {len(candidates)} of interest")
    for topic in ("import", "export", "capture|screenshot|image", "path|anim", "select"):
        hits = [n for n in candidates if re.search(topic, n, re.IGNORECASE)]
        print(f"  {topic:26s} {', '.join(hits[:8]) or '-'}")
    print(f"  Selected-pair check: {report['selection_check']}")
    return report


probe()
