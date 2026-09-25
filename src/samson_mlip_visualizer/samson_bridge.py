"""Conversion between complete SAMSON structural models and ASE."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


class SamsonBridgeError(RuntimeError):
    """Raised when the active SAMSON document cannot be mapped safely."""


@dataclass
class SamsonStructure:
    models: list[Any]
    samson_atoms: list[Any]
    ase_atoms: Atoms


def _angstrom(value: Any) -> float:
    converted = getattr(value, "angstrom", None)
    if converted is not None:
        converted = converted() if callable(converted) else converted
        raw_value = getattr(converted, "value", converted)
        return float(raw_value() if callable(raw_value) else raw_value)
    return float(value)


def _is_selected(node: Any) -> bool:
    for name in ("selectionFlag", "selected"):
        value = getattr(node, name, None)
        if value is not None:
            return bool(value() if callable(value) else value)
    getter = getattr(node, "getSelectionFlag", None)
    return bool(getter()) if getter else False


def set_selection_flag(node: Any, value: bool) -> None:
    """Select or deselect a SAMSON node."""
    setter = getattr(node, "setSelectionFlag", None)
    if callable(setter):
        setter(bool(value))
        return
    try:
        node.selectionFlag = bool(value)
    except AttributeError as exc:
        raise SamsonBridgeError(
            "This SAMSON build does not allow changing the selection from Python"
        ) from exc


def is_effectively_selected(node: Any) -> bool:
    """Selected itself or through a selected ancestor (SAMSON's ``isSelected``)."""
    value = getattr(node, "isSelected", None)
    if value is not None:
        return bool(value() if callable(value) else value)
    return _is_selected(node)


def element_type(symbol: str) -> Any:
    """SAMSON's element type for a chemical symbol such as ``"C"``."""
    from ase.data import chemical_symbols
    from samson import SBElement

    if symbol not in chemical_symbols[1:]:
        raise SamsonBridgeError(f"Unknown element symbol {symbol!r}")
    return getattr(SBElement, symbol)


def add_atom(parent: Any, symbol: str, position: Iterable[float]) -> Any:
    """Create an atom (Å) under ``parent``; call inside ``SAMSON.holding``."""
    from samson import SBAtom, SBQuantity

    x, y, z = (SBQuantity.angstrom(float(value)) for value in position)
    atom = SBAtom(element_type(symbol), x, y, z)
    atom.create()
    parent.addChild(atom)
    return atom


def atom_parent(model: Any) -> Any:
    """Where new atoms of ``model`` go: its first child group, else its root."""
    root = model.getStructuralRoot()
    children = list(root.getChildren()) if hasattr(root, "getChildren") else []
    return children[0] if children else root


def node_name(node: Any) -> str:
    """A SAMSON node's display name, or an empty string."""
    for name in ("name", "getName"):
        value = getattr(node, name, None)
        if value is not None:
            return str(value() if callable(value) else value)
    return ""


def _is_fixed(atom: Any) -> bool:
    for name in ("fixedFlag", "isFixed"):
        value = getattr(atom, name, None)
        if value is not None:
            return bool(value() if callable(value) else value)
    return False


def _symbol(atom: Any) -> str:
    value = getattr(atom, "elementSymbol", None)
    if value is not None:
        return str(value() if callable(value) else value)
    getter = getattr(atom, "getElementSymbol", None)
    if getter:
        return str(getter())
    raise SamsonBridgeError("A SAMSON atom has no readable element symbol")


def _vector_angstrom(vector: Any) -> list[float]:
    return [_angstrom(vector[index]) for index in range(3)]


def _require_active_document(samson: Any) -> None:
    getter = getattr(samson, "getActiveDocument", None)
    if getter is None:
        return
    try:
        document = getter()
    except (AttributeError, TypeError):
        return
    if document is None:
        raise SamsonBridgeError("No active SAMSON document. Open or build a structure first.")


def _reject_pseudo_atoms(model: Any) -> None:
    getter = getattr(model, "getNodes", None)
    if getter is None:
        return
    try:
        pseudo_atoms = list(getter("node.type pseudoAtom"))
    except (AssertionError, AttributeError, TypeError):
        return
    if pseudo_atoms:
        raise SamsonBridgeError(
            f"The selected model contains {len(pseudo_atoms)} pseudo-atom(s). A local MLIP "
            "needs real atomic environments; convert or remove pseudo-atoms before evaluating."
        )


def choose_structural_models(samson: Any) -> list[Any]:
    """Choose the selected structural models, or the only model in the document.

    Several selected models are evaluated together as one system (e.g. separate water
    molecules). Each model is complete, so this never evaluates a partial environment.
    """
    models = list(samson.getNodes("node.type structuralModel"))
    selected = [model for model in models if _is_selected(model)]
    if selected:
        return selected
    if len(models) == 1:
        return models
    if not models:
        raise SamsonBridgeError("The active SAMSON document contains no structural model")
    raise SamsonBridgeError(
        "The document contains multiple structural models. Select every model that belongs "
        "to the system (Ctrl/Shift-click in SAMSON's Document View); they are evaluated "
        "together."
    )


def _unit_cell(model: Any) -> tuple[np.ndarray, tuple[bool, bool, bool]] | None:
    has_cell = getattr(model, "hasFiniteUnitCell", None)
    if not (has_cell and bool(has_cell() if callable(has_cell) else has_cell)):
        return None
    unit_cell = model.getUnitCell()
    matrix = np.array(
        [
            _vector_angstrom(unit_cell.getVectorA()),
            _vector_angstrom(unit_cell.getVectorB()),
            _vector_angstrom(unit_cell.getVectorC()),
        ]
    )
    pbc = (
        bool(unit_cell.isPeriodicX()),
        bool(unit_cell.isPeriodicY()),
        bool(unit_cell.isPeriodicZ()),
    )
    return matrix, pbc


def _shared_unit_cell(models: list[Any]) -> tuple[np.ndarray | None, tuple[bool, bool, bool]]:
    """Return the one cell the models agree on; models without a cell adopt it."""
    cells = [cell for cell in (_unit_cell(model) for model in models) if cell is not None]
    if not cells:
        return None, (False, False, False)
    matrix, pbc = cells[0]
    for other_matrix, other_pbc in cells[1:]:
        if other_pbc != pbc or not np.allclose(other_matrix, matrix, atol=1e-6):
            raise SamsonBridgeError(
                "The selected structural models define different unit cells or periodicity. "
                "Give them one common cell before evaluating them together."
            )
    return matrix, pbc


def extract_structure(
    samson: Any | None = None, models: list[Any] | None = None
) -> SamsonStructure:
    """Copy the chosen full model(s) to ASE, including unit cell and fixed atoms.

    ``models`` overrides the selection rules of :func:`choose_structural_models`.
    """
    if samson is None:
        from samson import SAMSON as samson

    _require_active_document(samson)
    if models is None:
        models = choose_structural_models(samson)
    source_atoms = []
    for model in models:
        _reject_pseudo_atoms(model)
        source_atoms.extend(model.getNodes("node.type atom"))
    if not source_atoms:
        raise SamsonBridgeError("The selected structural model(s) contain no atoms")

    symbols = [_symbol(atom) for atom in source_atoms]
    positions = np.array([
        [_angstrom(atom.getX()), _angstrom(atom.getY()), _angstrom(atom.getZ())]
        for atom in source_atoms
    ])
    cell_matrix, pbc = _shared_unit_cell(models)

    ase_atoms = Atoms(symbols=symbols, positions=positions, cell=cell_matrix, pbc=pbc)
    fixed_indices = [index for index, atom in enumerate(source_atoms) if _is_fixed(atom)]
    if fixed_indices:
        ase_atoms.set_constraint(FixAtoms(indices=fixed_indices))
    return SamsonStructure(models=models, samson_atoms=source_atoms, ase_atoms=ase_atoms)


def selected_atom_indices(structure: SamsonStructure) -> list[int]:
    """Indices (into ``structure.ase_atoms``) of atoms selected in SAMSON."""
    return [index for index, atom in enumerate(structure.samson_atoms) if _is_selected(atom)]


def sync_positions(
    structure: SamsonStructure,
    positions: Iterable[Iterable[float]],
    samson: Any | None = None,
    *,
    process_events: bool = True,
) -> None:
    """Write ASE Angstrom positions to the matching SAMSON atoms.

    ``process_events`` lets SAMSON repaint immediately; callers already inside
    an event handler pass ``False`` to avoid re-entering the event loop.
    """
    if samson is None:
        from samson import SAMSON as samson
        from samson import SBQuantity
    else:
        try:
            from samson import SBQuantity
        except ImportError as exc:
            raise SamsonBridgeError("SBQuantity is unavailable outside SAMSON") from exc

    coordinates = np.asarray(positions, dtype=float)
    if coordinates.shape != (len(structure.samson_atoms), 3):
        raise SamsonBridgeError("Position array does not match the SAMSON atom count")
    for atom, (x, y, z) in zip(structure.samson_atoms, coordinates, strict=True):
        atom.setX(SBQuantity.angstrom(float(x)))
        atom.setY(SBQuantity.angstrom(float(y)))
        atom.setZ(SBQuantity.angstrom(float(z)))
    if process_events:
        samson.processEvents()
