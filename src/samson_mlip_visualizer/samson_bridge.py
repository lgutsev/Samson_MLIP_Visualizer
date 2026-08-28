"""Conversion between one complete SAMSON structural model and ASE."""

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
    model: Any
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


def choose_structural_model(samson: Any) -> Any:
    """Choose the selected structural model, or the only model in the document."""
    models = list(samson.getNodes("node.type structuralModel"))
    selected = [model for model in models if _is_selected(model)]
    if len(selected) == 1:
        return selected[0]
    if len(models) == 1:
        return models[0]
    if not models:
        raise SamsonBridgeError("The active SAMSON document contains no structural model")
    if len(selected) > 1:
        raise SamsonBridgeError("Select exactly one structural model in SAMSON's Document View")
    raise SamsonBridgeError(
        "The document contains multiple structures. Select one complete structural model "
        "in SAMSON's Document View."
    )


def extract_structure(samson: Any | None = None) -> SamsonStructure:
    """Copy the chosen full model to ASE, including unit cell and fixed atoms."""
    if samson is None:
        from samson import SAMSON as samson

    model = choose_structural_model(samson)
    source_atoms = list(model.getNodes("node.type atom"))
    if not source_atoms:
        raise SamsonBridgeError("The selected structural model contains no atoms")

    symbols = [_symbol(atom) for atom in source_atoms]
    positions = np.array([
        [_angstrom(atom.getX()), _angstrom(atom.getY()), _angstrom(atom.getZ())]
        for atom in source_atoms
    ])

    cell_matrix = None
    pbc = (False, False, False)
    has_cell = getattr(model, "hasFiniteUnitCell", None)
    if has_cell and bool(has_cell() if callable(has_cell) else has_cell):
        unit_cell = model.getUnitCell()
        cell_matrix = np.array(
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

    ase_atoms = Atoms(symbols=symbols, positions=positions, cell=cell_matrix, pbc=pbc)
    fixed_indices = [index for index, atom in enumerate(source_atoms) if _is_fixed(atom)]
    if fixed_indices:
        ase_atoms.set_constraint(FixAtoms(indices=fixed_indices))
    return SamsonStructure(model=model, samson_atoms=source_atoms, ase_atoms=ase_atoms)


def sync_positions(
    structure: SamsonStructure,
    positions: Iterable[Iterable[float]],
    samson: Any | None = None,
) -> None:
    """Write ASE Angstrom positions to the matching SAMSON atoms."""
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
    samson.processEvents()
