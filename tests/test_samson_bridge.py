import sys
from types import ModuleType

import numpy as np
import pytest

from samson_mlip_visualizer.samson_bridge import (
    SamsonBridgeError,
    choose_structural_model,
    extract_structure,
    sync_positions,
)


class Quantity:
    def __init__(self, value):
        self.value = value

    @property
    def angstrom(self):
        return self


class Vector:
    def __init__(self, values):
        self.values = [Quantity(value) for value in values]

    def __getitem__(self, index):
        return self.values[index]


class UnitCell:
    def getVectorA(self):
        return Vector([5, 0, 0])

    def getVectorB(self):
        return Vector([0, 6, 0])

    def getVectorC(self):
        return Vector([0, 0, 20])

    def isPeriodicX(self):
        return True

    def isPeriodicY(self):
        return True

    def isPeriodicZ(self):
        return False


class Atom:
    def __init__(self, symbol, position, fixed=False):
        self.elementSymbol = symbol
        self.position = list(position)
        self.fixedFlag = fixed

    def getX(self):
        return Quantity(self.position[0])

    def getY(self):
        return Quantity(self.position[1])

    def getZ(self):
        return Quantity(self.position[2])

    def setX(self, value):
        self.position[0] = value.angstrom.value

    def setY(self, value):
        self.position[1] = value.angstrom.value

    def setZ(self, value):
        self.position[2] = value.angstrom.value


class Model:
    def __init__(self, atoms, selected=False, pseudo_atoms=()):
        self.atoms = atoms
        self.selectionFlag = selected
        self.pseudo_atoms = list(pseudo_atoms)

    def getNodes(self, query):
        if query == "node.type atom":
            return self.atoms
        if query == "node.type pseudoAtom":
            return self.pseudo_atoms
        raise AssertionError(f"unexpected query: {query!r}")

    def hasFiniteUnitCell(self):
        return True

    def getUnitCell(self):
        return UnitCell()


class FakeSamson:
    def __init__(self, models):
        self.models = models
        self.events = 0

    def getNodes(self, query):
        assert query == "node.type structuralModel"
        return self.models

    def processEvents(self):
        self.events += 1


def test_requires_unambiguous_structural_model():
    samson = FakeSamson([Model([]), Model([])])
    with pytest.raises(SamsonBridgeError, match="multiple structures"):
        choose_structural_model(samson)


def test_selected_model_wins():
    selected = Model([], selected=True)
    assert choose_structural_model(FakeSamson([Model([]), selected])) is selected


def test_extracts_cell_pbc_and_fixed_atoms():
    atoms = [Atom("Si", [0, 0, 0], fixed=True), Atom("H", [1, 2, 3])]
    structure = extract_structure(FakeSamson([Model(atoms)]))
    assert structure.ase_atoms.get_chemical_symbols() == ["Si", "H"]
    assert structure.ase_atoms.get_pbc().tolist() == [True, True, False]
    assert structure.ase_atoms.cell.lengths().tolist() == pytest.approx([5, 6, 20])
    assert structure.ase_atoms.constraints[0].get_indices().tolist() == [0]


def test_rejects_pseudo_atoms():
    atoms = [Atom("Si", [0, 0, 0])]
    model = Model(atoms, pseudo_atoms=[object()])
    with pytest.raises(SamsonBridgeError, match="pseudo-atom"):
        extract_structure(FakeSamson([model]))


def test_syncs_positions(monkeypatch):
    atoms = [Atom("H", [0, 0, 0]), Atom("H", [1, 0, 0])]
    samson = FakeSamson([Model(atoms)])
    structure = extract_structure(samson)

    module = ModuleType("samson")

    class SBQuantity:
        @staticmethod
        def angstrom(value):
            return Quantity(value)

    module.SBQuantity = SBQuantity
    monkeypatch.setitem(sys.modules, "samson", module)
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
    sync_positions(structure, positions, samson=samson)
    assert atoms[0].position == pytest.approx([0.1, 0.2, 0.3])
    assert atoms[1].position == pytest.approx([1.1, 1.2, 1.3])
    assert samson.events == 1
