"""Stand-ins for the parts of SAMSON's Python API that the package uses."""

import contextlib
import sys
from types import ModuleType, SimpleNamespace

from ase.data import chemical_symbols


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
    def __init__(self, a=5):
        self.a = a

    def getVectorA(self):
        return Vector([self.a, 0, 0])

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
    def __init__(self, symbol, position, fixed=False, selected=False):
        self.elementSymbol = symbol
        self.position = list(position)
        self.fixedFlag = fixed
        self.selectionFlag = selected
        self.model = None
        self.created = False

    @property
    def isSelected(self):
        """Selected itself or through its model, like SAMSON's inherited flag."""
        return self.selectionFlag or bool(self.model and self.model.selectionFlag)

    @property
    def elementType(self):
        return self.elementSymbol

    @elementType.setter
    def elementType(self, value):
        self.elementSymbol = value

    def create(self):
        self.created = True

    def erase(self):
        self.model.atoms.remove(self)
        self.model = None

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
    def __init__(self, atoms, selected=False, pseudo_atoms=(), cell=True, name="model"):
        self.atoms = atoms
        self.selectionFlag = selected
        self.pseudo_atoms = list(pseudo_atoms)
        self.cell = UnitCell() if cell is True else cell
        self.name = name
        for atom in atoms:
            atom.model = self

    def getStructuralRoot(self):
        return Root(self)

    def getNodes(self, query):
        if query == "node.type atom":
            return self.atoms
        if query == "node.type pseudoAtom":
            return self.pseudo_atoms
        raise AssertionError(f"unexpected query: {query!r}")

    def hasFiniteUnitCell(self):
        return self.cell is not None

    def getUnitCell(self):
        return self.cell


class Group:
    """A structural group (e.g. a molecule) that new atoms are added to."""

    def __init__(self, model):
        self.model = model

    def addChild(self, atom):
        atom.model = self.model
        self.model.atoms.append(atom)
        return True


class Root:
    def __init__(self, model):
        self.model = model

    def getChildren(self):
        return [Group(self.model)]


class FakeSamson:
    """The ``SAMSON`` facade: document queries, events, undo, commands, import."""

    def __init__(self, models):
        self.models = models
        self.events = 0
        self.holds = []
        self.commands = []
        self.imports = []
        self.selections = []
        self.captures = []
        self.history = []

    def select(self, nsl):
        self.selections.append(nsl)
        return nsl != "invalid"

    def captureViewportToFile(self, path, width, height, transparent, path_tracing, progress):
        self.captures.append((width, height, transparent, path_tracing, progress))
        with open(path, "wb") as stream:
            stream.write(b"\x89PNG\r\n\x1a\n")

    def undo(self):
        self.history.append("undo")

    def redo(self):
        self.history.append("redo")

    def getNodes(self, query):
        assert query == "node.type structuralModel"
        return self.models

    def processEvents(self):
        self.events += 1

    @contextlib.contextmanager
    def holding(self, name):
        self.holds.append(name)
        yield

    def runCommand(self, name):
        self.commands.append(name)
        return True

    def importFromFile(self, path):
        self.imports.append(path)


def water_document():
    """Two separate water models; the first is selected, as is its oxygen atom."""
    first = Model(
        [
            Atom("O", [0, 0, 0], selected=True),
            Atom("H", [0.96, 0, 0]),
            Atom("H", [-0.24, 0.93, 0]),
        ],
        selected=True,
        cell=None,
        name="water 1",
    )
    second = Model(
        [Atom("O", [3, 0, 0], fixed=True), Atom("H", [3.96, 0, 0]), Atom("H", [2.76, 0.93, 0])],
        cell=None,
        name="water 2",
    )
    return FakeSamson([first, second])


def install_samson_module(monkeypatch, facade=None):
    """Register a fake ``samson`` module (``SAMSON`` and ``SBQuantity``)."""
    module = ModuleType("samson")
    module.SAMSON = facade

    class SBQuantity:
        @staticmethod
        def angstrom(value):
            return Quantity(value)

    module.SBQuantity = SBQuantity
    # Element types are plain symbols here; SBAtom(element, x, y, z) takes Quantities.
    module.SBElement = SimpleNamespace(**{symbol: symbol for symbol in chemical_symbols[1:]})
    module.SBAtom = lambda element, x, y, z: Atom(element, [x.value, y.value, z.value])
    monkeypatch.setitem(sys.modules, "samson", module)
    return module
