import numpy as np
import pytest
from samson_fakes import Atom, FakeSamson, Model, UnitCell, install_samson_module

from samson_mlip_visualizer.samson_bridge import (
    SamsonBridgeError,
    choose_structural_models,
    extract_structure,
    node_name,
    selected_atom_indices,
    set_selection_flag,
    sync_positions,
)


def test_requires_selection_when_several_models():
    samson = FakeSamson([Model([]), Model([])])
    with pytest.raises(SamsonBridgeError, match="multiple structural models"):
        choose_structural_models(samson)


def test_selected_model_wins():
    selected = Model([], selected=True)
    assert choose_structural_models(FakeSamson([Model([]), selected])) == [selected]


def test_selected_models_are_combined():
    first = Model([Atom("O", [0, 0, 0]), Atom("H", [1, 0, 0])], selected=True)
    second = Model([Atom("O", [3, 0, 0])], selected=True, cell=None)
    ignored = Model([Atom("Na", [9, 9, 9])])
    structure = extract_structure(FakeSamson([first, ignored, second]))
    assert structure.models == [first, second]
    assert structure.ase_atoms.get_chemical_symbols() == ["O", "H", "O"]
    assert structure.samson_atoms == first.atoms + second.atoms
    assert structure.ase_atoms.cell.lengths().tolist() == pytest.approx([5, 6, 20])


def test_explicit_models_override_selection():
    first = Model([Atom("O", [0, 0, 0])], selected=True)
    second = Model([Atom("Na", [3, 0, 0])])
    structure = extract_structure(FakeSamson([first, second]), models=[first, second])
    assert structure.ase_atoms.get_chemical_symbols() == ["O", "Na"]


def test_reports_selected_atom_indices():
    atoms = [Atom("O", [0, 0, 0]), Atom("H", [1, 0, 0]), Atom("H", [0, 1, 0])]
    atoms[0].selectionFlag = True
    atoms[2].selectionFlag = True
    structure = extract_structure(FakeSamson([Model(atoms)]))
    assert selected_atom_indices(structure) == [0, 2]


def test_rejects_conflicting_cells():
    first = Model([Atom("O", [0, 0, 0])], selected=True)
    second = Model([Atom("O", [3, 0, 0])], selected=True, cell=UnitCell(a=7))
    with pytest.raises(SamsonBridgeError, match="different unit cells"):
        extract_structure(FakeSamson([first, second]))


def test_models_without_cells_are_nonperiodic():
    models = [Model([Atom("O", [0, 0, 0])], selected=True, cell=None) for _ in range(2)]
    structure = extract_structure(FakeSamson(models))
    assert structure.ase_atoms.get_pbc().tolist() == [False, False, False]


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
    install_samson_module(monkeypatch)

    positions = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
    sync_positions(structure, positions, samson=samson)
    assert atoms[0].position == pytest.approx([0.1, 0.2, 0.3])
    assert atoms[1].position == pytest.approx([1.1, 1.2, 1.3])
    assert samson.events == 1

    sync_positions(structure, positions + 1, samson=samson, process_events=False)
    assert atoms[0].position == pytest.approx([1.1, 1.2, 1.3])
    assert samson.events == 1


def test_set_selection_flag_prefers_setter_then_attribute():
    class WithSetter:
        def __init__(self):
            self.calls = []

        def setSelectionFlag(self, value):
            self.calls.append(value)

    node = WithSetter()
    set_selection_flag(node, True)
    assert node.calls == [True]

    atom = Atom("H", [0, 0, 0])
    set_selection_flag(atom, True)
    assert atom.selectionFlag is True

    class ReadOnly:
        @property
        def selectionFlag(self):
            return False

    with pytest.raises(SamsonBridgeError, match="does not allow changing the selection"):
        set_selection_flag(ReadOnly(), True)


def test_node_name_reads_property_or_getter():
    assert node_name(Model([], name="water")) == "water"

    class WithGetter:
        def getName(self):
            return "slab"

    assert node_name(WithGetter()) == "slab"
    assert node_name(object()) == ""
