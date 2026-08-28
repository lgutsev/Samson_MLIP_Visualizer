import pytest
from ase import Atoms

from samson_mlip_visualizer.compat import (
    ModelCompatibilityError,
    assert_model_covers_structure,
    supported_species,
)


class MaceLikeZTable:
    def __init__(self, zs):
        self.zs = zs


class MaceLikeCalculator:
    def __init__(self, zs):
        self.z_table = MaceLikeZTable(zs)


class DeepMDLikeCalculator:
    def __init__(self, type_map):
        self._type_map = type_map

    def get_type_map(self):
        return self._type_map


def test_supported_species_from_mace_z_table():
    calc = MaceLikeCalculator([1, 8, 28])
    assert supported_species(calc) == {"H", "O", "Ni"}


def test_supported_species_from_deepmd_type_map():
    calc = DeepMDLikeCalculator(["O", "H"])
    assert supported_species(calc) == {"O", "H"}


def test_unknown_calculator_yields_none():
    assert supported_species(object()) is None


def test_assert_passes_when_structure_is_covered():
    calc = MaceLikeCalculator([1, 8])
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert assert_model_covers_structure(calc, atoms) == ["H", "O"]


def test_assert_rejects_uncovered_element():
    calc = MaceLikeCalculator([1, 8])
    atoms = Atoms("NiO", positions=[[0, 0, 0], [0, 0, 2]])
    with pytest.raises(ModelCompatibilityError, match="Ni"):
        assert_model_covers_structure(calc, atoms)


def test_assert_is_silent_when_model_cannot_be_read():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    assert assert_model_covers_structure(object(), atoms) is None
