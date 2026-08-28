import numpy as np
import pytest
from ase import Atoms

from samson_mlip_visualizer.sanity import (
    StructureSanityError,
    check_sane,
    close_contacts,
    max_displacement,
)


def test_close_contacts_flags_overlapping_atoms():
    atoms = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.3], [0, 0, 5.0]])
    contacts = close_contacts(atoms, min_distance=0.5)
    assert contacts == [(0, 1, pytest.approx(0.3))]


def test_close_contacts_honors_pbc():
    atoms = Atoms("H2", positions=[[0.1, 0, 0], [9.9, 0, 0]], cell=[10, 10, 10], pbc=True)
    contacts = close_contacts(atoms, min_distance=0.5)
    assert contacts and contacts[0][2] == pytest.approx(0.2)


def test_close_contacts_empty_when_well_separated():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.0]])
    assert close_contacts(atoms, min_distance=0.5) == []
    assert close_contacts(atoms, min_distance=0.0) == []


def test_max_displacement():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
    reference = np.array([[0, 0, 0], [0, 0, 0.0]])
    assert max_displacement(atoms, reference) == pytest.approx(1.0)


def test_check_sane_raises_on_close_contact():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.2]])
    with pytest.raises(StructureSanityError, match="0.200"):
        check_sane(atoms, min_distance=0.5)


def test_check_sane_raises_on_drift():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 3.0]])
    reference = np.zeros((2, 3))
    with pytest.raises(StructureSanityError, match="moved"):
        check_sane(atoms, min_distance=None, reference_positions=reference, max_drift=1.0)


def test_check_sane_passes_a_reasonable_geometry():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
    check_sane(atoms, min_distance=0.5, reference_positions=atoms.get_positions(), max_drift=1.0)
