"""Hamiltonian export for MACE-H: orbital order, the DeepH files, and (if Psi4 is
installed) that the exported blocks rotate with the molecule."""

import numpy as np
import pytest
from ase.build import molecule

from samson_mlip_visualizer.hamiltonian import (
    Psi4HamiltonianCalculator,
    assemble,
    openmx_order,
    orbital_energies,
    orbital_slices,
    read_blocks,
    read_orbital_types,
    write_deeph,
)
from samson_mlip_visualizer.psi4_backend import find_psi4

# The real harmonics of each shell in OpenMX/DeepH order, up to a common factor.
_S3 = np.sqrt(3.0)
HARMONICS = {
    0: lambda x, y, z: [np.ones_like(x)],
    1: lambda x, y, z: [x, y, z],
    2: lambda x, y, z: [(3 * z * z - (x * x + y * y + z * z)) / 2, _S3 / 2 * (x * x - y * y),
                        _S3 * x * y, _S3 * x * z, _S3 * y * z],
}


def wigner(ell, rotation):
    """D with f_k(Rr) = Σ_m D_km f_m(r), fitted on random points: Psi4 keeps lab-frame
    harmonics, so a rotated molecule's functions are the old ones mixed by D."""
    points = np.random.default_rng(0).normal(size=(50, 3))
    before = np.array(HARMONICS[ell](*points.T)).T
    after = np.array(HARMONICS[ell](*(points @ rotation.T).T)).T
    return np.linalg.lstsq(before, after, rcond=None)[0].T


def test_openmx_order_permutes_each_shell_and_groups_atoms():
    # Psi4 order: atom 0 s p d, atom 1 s p (p: z x y; d: 0 +1 -1 +2 -2).
    shells = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]
    order, types = openmx_order(shells, 2)
    assert types == [[0, 1, 2], [0, 1]]
    assert order.tolist() == [0, 2, 3, 1, 4, 7, 8, 5, 6, 9, 11, 12, 10]
    with pytest.raises(ValueError, match="l = 4"):
        openmx_order([[0, 4]], 1)


def test_deeph_folder_round_trip(tmp_path):
    pytest.importorskip("h5py")
    atoms = molecule("H2O")
    types = [[0, 0, 1], [0], [0]]
    size = sum(2 * ell + 1 for t in types for ell in t)
    rng = np.random.default_rng(1)
    h = rng.normal(size=(size, size))
    h = h + h.T
    s = np.eye(size) + 0.01 * (h + h.T)
    folder = write_deeph(tmp_path / "w", atoms, types, hamiltonian=h, overlap=s)
    assert read_orbital_types(folder) == types
    blocks = read_blocks(folder / "hamiltonians.h5")
    assert len(blocks) == 9 and (0, 0, 0, 1, 3) in blocks
    np.testing.assert_allclose(assemble(blocks, types), h)
    np.testing.assert_allclose(assemble(read_blocks(folder / "overlaps.h5"), types), s)
    positions = np.loadtxt(folder / "site_positions.dat").T
    np.testing.assert_allclose(positions - positions[0], atoms.positions - atoms.positions[0])
    with pytest.raises(ValueError, match="vacuum"):
        write_deeph(tmp_path / "small", atoms, types, overlap=s, box=10.0)


@pytest.mark.skipif(find_psi4() is None, reason="Psi4 is not installed")
def test_real_psi4_blocks_rotate_with_the_molecule():
    from scipy.spatial.transform import Rotation

    calc = Psi4HamiltonianCalculator(find_psi4(), method="pbe", basis="def2-svp", threads=2)
    try:
        atoms = molecule("H2O")
        atoms.rattle(0.05, seed=1)
        rotation = Rotation.random(random_state=3).as_matrix()
        results = []
        for positions in (atoms.positions, atoms.positions @ rotation.T):
            copy = atoms.copy()
            copy.positions = positions
            copy.calc = calc
            copy.get_potential_energy()
            results.append(dict(calc.results))
    finally:
        calc.close()
    first, second = results
    types = first["orbital_types"]
    assert types == [[0, 0, 0, 1, 1, 2], [0, 0, 1], [0, 0, 1]]  # def2-SVP O, H
    eps = orbital_energies(first["hamiltonian"], first["overlap"])
    np.testing.assert_allclose(eps, first["eigenvalues"], atol=1e-6)
    per_atom = []  # block-diagonal rotation of each atom's orbitals
    for shells in types:
        blocks = [wigner(ell, rotation) for ell in shells]
        size = sum(b.shape[0] for b in blocks)
        full, offset = np.zeros((size, size)), 0
        for b in blocks:
            full[offset:offset + len(b), offset:offset + len(b)] = b
            offset += len(b)
        per_atom.append(full)
    slices = orbital_slices(types)
    for key in ("hamiltonian", "overlap"):
        for i, rows in enumerate(slices):
            for j, cols in enumerate(slices):
                expected = per_atom[i] @ first[key][rows, cols] @ per_atom[j].T
                np.testing.assert_allclose(second[key][rows, cols], expected, atol=1e-6)
