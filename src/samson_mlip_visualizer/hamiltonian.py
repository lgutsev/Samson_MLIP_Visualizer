"""Electronic Hamiltonians in an atomic-orbital basis, for MACE-H.

MACE-H (https://github.com/maurergroup/MACE-H, arXiv:2508.15108) predicts the
Kohn-Sham Hamiltonian matrix of a structure block by block, one block per atom
pair, instead of an energy. It reads the DeepH-E3 data layout, which the DFT
codes it supports (OpenMX, ABACUS, FHI-aims) are converted into; this module
writes that layout from Psi4 so molecules labeled here can be trained on:

- :class:`Psi4HamiltonianCalculator` returns the converged Kohn-Sham matrix and
  the overlap (or only the overlap, which needs no SCF and is all MACE-H needs to
  predict a new structure);
- :func:`write_deeph` writes one structure folder (``element.dat``,
  ``site_positions.dat``, ``lat.dat``, ``rlat.dat``, ``orbital_types.dat``,
  ``info.json``, ``hamiltonians.h5``, ``overlaps.h5``);
- :func:`read_blocks` / :func:`assemble` read ``hamiltonians.h5`` or MACE-H's
  ``hamiltonians_pred.h5`` back into a full matrix, and :func:`orbital_energies`
  diagonalizes it against the overlap.

Conventions of the DeepH format: energies in eV, lengths in Å, blocks keyed
``"[Rx, Ry, Rz, i, j]"`` (lattice translation, 1-based atoms), orbitals of each
shell in OpenMX's real-harmonic order. A molecule sits in a cubic vacuum box
large enough that no periodic image comes within the model cutoff, so every
block has ``R = 0``. MACE-H needs every atom of an element to carry the same
shells, which a Gaussian basis set per element does.

MACE-H itself runs in its own environment (it needs e3nn 0.5, mace-torch
pins 0.4.4); this module needs only NumPy, plus h5py for the ``.h5`` files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.units import Hartree

from .psi4_backend import Psi4Calculator

# Psi4 orders the functions of a pure shell m = 0, +1, -1, +2, -2, ...; OpenMX
# (and so DeepH/MACE-H) orders p as (x, y, z) = (+1, -1, 0) and d as
# (z², x²-y², xy, xz, yz) = (0, +2, -2, +1, -1). f happens to agree. Both use the
# real harmonics without the Condon-Shortley sign, so a permutation is enough
# (tests/test_hamiltonian.py checks rotation equivariance against MACE-H's own
# rotation code when it is importable).
PSI4_TO_OPENMX = {0: [0], 1: [1, 2, 0], 2: [0, 3, 4, 1, 2], 3: [0, 1, 2, 3, 4, 5, 6]}
BOX = 40.0  # Å, edge of the vacuum box a molecule is written in


class Psi4HamiltonianCalculator(Psi4Calculator):
    """Psi4 Kohn-Sham matrix and overlap, in DeepH order and eV.

    ``results`` gets ``"hamiltonian"`` (eV), ``"overlap"``, ``"orbital_types"``
    (angular momenta of each atom's shells), ``"eigenvalues"`` (eV),
    ``"n_occupied"``, and ``"energy"``. With ``overlap_only=True`` only the
    overlap and the orbital types are computed (no SCF). Closed shells only.
    """

    implemented_properties = ["energy", "hamiltonian", "overlap"]

    def __init__(self, python, *, overlap_only: bool = False, **kwargs):
        super().__init__(python, **kwargs)
        if self.multiplicity != 1:
            raise ValueError("Hamiltonian export supports closed shells only")
        self.overlap_only = overlap_only

    def request(self, atoms) -> dict:
        return {**super().request(atoms), "task": "matrices", "overlap_only": self.overlap_only}

    def results_from(self, answer: dict, atoms) -> dict:
        order, orbital_types = openmx_order(answer["shells"], len(atoms))
        overlap = np.asarray(answer["overlap"], float)[np.ix_(order, order)]
        results = {"overlap": overlap, "orbital_types": orbital_types}
        if "hamiltonian" in answer:
            hamiltonian = np.asarray(answer["hamiltonian"], float)[np.ix_(order, order)]
            energy = float(answer["energy"]) * Hartree
            results.update(
                hamiltonian=hamiltonian * Hartree,
                eigenvalues=np.asarray(answer["eigenvalues"], float) * Hartree,
                n_occupied=int(answer["n_occupied"]),
                energy=energy,
                free_energy=energy,
            )
        return results


def openmx_order(shells, n_atoms: int) -> tuple[np.ndarray, list[list[int]]]:
    """AO permutation from Psi4 to DeepH order, and each atom's shell momenta.

    ``shells`` lists ``[atom, l]`` per shell in Psi4's AO order. The permutation
    also groups the functions atom by atom (Psi4 already does for one molecule).
    """
    starts, offset = [], 0
    for _, ell in shells:
        starts.append(offset)
        offset += 2 * ell + 1
    order, orbital_types = [], [[] for _ in range(n_atoms)]
    for atom in range(n_atoms):
        for (center, ell), start in zip(shells, starts, strict=True):
            if center == atom:
                if ell not in PSI4_TO_OPENMX:
                    raise ValueError(f"Shells with l = {ell} are not supported")
                order.extend(start + k for k in PSI4_TO_OPENMX[ell])
                orbital_types[atom].append(ell)
    return np.asarray(order), orbital_types


def orbital_slices(orbital_types) -> list[slice]:
    """Each atom's rows/columns in the full AO matrix."""
    slices, offset = [], 0
    for types in orbital_types:
        size = sum(2 * ell + 1 for ell in types)
        slices.append(slice(offset, offset + size))
        offset += size
    return slices


def write_deeph(folder, atoms, orbital_types, *, hamiltonian=None, overlap=None,
                box: float = BOX, info: dict | None = None) -> Path:
    """One DeepH-E3 structure folder for a molecule (all blocks at ``R = 0``).

    ``hamiltonian`` (eV) and ``overlap`` are full matrices in DeepH order, as
    :class:`Psi4HamiltonianCalculator` returns them; ``hamiltonians.h5`` is only
    written for training data, ``overlaps.h5`` is what MACE-H builds the graph
    from when it predicts. The molecule is centered in a cubic box of edge ``box``.
    """
    import h5py

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    positions = atoms.positions - atoms.positions.mean(axis=0) + box / 2
    span = np.ptp(atoms.positions, axis=0).max() if len(atoms) > 1 else 0.0
    if box - span < 15.0:
        raise ValueError(f"A {box:g} Å box leaves too little vacuum around a {span:.1f} Å molecule")
    lattice = np.eye(3) * box
    np.savetxt(folder / "element.dat", atoms.numbers, fmt="%d")
    # DeepH stores lattice vectors and positions as columns.
    np.savetxt(folder / "site_positions.dat", positions.T)
    np.savetxt(folder / "lat.dat", lattice.T)
    np.savetxt(folder / "rlat.dat", (2 * np.pi * np.linalg.inv(lattice)))
    with open(folder / "orbital_types.dat", "w", encoding="utf-8") as handle:
        for types in orbital_types:
            handle.write(" ".join(map(str, types)) + "\n")
    with open(folder / "info.json", "w", encoding="utf-8") as handle:
        json.dump({"isspinful": False, "fermi_level": 0.0, **(info or {})}, handle)
    slices = orbital_slices(orbital_types)
    for name, matrix in (("hamiltonians.h5", hamiltonian), ("overlaps.h5", overlap)):
        if matrix is None:
            continue
        with h5py.File(folder / name, "w") as handle:
            for i, rows in enumerate(slices):
                for j, cols in enumerate(slices):
                    handle[json.dumps([0, 0, 0, i + 1, j + 1])] = np.asarray(matrix)[rows, cols]
    return folder


def read_blocks(path) -> dict[tuple[int, ...], np.ndarray]:
    """Blocks of a DeepH ``.h5`` file, keyed ``(Rx, Ry, Rz, i, j)`` (1-based)."""
    import h5py

    with h5py.File(path, "r") as handle:
        return {tuple(json.loads(key)): np.asarray(value) for key, value in handle.items()}


def assemble(blocks, orbital_types) -> np.ndarray:
    """The full molecular matrix from ``R = 0`` blocks (missing blocks are 0)."""
    slices = orbital_slices(orbital_types)
    size = slices[-1].stop
    matrix = np.zeros((size, size))
    for (rx, ry, rz, i, j), block in blocks.items():
        if (rx, ry, rz) == (0, 0, 0):
            matrix[slices[i - 1], slices[j - 1]] = block
    return matrix


def read_orbital_types(folder) -> list[list[int]]:
    lines = (Path(folder) / "orbital_types.dat").read_text(encoding="utf-8").splitlines()
    return [list(map(int, line.split())) for line in lines if line.strip()]


def orbital_energies(hamiltonian, overlap) -> np.ndarray:
    """Eigenvalues of H c = ε S c (same units as ``hamiltonian``), ascending."""
    from scipy.linalg import eigh

    symmetric = (np.asarray(hamiltonian) + np.asarray(hamiltonian).T) / 2
    return eigh(symmetric, overlap, eigvals_only=True)
