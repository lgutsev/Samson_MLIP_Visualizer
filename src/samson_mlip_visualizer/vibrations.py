"""Finite-difference harmonic frequencies, mainly to classify stationary points.

A minimum has no imaginary mode; a first-order saddle (transition state) has
exactly one. The Hessian costs ``6 x (free atoms)`` force calls, so this is meant
for molecules and small clusters, not large periodic cells.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms, units
from ase.constraints import FixAtoms

# sqrt(eV / (Å^2 amu)) in rad/s, divided by 2 pi c, gives wavenumbers in cm^-1.
_TO_WAVENUMBER = np.sqrt(units._e / units._amu) * 1e10 / (2 * np.pi * units._c * 100)


@dataclass(frozen=True)
class FrequencyResult:
    """Harmonic wavenumbers (cm^-1, ascending); imaginary modes are negative.

    ``modes[k]`` is the Cartesian displacement pattern (unit norm, shape
    ``(len(free_indices), 3)``) of ``wavenumbers_cm[k]``. When rigid-body motion
    was projected out, those modes are omitted.
    """

    wavenumbers_cm: np.ndarray
    modes: np.ndarray
    free_indices: tuple[int, ...]
    imaginary_threshold_cm: float
    rigid_body_modes_removed: int = 0

    @property
    def imaginary(self) -> np.ndarray:
        return self.wavenumbers_cm[self.wavenumbers_cm < -self.imaginary_threshold_cm]

    @property
    def n_imaginary(self) -> int:
        return int(self.imaginary.size)

    def classification(self) -> str:
        if self.n_imaginary == 0:
            return "minimum (no imaginary modes)"
        if self.n_imaginary == 1:
            return f"first-order saddle point (one imaginary mode, {self.imaginary[0]:.1f} cm^-1)"
        return f"higher-order saddle point ({self.n_imaginary} imaginary modes)"

    def soft_mode_hint(self, below_cm: float = 100.0) -> str | None:
        """Advice when every imaginary mode is small, which usually means loose convergence."""
        if self.n_imaginary and np.all(np.abs(self.imaginary) < below_cm):
            return (
                f"All imaginary modes are below {below_cm:.0f} cm^-1. That usually means a "
                "floppy mode and a loosely converged geometry rather than a real saddle "
                "point: re-optimize to Fmax <= 0.001 eV/Å in float64 and check again."
            )
        return None


def _free_indices(atoms: Atoms) -> list[int]:
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed.update(int(index) for index in constraint.get_indices())
    return [index for index in range(len(atoms)) if index not in fixed]


def _rigid_body_basis(atoms: Atoms) -> np.ndarray:
    """Orthonormal mass-weighted translation (and, without PBC, rotation) vectors."""
    sqrt_mass = np.sqrt(atoms.get_masses())[:, None]
    centred = atoms.get_positions() - atoms.get_center_of_mass()
    vectors = []
    for axis in np.eye(3):
        vectors.append((np.broadcast_to(axis, centred.shape) * sqrt_mass).ravel())
    if not atoms.pbc.any():
        for axis in np.eye(3):
            vectors.append((np.cross(axis, centred) * sqrt_mass).ravel())
    # SVD rather than QR: a linear molecule's axial rotation is ~1e-18, not exactly
    # zero, and QR would turn it into an arbitrary direction that eats a real mode.
    u, singular, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
    return u[:, singular > 1e-6 * singular.max()]


def _hessian_block(
    atoms: Atoms,
    indices: list[int],
    delta: float,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Symmetrized central-difference Hessian (eV/Å²) over ``indices``; restores positions."""
    reference = atoms.get_positions().copy()
    coordinates = [(atom, axis) for atom in indices for axis in range(3)]
    size = len(coordinates)
    hessian = np.empty((size, size))
    try:
        for column, (atom, axis) in enumerate(coordinates):
            gradients = []
            for sign in (1.0, -1.0):
                displaced = reference.copy()
                displaced[atom, axis] += sign * delta
                atoms.set_positions(displaced, apply_constraint=False)
                forces = atoms.get_forces(apply_constraint=False)
                gradients.append(-forces[indices].ravel())
            hessian[:, column] = (gradients[0] - gradients[1]) / (2 * delta)
            if on_progress:
                on_progress(column + 1, size)
    finally:
        atoms.set_positions(reference, apply_constraint=False)
    return 0.5 * (hessian + hessian.T)


def cartesian_hessian(atoms: Atoms, *, delta: float = 0.01) -> np.ndarray:
    """Full ``(3N, 3N)`` Cartesian Hessian (eV/Å²) by central differences (6N force calls).

    With an MLIP this takes seconds for a molecule, which makes exact Hessians
    (Gaussian's ``CalcFC`` / ``RecalcFC``) cheap. Positions are restored.
    """
    return _hessian_block(atoms, list(range(len(atoms))), delta)


def harmonic_frequencies(
    atoms: Atoms,
    *,
    delta: float = 0.01,
    imaginary_threshold_cm: float = 20.0,
    project_rigid_body: bool | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> FrequencyResult:
    """Central-difference, mass-weighted Hessian of the free (non-FixAtoms) atoms.

    ``project_rigid_body`` removes overall translation (and rotation, for
    non-periodic systems) so they cannot masquerade as soft or imaginary modes.
    The default projects whenever no atom is fixed; fixed atoms, or an external
    field, break that symmetry and must not be projected. Positions are restored.
    """
    free = _free_indices(atoms)
    if not free:
        raise ValueError("Every atom is fixed; there are no vibrational degrees of freedom")
    if project_rigid_body is None:
        project_rigid_body = len(free) == len(atoms)
    if project_rigid_body and len(free) != len(atoms):
        raise ValueError("Rigid-body projection is invalid when some atoms are fixed")

    hessian = _hessian_block(atoms, free, delta, on_progress)
    size = len(hessian)
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(atoms.get_masses()[free]), 3)
    mass_weighted = hessian * np.outer(inverse_sqrt_mass, inverse_sqrt_mass)

    removed = 0
    if project_rigid_body:
        basis = _rigid_body_basis(atoms)
        projector = np.eye(size) - basis @ basis.T
        mass_weighted = projector @ mass_weighted @ projector
        removed = basis.shape[1]

    eigenvalues, eigenvectors = np.linalg.eigh(mass_weighted)
    if removed:
        overlap = np.einsum("ik,ik->k", eigenvectors, basis @ (basis.T @ eigenvectors))
        vibrational = np.argsort(overlap)[: size - removed]
        vibrational.sort()
        eigenvalues, eigenvectors = eigenvalues[vibrational], eigenvectors[:, vibrational]

    wavenumbers = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * _TO_WAVENUMBER
    cartesian = eigenvectors * inverse_sqrt_mass[:, None]
    cartesian /= np.linalg.norm(cartesian, axis=0)
    return FrequencyResult(
        wavenumbers_cm=wavenumbers,
        modes=cartesian.T.reshape(-1, len(free), 3),
        free_indices=tuple(free),
        imaginary_threshold_cm=imaginary_threshold_cm,
        rigid_body_modes_removed=removed,
    )
