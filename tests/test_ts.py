import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from samson_mlip_visualizer.ts import dimer_search, initial_mode
from samson_mlip_visualizer.vibrations import harmonic_frequencies


class DoubleWell(Calculator):
    """V = (x^2 - 1)^2 + y^2 + z^2 on atom 0: minima at x = ±1, saddle at the origin."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        forces = np.zeros((len(atoms), 3))
        x, y, z = atoms.positions[0]
        forces[0] = [-4 * x * (x * x - 1), -2 * y, -2 * z]
        self.results = {"energy": (x * x - 1) ** 2 + y * y + z * z, "forces": forces}


def well(position):
    atoms = Atoms("H", positions=[position])
    atoms.calc = DoubleWell()
    return atoms


def test_dimer_finds_saddle_and_reports_progress():
    atoms = well([0.4, 0.2, -0.1])
    reports = []
    result = dimer_search(
        atoms,
        fmax=1e-3,
        max_steps=200,
        seed=0,
        min_distance=None,
        on_progress=lambda *args: reports.append(args),
    )
    assert result.converged and not result.stopped
    assert atoms.positions[0] == pytest.approx([0, 0, 0], abs=1e-3)
    assert result.evaluation.energy_ev == pytest.approx(1.0, abs=1e-6)
    assert result.curvature == pytest.approx(-4.0, rel=1e-3)
    assert reports and reports[-1][3] < 0


def test_dimer_can_be_stopped():
    result = dimer_search(well([0.4, 0.2, 0.0]), should_stop=lambda: True, min_distance=None)
    assert result.stopped and not result.converged


def test_initial_mode_stretches_pair_and_respects_fixed_atoms():
    atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    mode = initial_mode(atoms, pair=(0, 1), magnitude=0.1)
    assert np.linalg.norm(mode) == pytest.approx(0.1)
    assert (mode[1] - mode[0])[0] > 0.05  # the pair separates along its axis
    assert mode.sum(axis=0) == pytest.approx([0, 0, 0], abs=1e-12)  # no net translation

    atoms.set_constraint(FixAtoms(indices=[2]))
    mode = initial_mode(atoms, pair=(0, 1), magnitude=0.1)
    assert mode[0] == pytest.approx([-0.1 / np.sqrt(2), 0, 0])
    assert mode[1] == pytest.approx([0.1 / np.sqrt(2), 0, 0])
    assert not mode[2].any()

    atoms.set_constraint(FixAtoms(indices=[0]))
    mode = initial_mode(atoms, pair=(0, 1), magnitude=0.1)
    assert not mode[0].any()
    assert np.linalg.norm(mode) == pytest.approx(0.1)
    random = initial_mode(atoms, magnitude=0.2, seed=1)
    assert not random[0].any() and np.linalg.norm(random) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="Invalid atom pair"):
        initial_mode(atoms, pair=(1, 1))
    with pytest.raises(ValueError, match="not both"):
        initial_mode(atoms, pair=(0, 1), use_hessian=True)


def test_frequencies_classify_minimum_and_saddle():
    # The double well is an external field, so rigid-body projection must be off.
    saddle = harmonic_frequencies(well([0.0, 0.0, 0.0]), project_rigid_body=False)
    assert saddle.n_imaginary == 1
    assert "first-order saddle" in saddle.classification()
    # Curvatures -4 and 2 (eV/Å^2): wavenumbers scale with sqrt(|k|).
    ratio = abs(saddle.wavenumbers_cm[0]) / saddle.wavenumbers_cm[1]
    assert ratio == pytest.approx(np.sqrt(2), rel=1e-3)

    minimum = well([1.0, 0.0, 0.0])
    reference = minimum.positions.copy()
    result = harmonic_frequencies(minimum, project_rigid_body=False)
    assert result.n_imaginary == 0
    assert result.classification().startswith("minimum")
    assert minimum.positions == pytest.approx(reference)


def test_frequencies_skip_fixed_atoms():
    atoms = Atoms("H2", positions=[[0, 0, 0], [5, 5, 5]])
    atoms.calc = DoubleWell()
    atoms.set_constraint(FixAtoms(indices=[1]))
    result = harmonic_frequencies(atoms)
    assert result.free_indices == (0,)
    assert result.wavenumbers_cm.shape == (3,)


def relaxed_copper(positions):
    atoms = Atoms(f"Cu{len(positions)}", positions=positions)
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-4)
    return atoms


def test_projection_leaves_3n_minus_6_modes():
    tetrahedron = relaxed_copper(
        [[0, 0, 0], [2.5, 0, 0], [1.25, 2.17, 0], [1.25, 0.72, 2.04]]
    )
    result = harmonic_frequencies(tetrahedron)
    assert result.rigid_body_modes_removed == 6
    assert result.wavenumbers_cm.shape == (6,)
    assert result.n_imaginary == 0
    assert result.wavenumbers_cm.min() > 20
    assert result.soft_mode_hint() is None


def test_projection_handles_linear_molecules():
    dimer = relaxed_copper([[0, 0, 0], [2.3, 0, 0]])
    result = harmonic_frequencies(dimer)
    assert result.rigid_body_modes_removed == 5
    (stretch,) = result.modes
    assert abs(stretch[0, 0]) == pytest.approx(np.sqrt(0.5), abs=1e-3)
    with pytest.raises(ValueError, match="fixed"):
        dimer.set_constraint(FixAtoms(indices=[0]))
        harmonic_frequencies(dimer, project_rigid_body=True)


def test_hessian_mode_guides_dimer_on_emt_cluster():
    atoms = relaxed_copper([[0, 0, 0], [2.5, 0, 0], [1.25, 2.17, 0], [1.25, 0.72, 2.04]])
    mode = initial_mode(atoms, use_hessian=True, magnitude=0.1)
    assert np.linalg.norm(mode) == pytest.approx(0.1)
    assert mode.sum(axis=0) == pytest.approx([0, 0, 0], abs=1e-9)


def test_soft_imaginary_modes_get_a_convergence_hint():
    from samson_mlip_visualizer.vibrations import FrequencyResult

    soft = FrequencyResult(np.array([-40.0, 150.0]), np.zeros((2, 1, 3)), (0,), 20.0)
    assert "loosely converged" in soft.soft_mode_hint()
    real = FrequencyResult(np.array([-600.0, 150.0]), np.zeros((2, 1, 3)), (0,), 20.0)
    assert real.soft_mode_hint() is None
