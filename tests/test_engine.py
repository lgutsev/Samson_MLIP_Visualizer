import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from samson_mlip_visualizer.engine import ModelUncertaintyError, evaluate, relax
from samson_mlip_visualizer.sanity import StructureSanityError


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results = {
            "energy": 0.5 * float((positions**2).sum()),
            "forces": -positions,
        }


def harmonic_atoms():
    atoms = Atoms("HH", positions=[[1.0, 0, 0], [2.0, 0, 0]])
    atoms.calc = HarmonicCalculator()
    return atoms


def test_evaluate_uses_constraint_aware_forces():
    atoms = harmonic_atoms()
    atoms.set_constraint(FixAtoms(indices=[1]))
    result = evaluate(atoms)
    assert result.energy_ev == pytest.approx(2.5)
    assert result.forces_ev_per_angstrom.tolist() == [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert result.max_force_ev_per_angstrom == pytest.approx(1.0)


def test_relax_moves_free_atoms_but_not_fixed_atoms():
    atoms = harmonic_atoms()
    atoms.set_constraint(FixAtoms(indices=[1]))
    reports = []
    result = relax(
        atoms,
        fmax=0.02,
        max_steps=100,
        on_progress=lambda *args: reports.append(args),
    )
    assert result.converged
    assert not result.stopped
    assert abs(atoms.positions[0, 0]) < 0.02
    assert atoms.positions[1, 0] == pytest.approx(2.0)
    assert reports
    assert np.asarray(reports[-1][3]).shape == (2, 3)


def test_relax_can_be_stopped():
    atoms = harmonic_atoms()
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls >= 2

    result = relax(atoms, fmax=1e-12, max_steps=100, should_stop=stop)
    assert result.stopped
    assert not result.converged


@pytest.mark.parametrize("optimizer", ["FIRE", "LBFGS", "BFGS"])
def test_relax_converges_with_each_optimizer(optimizer):
    atoms = harmonic_atoms()
    atoms.set_constraint(FixAtoms(indices=[1]))
    result = relax(atoms, fmax=0.02, max_steps=200, optimizer=optimizer)
    assert result.converged
    assert abs(atoms.positions[0, 0]) < 0.02


def test_relax_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        relax(harmonic_atoms(), optimizer="nope")


class CollapsingCalculator(Calculator):
    """Pulls every atom toward the origin, hard — atoms end up on top of it."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self.results = {"energy": float((positions**2).sum()), "forces": -50.0 * positions}


def test_relax_aborts_on_atomic_collapse():
    atoms = Atoms("HH", positions=[[0.4, 0, 0], [-0.4, 0, 0]])
    atoms.calc = CollapsingCalculator()
    with pytest.raises(StructureSanityError, match="unphysical forces"):
        relax(atoms, fmax=0.01, max_steps=50, min_distance=0.5)


def test_relax_aborts_on_drift():
    atoms = harmonic_atoms()
    with pytest.raises(StructureSanityError, match="moved"):
        relax(atoms, fmax=1e-9, max_steps=50, min_distance=None, max_drift=0.05)


class CommitteeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, force_std, mean_force=0.0):
        super().__init__()
        self._force_std = force_std
        self._mean_force = mean_force

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        forces = np.zeros((n, 3))
        forces[:, 0] = self._mean_force
        spread = np.zeros((3, n, 3))
        spread[1, :, 0] = self._force_std
        spread[2, :, 0] = -self._force_std
        self.results = {
            "energy": 0.0,
            "forces": forces,
            "energy_comm": np.array([0.0, 0.1, -0.1]),
            "forces_comm": forces[None, :, :] + spread,
        }


def test_evaluate_extracts_committee_spread():
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.calc = CommitteeCalculator(force_std=0.2)
    result = evaluate(atoms)
    assert result.energy_std_ev == pytest.approx(np.std([0.0, 0.1, -0.1]))
    assert result.max_force_std_ev_per_angstrom == pytest.approx(np.std([0.2, 0.0, -0.2]))


def test_relax_aborts_when_committee_spread_exceeds_limit():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 3.0]])
    atoms.calc = CommitteeCalculator(force_std=0.5, mean_force=1.0)
    with pytest.raises(ModelUncertaintyError, match="extrapolating"):
        relax(atoms, fmax=0.01, max_steps=10, min_distance=None, max_force_std=0.1)
