import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from samson_mlip_visualizer.engine import evaluate, relax


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
