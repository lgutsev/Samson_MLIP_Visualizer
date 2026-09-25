import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from test_ts import relaxed_copper

from samson_mlip_visualizer.reaction_path import irc, qst
from samson_mlip_visualizer.ts import prfo_search

pytest.importorskip("sella")


class BondWell(Calculator):
    """A diatomic with a double-well bond: minima at 1.5 and 2.5 Å, a 1 eV barrier at 2 Å.

    It depends only on the bond length, so translation and rotation are free, as
    for a real molecule, and the stretch at 2 Å is a genuine first-order saddle.
    """

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        vector = atoms.positions[0] - atoms.positions[1]
        r = np.linalg.norm(vector)
        u = r - 2.0
        energy = 16.0 * (u * u - 0.25) ** 2
        derivative = 64.0 * u * (u * u - 0.25)
        force = -derivative * vector / r
        self.results = {"energy": energy, "forces": np.array([force, -force])}


def diatomic(r):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    atoms.calc = BondWell()
    return atoms


def bond(positions):
    return float(np.linalg.norm(positions[0] - positions[1]))


def test_prfo_finds_the_bond_saddle():
    atoms = diatomic(1.85)
    reports = []
    result = prfo_search(atoms, fmax=1e-4, on_progress=lambda *args: reports.append(args))
    assert result.converged and not result.stopped
    assert bond(atoms.positions) == pytest.approx(2.0, abs=1e-4)
    assert result.evaluation.energy_ev == pytest.approx(1.0, abs=1e-6)
    assert reports


def test_prfo_can_be_stopped():
    result = prfo_search(diatomic(1.85), should_stop=lambda: True)
    assert result.stopped and not result.converged


def test_irc_descends_to_both_minima():
    atoms = diatomic(2.0)
    seen = []
    path = irc(
        atoms, step=0.05, fmax=1e-3, on_progress=lambda direction, *rest: seen.append(direction)
    )
    ends = sorted([bond(path.forward[-1].positions), bond(path.reverse[-1].positions)])
    assert ends == pytest.approx([1.5, 2.5], abs=0.05)
    assert path.forward_minimum_ev == pytest.approx(0.0, abs=1e-6)
    assert path.reverse_minimum_ev == pytest.approx(0.0, abs=1e-6)
    frames = path.frames(atoms.positions)
    energies = [frame.energy_ev for frame in frames]
    peak = len(path.reverse)
    assert energies[peak] == pytest.approx(1.0)
    assert np.all(np.diff(energies[: peak + 1]) >= 0) and np.all(np.diff(energies[peak:]) <= 0)
    assert [frame.arc for frame in frames] == sorted(frame.arc for frame in frames)
    assert bond(atoms.positions) == pytest.approx(2.0)  # left at the TS
    assert {"forward", "reverse"} <= set(seen)


def test_irc_refuses_a_minimum():
    atoms = relaxed_copper([[0, 0, 0], [2.5, 0, 0], [1.25, 2.17, 0], [1.25, 0.72, 2.04]])
    with pytest.raises(ValueError, match="not a transition state"):
        irc(atoms)


@pytest.mark.parametrize("use_guess", [False, True])
def test_qst_finds_barrier_between_minima(use_guess):
    reactant, product = diatomic(1.5), diatomic(2.5)
    guess = diatomic(2.1) if use_guess else None
    energies_seen = []
    result = qst(
        reactant, product, BondWell(), guess=guess, images=5, fmax=0.01,
        on_progress=lambda step, energies, max_force: energies_seen.append(energies),
    )
    assert result.neb_converged and not result.stopped
    assert len(result.images) == 7 and len(result.energies_ev) == 7
    assert result.ts.converged
    assert bond(result.ts_positions) == pytest.approx(2.0, abs=1e-3)
    assert result.barrier_forward_ev == pytest.approx(1.0, abs=1e-4)
    assert result.barrier_reverse_ev == pytest.approx(1.0, abs=1e-4)
    assert energies_seen


def test_qst_checks_atom_order():
    first = Atoms("CuAu", positions=[[0, 0, 0], [2.5, 0, 0]])
    second = Atoms("AuCu", positions=[[0, 0, 0], [2.5, 0, 0]])
    with pytest.raises(ValueError, match="same atoms in the same order"):
        qst(first, second, EMT())
