"""Free energies along a reaction coordinate, checked against exact answers."""

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from samson_mlip_visualizer import free_energy as fe

T = 300.0
KT = fe.KB * T


class Bonds(Calculator):
    """Harmonic bonds ½k(d − r0)² between the given pairs (none: free particles)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, pairs=(), k=2.0, r0=1.5):
        super().__init__()
        self.pairs, self.k, self.r0 = list(pairs), k, r0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = self.atoms.positions
        energy, forces = 0.0, np.zeros_like(x)
        for a, b in self.pairs:
            v = x[a] - x[b]
            d = np.linalg.norm(v)
            energy += 0.5 * self.k * (d - self.r0) ** 2
            f = -self.k * (d - self.r0) * v / d
            forces[a] += f
            forces[b] -= f
        self.results = {"energy": energy, "forces": forces}


def three_atoms(xi):
    """Atom 0 with atoms 1 and 2 on springs; ξ = d01 − d02 set to ``xi``."""
    r1, r2 = 1.5 + xi / 2, 1.5 - xi / 2
    atoms = Atoms("CHH", positions=[[0, 0, 0], [r1, 0, 0], [0, r2, 0]])
    atoms.calc = Bonds([(0, 1), (0, 2)])
    return atoms


def exact_free_energy(grid, k=2.0, r0=1.5):
    """A(ξ) for ξ = r1 − r2 with independent springs: P(ξ) ∝ ∫ r1² r2² e^{−βU} dr2."""
    r2 = np.linspace(0.3, 3.0, 4000)
    density = []
    for xi in grid:
        r1 = r2 + xi
        integrand = np.where(r1 > 0, r1**2 * r2**2 * np.exp(
            -(0.5 * k * (r1 - r0) ** 2 + 0.5 * k * (r2 - r0) ** 2) / KT), 0.0)
        density.append(np.trapezoid(integrand, r2))
    return -KT * np.log(np.array(density))


def test_coordinate_gradient_and_curvature_match_finite_differences():
    coordinate = fe.DistanceCombination([(0, 1, 1.0), (0, 2, -1.0)])
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 3)) * 1.2
    m = np.array([12.0, 3.0, 35.0])
    step = 1e-6
    numeric = np.zeros_like(x)
    for i in range(3):
        for j in range(3):
            plus, minus = x.copy(), x.copy()
            plus[i, j] += step
            minus[i, j] -= step
            numeric[i, j] = (coordinate.value(plus) - coordinate.value(minus)) / (2 * step)
    assert coordinate.gradient(x) == pytest.approx(numeric, abs=1e-6)
    # G from a finite-difference Hessian: Z⁻² Σ (∇ξ/m)ᵀ H (∇ξ/m).
    hessian = np.zeros((9, 9))
    for n in range(9):
        plus, minus = x.copy().reshape(-1), x.copy().reshape(-1)
        plus[n] += 1e-5
        minus[n] -= 1e-5
        hessian[n] = (coordinate.gradient(plus.reshape(3, 3)) -
                      coordinate.gradient(minus.reshape(3, 3))).reshape(-1) / 2e-5
    v = (coordinate.gradient(x) / m[:, None]).reshape(-1)
    expected = v @ hessian @ v / coordinate.z(x, m) ** 2
    assert coordinate.g(x, m) == pytest.approx(expected, rel=1e-5)


def test_free_pair_mean_force_is_the_centrifugal_term():
    """Two free atoms held at distance r: dA/dr = −2 k_B T / r exactly."""
    atoms = Atoms("HH", positions=[[0, 0, 0], [1.5, 0, 0]])
    atoms.calc = Bonds()
    record = fe.constrained_md(atoms, fe.DistanceCombination([(0, 1, 1.0)]), steps=20000,
                               temperature_k=T, timestep_fs=0.5, andersen_probability=0.1)
    assert np.allclose(record.value, 1.5, atol=1e-8)
    gradient, error = fe.blue_moon_gradient(record, T, skip=500)
    assert gradient == pytest.approx(-2 * KT / 1.5, rel=0.08)
    assert error < 0.1 * abs(gradient)


def test_blue_moon_matches_the_exact_free_energy_with_a_varying_metric():
    """ξ = d01 − d02: Z depends on the angle, so the G term matters."""
    coordinate = fe.DistanceCombination([(0, 1, 1.0), (0, 2, -1.0)])
    grid = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    measured = []
    for xi in grid:
        atoms = three_atoms(xi)
        record = fe.constrained_md(atoms, coordinate, steps=15000, temperature_k=T,
                                   timestep_fs=0.5, andersen_probability=0.1,
                                   masses=[12.0, 3.0, 35.0], seed=int(10 * xi) + 7)
        measured.append(fe.blue_moon_gradient(record, T, skip=500)[0])
    fine = np.linspace(-0.7, 0.7, 141)
    exact_gradient = np.interp(grid, fine[1:-1], np.gradient(exact_free_energy(fine), fine)[1:-1])
    assert np.array(measured) == pytest.approx(exact_gradient, abs=0.12 * np.abs(
        exact_gradient).max())
    # Integrated: the profile difference across the grid.
    profile = fe.integrate_gradient(grid, measured)
    exact = exact_free_energy(grid)
    assert profile[-1] == pytest.approx(exact[-1] - exact[0], abs=2 * KT)


def test_slow_growth_follows_the_target_and_integrates_lambda():
    coordinate = fe.DistanceCombination([(0, 1, 1.0), (0, 2, -1.0)])
    atoms = three_atoms(-0.5)
    record = fe.constrained_md(atoms, coordinate, steps=4000, temperature_k=T,
                               timestep_fs=0.5, increment=2.5e-4, andersen_probability=0.1,
                               masses=[12.0, 3.0, 35.0])
    assert record.value == pytest.approx(record.target, abs=1e-8)
    assert record.target[-1] == pytest.approx(-0.5 + 4000 * 2.5e-4)
    xi, work = fe.slow_growth_profile(record)
    exact = exact_free_energy(np.array([xi[0], xi[-1]]))
    # A fast, noisy single run: only the sign and rough size of the symmetric profile.
    assert abs(work[-1] - (exact[1] - exact[0])) < 0.1


def test_metadynamics_bias_forces_and_reconstruction():
    coordinate = fe.DistanceCombination([(0, 1, 1.0), (0, 2, -1.0)])
    atoms = three_atoms(0.2)
    bias = fe.MetadynamicsCalculator(Bonds([(0, 1), (0, 2)]), coordinate, height=0.05,
                                     sigma=0.1, temperature_k=T, bias_factor=6.0,
                                     walls=[("xi_upper", None, 0.25, 50.0),
                                            ("distance_upper", (0, 1), 1.6, 50.0)])
    for shift in (0.0, 0.05, 0.1):
        atoms.positions[1, 0] += shift
        bias.deposit(atoms)
    assert bias.heights[1] < bias.heights[0]  # well-tempered: later hills are lower
    atoms.calc = bias
    forces = atoms.get_forces()
    step = 1e-5
    for i, j in ((1, 0), (2, 1), (0, 0)):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[i, j] += step
        minus.positions[i, j] -= step
        plus.calc = minus.calc = bias
        numeric = -(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * step)
        assert forces[i, j] == pytest.approx(numeric, abs=1e-5)
    grid = np.linspace(-0.5, 0.8, 50)
    fes = fe.fes_from_hills(grid, bias.centers, bias.heights, 0.1, 6.0)
    assert fes.min() == 0.0 and fes[0] > fes[np.argmin(np.abs(grid - 0.3))]


def test_rate_helpers():
    # A single heavy distance: Z = 1/m1 + 1/m2.
    z = np.full(100, 1 / 12.0 + 1 / 35.0)
    velocity = fe.generalized_velocity(z, T)
    expected = np.sqrt(2 * KT / np.pi) * np.sqrt(1 / 12.0 + 1 / 35.0) * units.s
    assert velocity == pytest.approx(expected)
    assert 1e12 < velocity < 1e13  # Å/s, the size the VASP tutorial reports
    rate, phenomenological = fe.rate_constant(0.4, 2.0, velocity, T)
    assert rate == pytest.approx(0.5 * velocity * 2.0 * np.exp(-0.4 / KT))
    # Eyring with ΔA‡ gives back the same rate.
    assert KT / fe.PLANCK * np.exp(-phenomenological / KT) == pytest.approx(rate, rel=1e-9)
    centers, density = fe.probability_density(np.random.default_rng(0).normal(size=5000), 40)
    assert np.trapezoid(density, centers) == pytest.approx(1.0, abs=0.05)
