"""Free energies along a reaction coordinate: blue moon, slow growth, metadynamics.

The methods of the VASP molecular-dynamics tutorial on the SN2 reaction
(https://vasp.at/tutorials/latest/md/part3/), for any ASE calculator:

- :class:`DistanceCombination`: a reaction coordinate ξ = Σ c_k d(a_k, b_k),
  e.g. ξ = d(C–Cl) − d(C–F), with its analytic gradient and the curvature term
  the blue-moon estimator needs.
- :func:`constrained_md`: velocity Verlet with the holonomic constraint
  ξ(R) = ξ_target enforced by SHAKE (positions) and RATTLE (velocities), an
  Andersen thermostat, and per-step records of the Lagrange multiplier λ,
  Z = Σ_i |∇_i ξ|²/m_i, and G. With a fixed target this is a *blue-moon*
  window; with a target that moves by a fixed increment per step it is a
  *slow-growth* run.
- :func:`blue_moon_gradient`: dA/dξ = ⟨Z^-½ (λ + k_B T G)⟩ / ⟨Z^-½⟩
  (Sprik & Ciccotti, J. Chem. Phys. 109, 7737 (1998)), with λ the multiplier of
  the constraint force **+λ ∇ξ** (so at rest λ = ∂U/∂ξ, the mean force's sign).
- :class:`MetadynamicsCalculator`: well-tempered metadynamics (Barducci,
  Bussi, Parrinello, PRL 100, 020603 (2008)) on ξ, as a bias added to any
  calculator, with optional harmonic walls; :func:`metadynamics` runs it with
  Langevin dynamics and :func:`fes_from_hills` rebuilds the free energy.
- Rate-theory helpers: :func:`probability_density`,
  :func:`generalized_velocity` (⟨|ξ̇*|⟩ = sqrt(2 k_B T / π) / ⟨Z^-½⟩), and
  :func:`rate_constant` (k = ⟨|ξ̇*|⟩/2 · P(ξ_ref) · exp(−ΔA/k_B T), and the
  phenomenological ΔA‡ that includes the prefactor).

Units: Å, eV, amu, fs (ASE's internal time is used inside); free energies in eV.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

KB = units.kB  # eV/K
PLANCK = 4.135667696e-15  # eV s


# --- the reaction coordinate --------------------------------------------------------


@dataclass
class DistanceCombination:
    """ξ(R) = Σ_k c_k |R_{a_k} − R_{b_k}|, e.g. ``[(0, 4, 1.0), (0, 5, -1.0)]``."""

    terms: Sequence[tuple[int, int, float]]

    def value(self, positions: np.ndarray) -> float:
        return float(sum(c * np.linalg.norm(positions[a] - positions[b])
                         for a, b, c in self.terms))

    def gradient(self, positions: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(positions, dtype=float)
        for a, b, c in self.terms:
            vector = positions[a] - positions[b]
            unit = vector / np.linalg.norm(vector)
            grad[a] += c * unit
            grad[b] -= c * unit
        return grad

    def z(self, positions: np.ndarray, masses: np.ndarray) -> float:
        """Z = Σ_i |∇_i ξ|² / m_i (inverse mass metric, amu⁻¹)."""
        grad = self.gradient(positions)
        return float(((grad**2).sum(axis=1) / masses).sum())

    def g(self, positions: np.ndarray, masses: np.ndarray) -> float:
        """G = Z⁻² Σ_ij m_i⁻¹ m_j⁻¹ ∇_i ξ · ∇_i∇_j ξ · ∇_j ξ (Å⁻¹).

        For one distance term, ∇∇d = c/d (1 − u uᵀ) in the blocks
        [[P, −P], [−P, P]], so the double sum reduces per term to
        c/d · wᵀ (1 − u uᵀ) w with w = ∇_a ξ/m_a − ∇_b ξ/m_b."""
        grad = self.gradient(positions)
        scaled = grad / masses[:, None]
        total = 0.0
        for a, b, c in self.terms:
            vector = positions[a] - positions[b]
            distance = np.linalg.norm(vector)
            unit = vector / distance
            w = scaled[a] - scaled[b]
            total += c / distance * (w @ w - (w @ unit) ** 2)
        return total / self.z(positions, masses) ** 2


# --- constrained molecular dynamics (blue moon, slow growth) ------------------------


@dataclass
class ConstrainedRecord:
    """Per-step records of a constrained run (arrays of one entry per step)."""

    target: np.ndarray  # ξ target, Å
    value: np.ndarray  # ξ reached (equal to the target within tolerance), Å
    lam: np.ndarray  # Lagrange multiplier of the constraint force +λ∇ξ, eV/Å
    z: np.ndarray  # amu⁻¹
    g: np.ndarray  # Å⁻¹
    temperature: np.ndarray  # instantaneous, K (constrained degrees of freedom removed)
    energy: np.ndarray  # potential energy, eV
    frames: list = field(default_factory=list)  # positions, every ``record_every`` steps

    def as_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items() if k != "frames"}


def _forces(atoms: Atoms) -> tuple[np.ndarray, float]:
    return atoms.get_forces(), float(atoms.get_potential_energy())


def constrained_md(
    atoms: Atoms,
    coordinate: DistanceCombination,
    *,
    steps: int,
    temperature_k: float = 300.0,
    timestep_fs: float = 1.0,
    target: float | None = None,
    increment: float = 0.0,
    andersen_probability: float = 0.05,
    masses: Sequence[float] | None = None,
    seed: int = 0,
    record_every: int = 0,
    tolerance: float = 1e-10,
    on_step: Callable[[int, ConstrainedRecord], None] | None = None,
) -> ConstrainedRecord:
    """Velocity Verlet with ξ(R) held at ``target`` (default: its current value)
    by SHAKE/RATTLE, moved by ``increment`` Å per step (slow growth when ≠ 0),
    with an Andersen thermostat (each atom's velocity redrawn with
    ``andersen_probability`` per step). ``masses`` overrides the atoms' masses
    (e.g. tritium, 3.0, to allow a longer step). The atoms keep their calculator;
    positions and velocities are updated in place."""
    rng = np.random.default_rng(seed)
    m = np.asarray(masses if masses is not None else atoms.get_masses(), float)
    kt = KB * temperature_k
    dt = timestep_fs * units.fs
    rate = increment / dt  # dξ/dt in Å per ASE time unit
    positions = atoms.get_positions().copy()
    xi_target = coordinate.value(positions) if target is None else float(target)
    if atoms.get_velocities() is None or not np.any(atoms.get_velocities()):
        velocities = rng.normal(size=positions.shape) * np.sqrt(kt / m)[:, None]
    else:
        velocities = atoms.get_velocities().copy()

    def project(v, x, want):
        grad = coordinate.gradient(x)
        weight = ((grad**2).sum(axis=1) / m).sum()
        mu = (want - (grad * v).sum()) / weight
        return v + mu * grad / m[:, None], mu

    # Put the starting point on the constraint surface: move along ∇ξ until ξ = target.
    for _ in range(100):
        miss = coordinate.value(positions) - xi_target
        if abs(miss) < tolerance:
            break
        grad = coordinate.gradient(positions)
        positions -= miss * (grad / m[:, None]) / ((grad**2).sum(axis=1) / m).sum()
    atoms.set_positions(positions)
    velocities, _ = project(velocities, positions, rate)
    forces, energy = _forces(atoms)
    records = {k: [] for k in ("target", "value", "lam", "z", "g", "temperature", "energy")}
    frames = []
    dof = 3 * len(atoms) - 1  # one constrained degree of freedom
    for step in range(steps):
        # Andersen collisions, then back onto the constraint's velocity manifold.
        hit = rng.random(len(atoms)) < andersen_probability
        if hit.any():
            velocities[hit] = rng.normal(size=(hit.sum(), 3)) * np.sqrt(kt / m[hit])[:, None]
            velocities, _ = project(velocities, positions, rate)
        grad_old = coordinate.gradient(positions)
        half = velocities + 0.5 * dt * forces / m[:, None]
        trial = positions + dt * half
        xi_target += increment
        # SHAKE: r = trial + (dt²/2) λ ∇ξ(r_old)/m with ξ(r) = target (Newton on λ).
        lam = 0.0
        step_dir = 0.5 * dt * dt * grad_old / m[:, None]
        new = trial
        for _ in range(100):
            new = trial + lam * step_dir
            miss = coordinate.value(new) - xi_target
            if abs(miss) < tolerance:
                break
            lam -= miss / (coordinate.gradient(new) * step_dir).sum()
        else:
            raise RuntimeError(f"SHAKE did not converge at step {step} (miss {miss:.2e} Å)")
        half += 0.5 * dt * lam * grad_old / m[:, None]
        positions = new
        atoms.set_positions(positions)
        forces, energy = _forces(atoms)
        velocities = half + 0.5 * dt * forces / m[:, None]
        velocities, mu = project(velocities, positions, rate)
        # Constraint force over the step: SHAKE's λ at t and RATTLE's at t + dt,
        # each applied for dt/2 (μ carries the dt/2 factor inside ``project``).
        lam_step = 0.5 * (lam + mu / (0.5 * dt))
        kinetic = 0.5 * (m[:, None] * velocities**2).sum()
        records["target"].append(xi_target)
        records["value"].append(coordinate.value(positions))
        records["lam"].append(lam_step)
        records["z"].append(coordinate.z(positions, m))
        records["g"].append(coordinate.g(positions, m))
        records["temperature"].append(2 * kinetic / (dof * KB))
        records["energy"].append(energy)
        if record_every and step % record_every == 0:
            frames.append(positions.copy())
        if on_step is not None:
            on_step(step, records)
    atoms.set_velocities(velocities)
    result = ConstrainedRecord(**{k: np.array(v) for k, v in records.items()})
    result.frames = frames
    return result


def blue_moon_gradient(record: ConstrainedRecord, temperature_k: float,
                       skip: int = 0) -> tuple[float, float]:
    """dA/dξ (eV/Å) at a fixed-ξ window and its standard error (blocked, 10 blocks):
    ⟨Z^-½ (λ + k_B T G)⟩ / ⟨Z^-½⟩."""
    kt = KB * temperature_k
    weight = record.z[skip:] ** -0.5
    numerator = weight * (record.lam[skip:] + kt * record.g[skip:])
    value = float(numerator.mean() / weight.mean())
    blocks = [n.mean() / w.mean() for n, w in zip(np.array_split(numerator, 10),
                                                   np.array_split(weight, 10), strict=True)]
    return value, float(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))


def slow_growth_profile(record: ConstrainedRecord) -> tuple[np.ndarray, np.ndarray]:
    """ξ and ΔA(ξ) (eV) from a slow-growth run: ∫ λ dξ along the run, as VASP's
    tutorial does (the Z/G corrections average out only in a converged window)."""
    xi = record.target
    work = np.concatenate([[0.0], np.cumsum(0.5 * (record.lam[1:] + record.lam[:-1])
                                            * np.diff(xi))])
    return xi, work


def integrate_gradient(xi: Sequence[float], gradient: Sequence[float]) -> np.ndarray:
    """Cumulative trapezoid of dA/dξ (thermodynamic integration), A(ξ₀) = 0."""
    xi, gradient = np.asarray(xi, float), np.asarray(gradient, float)
    return np.concatenate([[0.0], np.cumsum(0.5 * (gradient[1:] + gradient[:-1]) * np.diff(xi))])


# --- metadynamics ---------------------------------------------------------------------


class MetadynamicsCalculator(Calculator):
    """``base`` plus a well-tempered metadynamics bias on ξ and optional walls.

    Hills of height w·exp(−V(ξ)/(k_B ΔT)) and width ``sigma`` are added by
    :meth:`deposit`; ``bias_factor`` γ = (T + ΔT)/T. ``walls``: harmonic
    restraints ``(kind, index_or_terms, limit, k)`` with kind "xi_upper",
    "xi_lower", or "distance_upper" (``(a, b)``), energy ½k(x − limit)² beyond
    the limit. The base calculator's energy and forces are reported with the
    bias added; ``results["bias"]`` holds the bias energy."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, base: Calculator, coordinate: DistanceCombination, *, height: float,
                 sigma: float, temperature_k: float, bias_factor: float = 10.0,
                 walls: Sequence[tuple] = (), **kwargs):
        super().__init__(**kwargs)
        self.base, self.coordinate = base, coordinate
        self.height, self.sigma = float(height), float(sigma)
        self.delta_t = temperature_k * (bias_factor - 1.0)
        self.temperature_k, self.walls = temperature_k, list(walls)
        self.centers: list[float] = []
        self.heights: list[float] = []

    def bias(self, xi: float | np.ndarray) -> np.ndarray:
        xi = np.atleast_1d(np.asarray(xi, float))
        if not self.centers:
            return np.zeros_like(xi)
        c, h = np.array(self.centers), np.array(self.heights)
        return (h * np.exp(-0.5 * ((xi[:, None] - c) / self.sigma) ** 2)).sum(axis=1)

    def bias_derivative(self, xi: float) -> float:
        if not self.centers:
            return 0.0
        c, h = np.array(self.centers), np.array(self.heights)
        gauss = h * np.exp(-0.5 * ((xi - c) / self.sigma) ** 2)
        return float((-gauss * (xi - c) / self.sigma**2).sum())

    def deposit(self, atoms: Atoms) -> None:
        xi = self.coordinate.value(atoms.get_positions())
        height = self.height * np.exp(-self.bias(xi)[0] / (KB * self.delta_t))
        self.centers.append(xi)
        self.heights.append(height)

    def _walls(self, positions):
        energy, forces = 0.0, np.zeros_like(positions)
        for kind, what, limit, k in self.walls:
            if kind in ("xi_upper", "xi_lower"):
                x = self.coordinate.value(positions)
                excess = x - limit if kind == "xi_upper" else limit - x
                if excess > 0:
                    energy += 0.5 * k * excess**2
                    sign = 1.0 if kind == "xi_upper" else -1.0
                    forces -= sign * k * excess * self.coordinate.gradient(positions)
            elif kind == "distance_upper":
                a, b = what
                vector = positions[a] - positions[b]
                d = np.linalg.norm(vector)
                if d > limit:
                    energy += 0.5 * k * (d - limit) ** 2
                    push = k * (d - limit) * vector / d
                    forces[a] -= push
                    forces[b] += push
            else:
                raise ValueError(f"Unknown wall kind {kind!r}")
        return energy, forces

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        probe = self.atoms.copy()
        probe.calc = self.base
        energy, forces = float(probe.get_potential_energy()), probe.get_forces().copy()
        positions = self.atoms.get_positions()
        xi = self.coordinate.value(positions)
        bias = float(self.bias(xi)[0])
        forces -= self.bias_derivative(xi) * self.coordinate.gradient(positions)
        wall_energy, wall_forces = self._walls(positions)
        self.results = {"energy": energy + bias + wall_energy,
                        "free_energy": energy + bias + wall_energy,
                        "forces": forces + wall_forces, "bias": bias,
                        "unbiased_energy": energy}


def metadynamics(atoms: Atoms, bias: MetadynamicsCalculator, *, steps: int, pace: int = 50,
                 timestep_fs: float = 1.0, friction_per_fs: float = 0.01, seed: int = 0,
                 record_every: int = 10, on_progress: Callable[[int, float], None] | None = None,
                 ) -> dict:
    """Langevin dynamics on ``bias`` (set as the atoms' calculator), a hill every
    ``pace`` steps. Returns ξ every ``record_every`` steps, the hills, and frames."""
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms.calc = bias
    if atoms.get_velocities() is None or not np.any(atoms.get_velocities()):
        MaxwellBoltzmannDistribution(atoms, temperature_K=bias.temperature_k,
                                     rng=np.random.default_rng(seed))
    dynamics = Langevin(atoms, timestep_fs * units.fs, temperature_K=bias.temperature_k,
                        friction=friction_per_fs / units.fs, rng=np.random.default_rng(seed))
    trace, frames = [], []
    for step in range(steps):
        dynamics.run(1)
        if step % pace == 0:
            bias.deposit(atoms)
        if step % record_every == 0:
            xi = bias.coordinate.value(atoms.get_positions())
            trace.append(xi)
            frames.append(atoms.get_positions().copy())
            if on_progress:
                on_progress(step, xi)
    return {"xi": np.array(trace), "centers": np.array(bias.centers),
            "heights": np.array(bias.heights), "frames": frames}


def fes_from_hills(grid: Sequence[float], centers: Sequence[float], heights: Sequence[float],
                   sigma: float, bias_factor: float) -> np.ndarray:
    """Well-tempered estimate A(ξ) = −γ/(γ − 1) · V(ξ), shifted to a minimum of 0."""
    grid = np.asarray(grid, float)
    c, h = np.asarray(centers, float), np.asarray(heights, float)
    bias = (h * np.exp(-0.5 * ((grid[:, None] - c) / sigma) ** 2)).sum(axis=1)
    fes = -bias_factor / (bias_factor - 1.0) * bias
    return fes - fes.min()


# --- rate theory ----------------------------------------------------------------------


def probability_density(values: Sequence[float], bins: int | Sequence[float] = 100):
    """Normalized histogram P(ξ) (Å⁻¹) and bin centers."""
    density, edges = np.histogram(np.asarray(values, float), bins=bins, density=True)
    return 0.5 * (edges[1:] + edges[:-1]), density


def generalized_velocity(z: Sequence[float], temperature_k: float) -> float:
    """⟨|ξ̇*|⟩ = sqrt(2 k_B T / π) / ⟨Z^-½⟩ from a constrained run at ξ*, in Å/s."""
    kt = KB * temperature_k
    mean = float(np.mean(np.asarray(z, float) ** -0.5))
    # sqrt(eV/amu) is Å per ASE time unit; units.s is the number of those per second.
    return float(np.sqrt(2 * kt / np.pi) / mean * units.s)


def rate_constant(barrier_ev: float, density_at_reference: float, velocity: float,
                  temperature_k: float) -> tuple[float, float]:
    """k = ⟨|ξ̇*|⟩/2 · P(ξ_ref) · exp(−ΔA/k_B T) (s⁻¹, P in Å⁻¹, ξ̇ in Å/s) and the
    phenomenological ΔA‡ = ΔA − k_B T ln(h/(k_B T) · ⟨|ξ̇*|⟩/2 · P(ξ_ref)) (eV)."""
    kt = KB * temperature_k
    prefactor = 0.5 * velocity * density_at_reference
    rate = prefactor * np.exp(-barrier_ev / kt)
    phenomenological = barrier_ev - kt * np.log(PLANCK / kt * prefactor)
    return float(rate), float(phenomenological)
