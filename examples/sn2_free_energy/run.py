"""One task of the SN2 free-energy example; ``launch.py`` runs them in parallel.

    python run.py SYSTEM TASK [ARG]

SYSTEM is ``F`` or ``Cl`` (see ``common.py``). Tasks, each writing to
``<work>/<SYSTEM>/`` and skipped when its output exists:

- ``slow_growth forward|reverse``: 500 steps at the start of the ξ range, then
  ξ moved across it at 1e-3 Å per 2-fs step (the tutorial's INCREM is 8e-4).
- ``window XI``: blue-moon run at fixed ξ = XI (3000 steps, the first 500 dropped).
- ``free_md``: unconstrained MD (Andersen) from the reactant complex, 10 000 steps,
  for P(ξ); a harmonic wall keeps the nucleophile within 5 Å of carbon.
- ``ts_velocity XI``: constrained run at the free-energy maximum ξ* for ⟨|ξ̇*|⟩.
- ``metadynamics``: well-tempered, 25 000 steps (50 ps) with walls at ξ = −1.6 and
  +0.5 Å: the reactant well and the barrier. (With the upper wall at +1.0 Å the walker
  crossed once, after 12 ps, and spent the rest filling the 1 eV deep product side.)
"""

import sys

import numpy as np
from ase.io import write
from common import (
    ANDERSEN,
    COORDINATE,
    SYSTEMS,
    TEMPERATURE,
    TIMESTEP_FS,
    calculator,
    folder,
    prepare,
    reactant_complex,
    start_near,
)

from samson_mlip_visualizer import free_energy as fe

META_UPPER = 0.5  # just past ξ* ≈ 0.08, so the walker keeps recrossing the barrier


def npz(path):
    """``path`` + ".npz" (not ``with_suffix``: "xi_+0.750" would become "xi_+0.npz")."""
    return path.parent / (path.name + ".npz")


def save(path, record, symbols):
    np.savez(npz(path), target=record.target, value=record.value,
             lam=record.lam, z=record.z, g=record.g, temperature=record.temperature,
             energy=record.energy)
    from ase import Atoms

    frames = [Atoms(symbols, positions=p) for p in record.frames]
    if frames:
        write(path.parent / (path.name + ".extxyz"), frames)


def slow_growth(system, direction):
    out = folder(system) / f"slow_growth_{direction}"
    if npz(out).exists():
        return
    low, high = SYSTEMS[system]["xi_range"]
    start, end = (low, high) if direction == "forward" else (high, low)
    if system == "Cl" and direction == "reverse":
        # The mirror image of the forward start: swap the chlorines of the reactant
        # complex (ξ changes sign), then pull the far one out to the start value.
        atoms = reactant_complex(system)
        atoms.positions[[4, 5]] = atoms.positions[[5, 4]]
        axis = atoms.positions[4] - atoms.positions[0]
        atoms.positions[4] += (start - COORDINATE.value(atoms.positions)) * axis / np.linalg.norm(
            axis)
    else:
        atoms = start_near(system, start)
    prepare(atoms, system)
    masses = SYSTEMS[system]["masses"]
    fe.constrained_md(atoms, COORDINATE, steps=500, target=start, temperature_k=TEMPERATURE,
                      timestep_fs=TIMESTEP_FS, andersen_probability=ANDERSEN, masses=masses)
    steps = int(round(abs(end - start) / 1e-3))
    record = fe.constrained_md(atoms, COORDINATE, steps=steps, target=start,
                               increment=(end - start) / steps, temperature_k=TEMPERATURE,
                               timestep_fs=TIMESTEP_FS, andersen_probability=ANDERSEN,
                               masses=masses, seed=1, record_every=25)
    save(out, record, SYSTEMS[system]["symbols"])


def window(system, xi):
    out = folder(system) / "windows" / f"xi_{xi:+.3f}"
    out.parent.mkdir(exist_ok=True)
    if npz(out).exists():
        return
    atoms = prepare(start_near(system, xi), system)
    record = fe.constrained_md(atoms, COORDINATE, steps=3000, target=xi,
                               temperature_k=TEMPERATURE, timestep_fs=TIMESTEP_FS,
                               andersen_probability=ANDERSEN, masses=SYSTEMS[system]["masses"],
                               seed=int(abs(xi) * 1000) + 3, record_every=100)
    save(out, record, SYSTEMS[system]["symbols"])


def ts_velocity(system, xi):
    out = folder(system) / "ts_velocity"
    if npz(out).exists():
        return
    atoms = prepare(start_near(system, xi), system)
    record = fe.constrained_md(atoms, COORDINATE, steps=3000, target=xi,
                               temperature_k=TEMPERATURE, timestep_fs=TIMESTEP_FS,
                               andersen_probability=ANDERSEN, masses=SYSTEMS[system]["masses"],
                               seed=11, record_every=100)
    save(out, record, SYSTEMS[system]["symbols"])


def free_md(system):
    out = folder(system) / "free_md.npz"
    if out.exists():
        return
    from ase import units
    from ase.md.andersen import Andersen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = reactant_complex(system)
    atoms.set_masses(SYSTEMS[system]["masses"])
    # A wall on the nucleophile's distance only: F⁻ (or Cl⁻) cannot fly off in vacuum.
    atoms.calc = fe.MetadynamicsCalculator(calculator(system), COORDINATE, height=0.0,
                                           sigma=0.1, temperature_k=TEMPERATURE,
                                           walls=[("distance_upper", (0, 5), 5.0, 5.0)])
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE, rng=np.random.default_rng(5))
    dynamics = Andersen(atoms, TIMESTEP_FS * units.fs, temperature_K=TEMPERATURE,
                        andersen_prob=ANDERSEN, rng=np.random.default_rng(5))
    xi, frames = [], []
    for step in range(10000):
        dynamics.run(1)
        xi.append(COORDINATE.value(atoms.positions))
        if step % 25 == 0:
            frames.append(atoms.copy())
    np.savez(out, xi=np.array(xi))
    write(out.with_suffix(".extxyz"), frames)


def metadynamics(system):
    out = folder(system) / "metadynamics.npz"
    if out.exists():
        return
    atoms = prepare(start_near(system, -0.6), system)
    bias = fe.MetadynamicsCalculator(atoms.calc, COORDINATE, height=0.02, sigma=0.08,
                                     temperature_k=TEMPERATURE, bias_factor=10.0,
                                     walls=[("xi_lower", None, -1.6, 20.0),
                                            ("xi_upper", None, META_UPPER, 20.0)])
    result = fe.metadynamics(atoms, bias, steps=25000, pace=50, timestep_fs=TIMESTEP_FS,
                             friction_per_fs=0.01, seed=7, record_every=10)
    np.savez(out, xi=result["xi"], centers=result["centers"], heights=result["heights"],
             sigma=0.08, bias_factor=10.0, upper=META_UPPER)
    from ase import Atoms

    write(out.with_suffix(".extxyz"), [Atoms(SYSTEMS[system]["symbols"], positions=p)
                                       for p in result["frames"][::5]])


if __name__ == "__main__":
    system, task, *rest = sys.argv[1:]
    {"slow_growth": lambda: slow_growth(system, rest[0]),
     "window": lambda: window(system, float(rest[0])),
     "ts_velocity": lambda: ts_velocity(system, float(rest[0])),
     "free_md": lambda: free_md(system),
     "metadynamics": lambda: metadynamics(system)}[task]()
