"""Step 1: geometries from GFN2-xTB, labels from PBE0/def2-TZVP.
-> pool.extxyz (training candidates), test_set.extxyz (held out)

Geometries (``info["group"]``):

- ``scan``: one CO pulled off, Ni–C from 1.70 to 5.0 Å, everything else relaxed
  with GFN2-xTB at each distance. Every other point, plus two rattled copies
  (σ 0.03 Å) of each, goes to the pool; the points between go to the test set;
- ``md400`` / ``md900``: GFN2-xTB Langevin MD at 400 and 900 K (pool);
- ``md650``: an independent trajectory at 650 K (test set only).

Every frame records its Ni–C distance of the leaving CO (``r_nic``). Frames
whose PBE0 calculation fails are left out and counted.
"""

import time

import numpy as np
from ase import Atoms, units
from ase.constraints import FixBondLength
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from common import C1, NI, POOL, TEST_SET, WORK, ni_co4, pbe0, xtb

WORK.mkdir(parents=True, exist_ok=True)
gfn2 = xtb("gfn2")
rng = np.random.default_rng(3)


def plain(atoms):
    return Atoms(atoms.get_chemical_symbols(), positions=atoms.positions)


# --- geometries ------------------------------------------------------------------
start = ni_co4()
start.calc = gfn2
BFGS(start, logfile=None).run(fmax=0.005, steps=300)
equilibrium = start.copy()

scan = []
for r in (1.70, 1.80, 1.90, 2.00, 2.15, 2.30, 2.50, 2.70, 2.95, 3.20, 3.50, 3.85, 4.25, 4.60,
          5.00):
    atoms = equilibrium.copy()
    # move the leaving CO rigidly along its axis, then relax with the Ni–C bond fixed
    axis = atoms.positions[C1] - atoms.positions[NI]
    axis /= np.linalg.norm(axis)
    shift = (r - atoms.get_distance(NI, C1)) * axis
    atoms.positions[[C1, C1 + 1]] += shift
    atoms.set_constraint(FixBondLength(NI, C1))
    atoms.calc = gfn2
    BFGS(atoms, logfile=None).run(fmax=0.01, steps=300)
    atoms.set_constraint()
    scan.append(plain(atoms))


def md(temperature, frames, seed, equilibrate=200, every=20):
    atoms = equilibrium.copy()
    atoms.calc = gfn2
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(seed))
    dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=temperature, friction=0.02,
                   rng=np.random.default_rng(seed + 1))
    dyn.run(equilibrate)
    out = []
    for _ in range(frames):
        dyn.run(every)
        out.append(plain(atoms))
    return out


t0 = time.perf_counter()
pool_frames = [("scan", a) for a in scan[::2]]
pool_frames += [("scan_rattled", Atoms(a.get_chemical_symbols(),
                                       positions=a.positions + rng.normal(0, 0.03, (9, 3))))
                for a in scan[::2] for _ in range(2)]
pool_frames += [("md400", a) for a in md(400, 25, seed=10)]
pool_frames += [("md900", a) for a in md(900, 25, seed=20)]
test_frames = [("scan", a) for a in scan[1::2]] + [("md650", a) for a in md(650, 15, seed=30)]
print(f"geometries: {len(pool_frames)} pool + {len(test_frames)} test "
      f"({time.perf_counter() - t0:.0f} s of GFN2-xTB)", flush=True)

# --- PBE0 labels ------------------------------------------------------------------
reference = pbe0()


def label(frames, path):
    labeled, failed, t = [], 0, time.perf_counter()
    for k, (group, atoms) in enumerate(frames):
        atoms = atoms.copy()
        atoms.calc = reference
        try:
            energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        except Exception as exc:  # an SCF that does not converge: leave the frame out
            print(f"  {group} frame {k}: PBE0 failed ({str(exc)[:120]})", flush=True)
            failed += 1
            continue
        out = plain(atoms)
        out.info.update(group=group, r_nic=float(atoms.get_distance(NI, C1)),
                        REF_energy=float(energy))
        out.arrays["REF_forces"] = np.asarray(forces)
        labeled.append(out)
        if k % 10 == 9:
            print(f"  {path.name}: {k + 1}/{len(frames)} ({time.perf_counter() - t:.0f} s)",
                  flush=True)
    write(path, labeled)
    print(f"{path.name}: {len(labeled)} labeled, {failed} failed, "
          f"{time.perf_counter() - t:.0f} s", flush=True)


label(test_frames, TEST_SET)
label(pool_frames, POOL)
