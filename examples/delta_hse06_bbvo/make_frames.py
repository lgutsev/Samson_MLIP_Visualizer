"""Step 1 (laptop): frames to label, from MACE-MP-0.  -> frames.extxyz

Groups (``info["group"]``):

- ``strain``: the 10-atom primitive cell, isotropically scaled by 0.96–1.04 and
  with three shears, each also rattled (σ 0.03 Å) — how energy and stress change
  with the lattice;
- ``md300`` / ``md900``: the 40-atom cubic cell in MACE-MP-0 Langevin NVT MD,
  at 300 and 900 K (training);
- ``md600``: an independent 600 K trajectory (held out);
- with ``--doped``: ``nb`` / ``ta``, the 40-atom cell with one V replaced by Nb
  or Ta, at 600 K (the same B-site chemistry the correction will need later).

MD at the PBE+U lattice (MACE-MP-0 relaxes it 0.17 % larger), 1 fs steps,
400 steps of equilibration, one frame every 50 fs.
"""

import sys
import time

import numpy as np
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from common import WORK, conventional, grouped, mace_mp0, primitive

DOPED = "--doped" in sys.argv
WORK.mkdir(parents=True, exist_ok=True)
calc = mace_mp0()
rng = np.random.default_rng(11)


def plain(atoms, group):
    out = atoms.copy()
    out.calc = None
    out.info = {"group": group}
    return out


frames = []
for scale in (0.96, 0.98, 1.00, 1.02, 1.04):
    frames.append(plain(primitive(scale), "strain"))
for strain in ([[0.02, 0.0, 0.0], [0.0, -0.01, 0.0], [0.0, 0.0, -0.01]],
               [[0.0, 0.015, 0.0], [0.015, 0.0, 0.0], [0.0, 0.0, 0.0]],
               [[0.0, 0.0, 0.02], [0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]):
    atoms = primitive()
    atoms.set_cell(atoms.cell @ (np.eye(3) + np.array(strain)), scale_atoms=True)
    frames.append(plain(atoms, "strain"))
frames[2].info["pbe_u_geometry"] = True  # scale 1.00, unrattled: the earlier runs' structure
frames += [plain(a, "strain") for a in [f.copy() for f in frames]]
for atoms in frames[len(frames) // 2:]:
    atoms.positions += rng.normal(0, 0.03, atoms.positions.shape)


def md(start, temperature, count, seed, group):
    atoms = start.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,
                                 rng=np.random.default_rng(seed))
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temperature, friction=0.02,
                   rng=np.random.default_rng(seed + 1))
    dyn.run(400)
    out = []
    for _ in range(count):
        dyn.run(50)
        out.append(plain(atoms, group))
    return out


t0 = time.perf_counter()
frames += md(conventional(), 300, 12, seed=1, group="md300")
frames += md(conventional(), 900, 12, seed=2, group="md900")
frames += md(conventional(), 600, 6, seed=3, group="md600")
if DOPED:
    for dopant, seed in (("Nb", 4), ("Ta", 5)):
        cell = conventional()
        cell[[a.index for a in cell if a.symbol == "V"][0]].symbol = dopant
        frames += md(grouped(cell), 600, 6, seed=seed, group=dopant.lower())
write(WORK / "frames.extxyz", frames)
counts = {}
for frame in frames:
    counts[frame.info["group"]] = counts.get(frame.info["group"], 0) + 1
print(f"{len(frames)} frames {counts} in {time.perf_counter() - t0:.0f} s -> "
      f"{WORK / 'frames.extxyz'}")
