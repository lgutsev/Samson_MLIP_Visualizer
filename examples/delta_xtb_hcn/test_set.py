"""Step 1: a held-out test set labeled with PBE/def2-TZVP.  -> test_set.extxyz

Groups (``info["group"]``), from close to the training data to far from it:

- ``anchor``: the HCN minimum of the training IRC; every energy is taken
  relative to it, so constant offsets between methods drop out;
- ``near``: the 29 training IRC frames, rattled with σ 0.08 Å (the training
  copies used 0.04 Å and another seed);
- ``ch_stretch``: HCN with C–H from 1.0 to 2.0 Å (bond breaking);
- ``nh_stretch``: HNC with N–H from 0.9 to 1.9 Å;
- ``cn_stretch``: HCN with C≡N from 1.0 to 1.5 Å.

None of the stretches is in the training data, which only follows the
isomerization path. Each frame records its aligned RMSD to the nearest
training structure (``rmsd_to_pool``).
"""

import time

import numpy as np
from ase import Atoms
from ase.io import read, write
from common import POOL, TEST_SET, WORK, hcn, hnc, pbe

from samson_mlip_visualizer.finetune import distances_to

pool = read(POOL, ":")
irc = [a for a in pool if a.info["tag"] == "irc0"]
rng = np.random.default_rng(1)
groups = [("anchor", [irc[0].copy()])]
groups.append(("near", [Atoms("CNH", positions=a.positions + rng.normal(0, 0.08, (3, 3)))
                        for a in irc]))
groups.append(("ch_stretch", [hcn(r_ch=r) for r in np.linspace(1.0, 2.0, 9)]))
groups.append(("nh_stretch", [hnc(r_nh=r) for r in np.linspace(0.9, 1.9, 9)]))
groups.append(("cn_stretch", [hcn(r_cn=r) for r in np.linspace(1.0, 1.5, 6)]))

WORK.mkdir(parents=True, exist_ok=True)
reference = pbe()
start = time.perf_counter()
test, failed = [], 0
for group, frames in groups:
    distances = distances_to(frames, pool)
    for frame, distance in zip(frames, distances, strict=True):
        atoms = Atoms(frame.get_chemical_symbols(), positions=frame.positions)
        atoms.calc = reference
        try:
            energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        except Exception as exc:  # an SCF that does not converge: leave the frame out
            print(f"{group}: PBE failed ({exc})")
            failed += 1
            continue
        atoms.calc = None
        atoms.info.update(group=group, PBE_energy=float(energy), rmsd_to_pool=float(distance))
        atoms.arrays["PBE_forces"] = np.asarray(forces)
        test.append(atoms)
write(TEST_SET, test)
print(f"{len(test)} frames ({failed} failed) in {time.perf_counter() - start:.0f} s -> {TEST_SET}")
for group, _ in groups:
    d = [a.info["rmsd_to_pool"] for a in test if a.info["group"] == group]
    print(f"  {group:<11} {len(d):2d} frames, RMSD to training {min(d):.3f}-{max(d):.3f} Å")
