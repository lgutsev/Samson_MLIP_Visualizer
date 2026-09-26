"""Step 1, seed data: the MACE-MP-0 IRC, 29 frames of it labeled with PBE, and two
rattled copies (σ 0.04 Å) of each, also labeled with PBE.  -> train_r0.extxyz"""

import json
import time

import numpy as np
from ase import Atoms
from ase.io import write
from common import FOUNDATION, HERE, label, mace, pbe, record, ts_guess

from samson_mlip_visualizer.benchmark import select_frames
from samson_mlip_visualizer.reaction_path import irc
from samson_mlip_visualizer.ts import prfo_search

start = time.perf_counter()
model = mace(FOUNDATION)
ts = ts_guess()
ts.calc = model
prfo_search(ts, fmax=1e-4, exact_hessian=True)
path = irc(ts, step=0.05, fmax=0.005, relax_ends=True)
frames = path.frames(ts.get_positions())
positions = [
    path.reverse_minimum_positions,
    *(f.positions for f in frames),
    path.forward_minimum_positions,
]
arcs = [frames[0].arc - 0.05, *(f.arc for f in frames), frames[-1].arc + 0.05]
picked = select_frames(len(positions), 28, keep=(len(path.reverse) + 1,), coordinate=arcs)
seed = [Atoms("CNH", positions=positions[k]) for k in picked]
record("seed_mace_irc_s", time.perf_counter() - start)

start = time.perf_counter()
reference = pbe()
first = seed[0].copy()
first.calc = model
e_mace = first.get_potential_energy()
first.calc = reference
(HERE / "shift.json").write_text(json.dumps({"shift_ev": first.get_potential_energy() - e_mace}))
data = label(seed, reference, "irc0")
rng = np.random.default_rng(0)
rattled = [
    Atoms("CNH", positions=a.positions + rng.normal(0, 0.04, (3, 3)))
    for a in seed
    for _ in range(2)
]
data += label(rattled, reference, "rattle0")
write(HERE / "train_r0.extxyz", data)
record("seed_labeling_s", time.perf_counter() - start)
print(f"{len(data)} configurations ({len(seed)} IRC frames + {len(rattled)} rattled)")
