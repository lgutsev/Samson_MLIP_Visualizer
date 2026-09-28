"""Step 2: every model of the learning curves.  -> models/<name>/

For each training-set size N (the first N of one fixed shuffle of the pool, so
the sets are nested):

- ``direct_N{N}``: MACE-MP-0 small fine-tuned on PBE0 (plain). Psi4's
  all-electron energies are moved onto the foundation scale by one constant, the
  mean E_PBE0 − E_MACE-MP-0 over the pool (one composition: relative energies
  are unchanged);
- ``delta-gfn2_N{N}`` / ``delta-gfn1_N{N}``: a small MACE from scratch on
  PBE0 − GFN2-xTB / PBE0 − GFN1-xTB;
- ``delta-mace_N{N}``: the same on PBE0 − MACE-MP-0 (an MLIP baseline).

Every model gets the same number of optimizer steps. Models already trained are
kept. Usage: ``train.py [SEED ...]`` (default 1).
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase.io import read, write
from common import (
    BASELINES,
    BATCH,
    FOUNDATION,
    POOL,
    VALID_FRACTION,
    WORK,
    baseline,
    epochs,
    mace_mp0,
)

from samson_mlip_visualizer.delta import (
    delta_e0s,
    delta_labels,
    mace_baseline_card,
    xtb_baseline_card,
)
from samson_mlip_visualizer.training import TrainingSpec, install_models, train_local
from samson_mlip_visualizer.xtb_backend import find_xtb

SIZES = (10, 20, 40, 74)
SEEDS = tuple(int(s) for s in sys.argv[1:]) or (1,)
WORKERS = 6

pool = read(POOL, ":")
order = np.random.default_rng(0).permutation(len(pool))
data = WORK / "data"
data.mkdir(parents=True, exist_ok=True)

# the constant between Psi4's scale and the foundation's (direct fine-tuning only)
shift_file = data / "shift.json"
if not shift_file.exists():
    calc = mace_mp0()
    differences = []
    for frame in pool:
        atoms = frame.copy()
        atoms.calc = calc
        differences.append(frame.info["REF_energy"] - atoms.get_potential_energy())
    shift_file.write_text(json.dumps({"shift_ev": float(np.mean(differences)),
                                      "spread_ev": float(np.std(differences))}))
shift = json.loads(shift_file.read_text())["shift_ev"]
labels = {}
for method in BASELINES:
    cached = data / f"pool_delta_{method}.extxyz"
    if not cached.exists():
        write(cached, delta_labels(pool, baseline(method)))
    labels[method] = read(cached, ":")


def baseline_card(method):
    if method == "mace":
        return mace_baseline_card(FOUNDATION)
    return xtb_baseline_card(method, executable=find_xtb())


common = dict(seeds=SEEDS, batch_size=BATCH, valid_fraction=VALID_FRACTION, device="cuda")
card = {"reference": "PBE0/def2-TZVP (Psi4)", "elements": "Ni, C, O only",
        "scope": "Ni(CO)4 -> Ni(CO)3 + CO learning-curve model; not a general-purpose model"}
specs = []
for n in SIZES:
    subset = [int(i) for i in order[:n]]
    direct = []
    for i in subset:
        frame = pool[i].copy()
        frame.info["REF_energy"] = pool[i].info["REF_energy"] - shift
        direct.append(frame)
    direct_file = data / f"direct_N{n}.extxyz"
    write(direct_file, direct)
    specs.append(TrainingSpec(
        name=f"direct_N{n}", foundation=FOUNDATION, train_file=str(direct_file),
        epochs=epochs(n), card={**card, "trained_on": f"{n} structures",
                                "energy_shift_ev": shift}, **common))
    for method in BASELINES:
        frames = [labels[method][i] for i in subset]
        delta_file = data / f"delta-{method}_N{n}.extxyz"
        write(delta_file, frames)
        specs.append(TrainingSpec(
            name=f"delta-{method}_N{n}", foundation="", train_file=str(delta_file),
            mode="scratch", e0s=delta_e0s(frames), energy_key="DELTA_energy",
            forces_key="DELTA_forces", lr=0.01, epochs=epochs(n),
            card={**card, "trained_on": f"{n} structures",
                  "target": f"PBE0/def2-TZVP − {method}",
                  "delta_baseline": baseline_card(method)},
            **common))


def run(spec):
    target = WORK / "models" / spec.name
    if all((target / f"{spec.model_name(seed)}.model").exists() for seed in spec.seeds):
        return spec.name, "kept", 0.0
    start = time.perf_counter()
    runs = train_local(spec, WORK / "runs" / spec.name)
    failed = [run.seed for run in runs if not run.ok]
    install_models(WORK / "runs" / spec.name, destination=WORK / "models")
    return spec.name, f"FAILED seeds {failed}" if failed else "ok", time.perf_counter() - start


start = time.perf_counter()
with ThreadPoolExecutor(WORKERS) as executor:
    for name, status, seconds in executor.map(run, specs):
        print(f"{name:<18} {status} {seconds:6.0f} s", flush=True)
print(f"all trainings: {time.perf_counter() - start:.0f} s wall")
