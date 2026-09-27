"""Step 2: train every model of the learning curves.  -> models/<name>/

For each training-set size N, the first N structures of one fixed shuffle of
the 87 PBE-labeled structures (so the sets are nested), three models:

- ``direct_N{N}``: MACE-MP-0 small fine-tuned on PBE (plain, as in
  docs/fine_tuning.md);
- ``delta-gfn1_N{N}`` / ``delta-gfn2_N{N}``: a small MACE trained from scratch
  on PBE − GFN1-xTB / PBE − GFN2-xTB. Its model card names the baseline, so
  the tool always evaluates it as xTB + correction.

Every model gets the same number of optimizer steps (``STEPS``), whatever N.
Models already trained are kept. Usage: ``train.py [SEED ...]`` (default 1).
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from ase.io import read, write
from common import BASELINES, BATCH, FOUNDATION, POOL, VALID_FRACTION, WORK, epochs, xtb

from samson_mlip_visualizer.delta import delta_e0s, delta_labels, xtb_baseline_card
from samson_mlip_visualizer.training import TrainingSpec, install_models, train_local
from samson_mlip_visualizer.xtb_backend import find_xtb

SIZES = (10, 20, 40, 87)
SEEDS = tuple(int(s) for s in sys.argv[1:]) or (1,)
WORKERS = 6  # trainings at once: 3-atom batches leave the GPU mostly idle

pool = read(POOL, ":")
order = np.random.default_rng(0).permutation(len(pool))
data = WORK / "data"
data.mkdir(parents=True, exist_ok=True)
labels = {}
for method in BASELINES:
    cached = data / f"pool_delta_{method}.extxyz"
    if not cached.exists():
        write(cached, delta_labels(pool, xtb(method)))
    labels[method] = read(cached, ":")


common = dict(seeds=SEEDS, batch_size=BATCH, valid_fraction=VALID_FRACTION, device="cuda")
card = {"reference": "PBE/def2-TZVP (Psi4)", "elements": "H, C, N only",
        "scope": "HCN <-> HNC learning-curve model; not a general-purpose model"}
specs = []
for n in SIZES:
    subset = [int(i) for i in order[:n]]
    direct_file = data / f"direct_N{n}.extxyz"
    write(direct_file, [pool[i] for i in subset])
    specs.append(TrainingSpec(
        name=f"direct_N{n}", foundation=FOUNDATION, train_file=str(direct_file),
        epochs=epochs(n), card={**card, "trained_on": f"{n} structures"}, **common))
    for method in BASELINES:
        frames = [labels[method][i] for i in subset]
        delta_file = data / f"delta-{method}_N{n}.extxyz"
        write(delta_file, frames)
        specs.append(TrainingSpec(
            name=f"delta-{method}_N{n}", foundation="", train_file=str(delta_file),
            mode="scratch", e0s=delta_e0s(frames), energy_key="DELTA_energy",
            forces_key="DELTA_forces", lr=0.01, epochs=epochs(n),
            card={**card, "trained_on": f"{n} structures",
                  "target": f"PBE/def2-TZVP − {method.upper()}-xTB",
                  "delta_baseline": xtb_baseline_card(method, executable=find_xtb())},
            **common))


def run(spec):
    target = WORK / "models" / spec.name
    if all((target / f"{spec.model_name(seed)}.model").exists() for seed in spec.seeds):
        return spec.name, "kept", 0.0
    start = time.perf_counter()
    runs = train_local(spec, WORK / "runs" / spec.name)
    failed = [run.seed for run in runs if not run.ok]
    install_models(WORK / "runs" / spec.name, destination=WORK / "models")
    status = f"FAILED seeds {failed} (see runs/{spec.name})" if failed else "ok"
    return spec.name, status, time.perf_counter() - start


start = time.perf_counter()
with ThreadPoolExecutor(WORKERS) as executor:
    for name, status, seconds in executor.map(run, specs):
        print(f"{name:<18} {status} {seconds:6.0f} s", flush=True)
print(f"all trainings: {time.perf_counter() - start:.0f} s wall")
Path(WORK / "models" / "README.txt").write_text(
    "Learning-curve models of examples/delta_xtb_hcn (train.py). delta-* models are "
    "corrections: load them through samson_mlip_visualizer.calculators.create_calculator, "
    "which adds their GFN-xTB baseline.\n")
