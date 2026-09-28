"""Step 2: every model of the learning curves.  -> models/<name>/

For each training-set size N (the first N of one fixed shuffle of the pool, so
the sets are nested):

- ``direct_N{N}``: MACE-MP-0 small fine-tuned on PBE0 (plain). Psi4's
  all-electron energies are moved onto the foundation scale by one constant, the
  mean E_PBE0 − E_MACE-MP-0 over the pool (one composition: relative energies
  are unchanged);
- ``delta-gfn2_N{N}`` / ``delta-gfn1_N{N}``: a small MACE from scratch on
  PBE0 − GFN2-xTB / PBE0 − GFN1-xTB (with ``--rmax 7`` also
  ``delta-gfn*_rmax7_N{N}``, the same with a 7 Å cutoff);
- ``delta-mace_N{N}``: the same on PBE0 − MACE-MP-0 (an MLIP baseline).

Every model gets the same number of optimizer steps. Models already trained are
kept.

Usage::

    train.py [--seeds 1 2 3] [--rmax 7]            # train here (the laptop GPU)
    train.py --package DIR [--seeds ...] [--rmax 7] [--smoke]

``--package`` writes the same models as HPC training packages instead (one SLURM
array per model, one task per seed; ``submit_all.sh`` submits them all). The
residual labels are computed here, so the cluster needs only mace-torch. Copy
the ``runs/`` folders back and install with ``install_models``; evaluate here,
where xTB is. ``--smoke`` packages two models on 74 structures for 3 epochs,
one seed: a check of the cluster before the full set.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
from samson_mlip_visualizer.training import (
    GpuSlurmSettings,
    TrainingSpec,
    install_models,
    train_local,
    write_training_package,
)
from samson_mlip_visualizer.xtb_backend import find_xtb

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--seeds", type=int, nargs="+", default=[1])
parser.add_argument("--rmax", type=float, nargs="*", default=[],
                    help="extra cutoffs (Å) for the xTB corrections, besides the default 5")
parser.add_argument("--package", type=Path, help="write HPC training packages here instead")
parser.add_argument("--smoke", action="store_true", help="with --package: 2 models, 3 epochs")
args = parser.parse_args()
SIZES = (10, 20, 40, 74)
SEEDS = tuple(args.seeds)
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
        cutoffs = [None] + ([r for r in args.rmax] if method != "mace" else [])
        for r_max in cutoffs:
            tag = f"_rmax{r_max:g}" if r_max else ""
            specs.append(TrainingSpec(
                name=f"delta-{method}{tag}_N{n}", foundation="", train_file=str(delta_file),
                mode="scratch", e0s=delta_e0s(frames), energy_key="DELTA_energy",
                forces_key="DELTA_forces", lr=0.01, epochs=epochs(n),
                architecture={"r_max": r_max} if r_max else {},
                card={**card, "trained_on": f"{n} structures",
                      "target": f"PBE0/def2-TZVP − {method}",
                      "delta_baseline": baseline_card(method)},
                **common))

if args.package:
    if args.smoke:
        keep = {"direct_N74", f"delta-gfn2_rmax{args.rmax[0]:g}_N74" if args.rmax
                else "delta-gfn2_N74"}
        specs = [TrainingSpec.from_json({**s.to_json(), "epochs": 3, "seeds": (1,)})
                 for s in specs if s.name in keep]
    root = args.package
    root.mkdir(parents=True, exist_ok=True)
    slurm = GpuSlurmSettings(account="loni_perovsk27", partition="gpu2", modules=(),  # QB4
                             activate=("source /home/lgutsev/miniforge3/etc/profile.d/conda.sh && "
                                       "conda activate /project/lgutsev/env/mace_env"),
                             time="02:00:00" if args.smoke else "06:00:00", memory_gb=16)
    for spec in specs:
        write_training_package(spec, root / spec.name, slurm=slurm)
    names = [spec.name for spec in specs]
    (root / "submit_all.sh").write_text("#!/bin/bash\n# Submits every package here.\nset -e\n"
                                        + "".join(f"(cd {name} && sbatch run_train.slurm)\n"
                                                  for name in names), newline="\n")
    (root / "README.md").write_text(
        f"# Ni(CO)4 learning-curve training: {len(specs)} models, seeds {list(SEEDS)}\n\n"
        "Written by `examples/delta_nico4/train.py --package`. Each folder is a training\n"
        "package (see its README); the residual labels are already in its data, so the\n"
        "cluster needs only mace-torch 0.3.16. Fill in the placeholders in every\n"
        "`run_train.slurm` (the same four in each), then `bash submit_all.sh`.\n\n"
        "Afterwards copy every `<model>/runs/` back into the same folders and, on the\n"
        "desktop, install and evaluate:\n\n"
        "```python\n"
        "from samson_mlip_visualizer.training import install_models\n"
        "for folder in Path(r'<this folder>').iterdir():\n"
        "    if (folder / 'spec.json').exists():\n"
        "        install_models(folder, destination=Path(r'<DELTA_DIR>') / 'models')\n"
        "```\n\n"
        "then `python evaluate.py` in `examples/delta_nico4` (it needs xtb for the\n"
        "xTB corrections).\n\n" + "".join(f"- `{name}`\n" for name in names),
        encoding="utf-8")
    print(f"{len(specs)} training packages -> {root}")
    raise SystemExit


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
        print(f"{name:<22} {status} {seconds:6.0f} s", flush=True)
print(f"all trainings: {time.perf_counter() - start:.0f} s wall")
