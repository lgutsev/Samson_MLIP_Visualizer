"""Step 2b: one active-learning round for bond breaking.  -> models/*+stretch_N95/

The models trained on the isomerization path alone extrapolate badly when a
bond breaks: the Δ-correction has never seen a hydrogen that far out, and it
spoils a baseline that was right there. This round labels 8 stretched
geometries with PBE, at bond lengths between and beyond the test points (C–H
1.3/1.6/1.9/2.2 Å in HCN, N–H 1.2/1.5/1.8/2.1 Å in HNC), adds them to the 87
path structures, and retrains the three kinds of model on the 95.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from ase.io import read, write
from common import BASELINES, BATCH, FOUNDATION, POOL, WORK, epochs, hcn, hnc, pbe, xtb

from samson_mlip_visualizer.delta import delta_e0s, delta_labels, xtb_baseline_card
from samson_mlip_visualizer.finetune import labeled_structure
from samson_mlip_visualizer.training import TrainingSpec, install_models, train_local
from samson_mlip_visualizer.xtb_backend import find_xtb

data = WORK / "data"
labels_file = data / "stretch_labels.extxyz"
pool = read(POOL, ":")
if not labels_file.exists():
    # PBE on the same scale as the pool: its REF_energy is PBE minus one constant,
    # recovered from a pool frame recomputed with PBE.
    reference = pbe()
    probe = pool[0].copy()
    probe.calc = reference
    shift = probe.get_potential_energy() - pool[0].info["REF_energy"]
    frames = [hcn(r_ch=r) for r in (1.3, 1.6, 1.9, 2.2)]
    frames += [hnc(r_nh=r) for r in (1.2, 1.5, 1.8, 2.1)]
    start = time.perf_counter()
    labeled = []
    for frame in frames:
        frame.calc = reference
        energy, forces = frame.get_potential_energy(), frame.get_forces()
        labeled.append(labeled_structure(frame, energy - shift, forces, tag="stretch"))
    write(labels_file, labeled)
    print(f"8 PBE labels in {time.perf_counter() - start:.0f} s (shift {shift:.4f} eV)")
extra = read(labels_file, ":")
train = pool + extra
n = len(train)
card = {"reference": "PBE/def2-TZVP (Psi4)", "elements": "H, C, N only",
        "scope": "HCN <-> HNC plus C-H / N-H stretches; not a general-purpose model",
        "trained_on": f"{len(pool)} path structures + {len(extra)} bond stretches"}
common = dict(seeds=(1,), batch_size=BATCH, valid_fraction=0.1, device="cuda",
              epochs=epochs(n))
specs = []
direct_file = data / f"direct+stretch_N{n}.extxyz"
write(direct_file, train)
specs.append(TrainingSpec(name=f"direct+stretch_N{n}", foundation=FOUNDATION,
                          train_file=str(direct_file), card=card, **common))
for method in BASELINES:
    pool_delta = read(data / f"pool_delta_{method}.extxyz", ":")  # written by train.py
    frames = pool_delta + delta_labels(extra, xtb(method))
    delta_file = data / f"delta-{method}+stretch_N{n}.extxyz"
    write(delta_file, frames)
    specs.append(TrainingSpec(
        name=f"delta-{method}+stretch_N{n}", foundation="", train_file=str(delta_file),
        mode="scratch", e0s=delta_e0s(frames), energy_key="DELTA_energy",
        forces_key="DELTA_forces", lr=0.01,
        card={**card, "target": f"PBE/def2-TZVP − {method.upper()}-xTB",
              "delta_baseline": xtb_baseline_card(method, executable=find_xtb())},
        **common))


def run(spec):
    if (WORK / "models" / spec.name / f"{spec.model_name(1)}.model").exists():
        return spec.name, "kept"
    runs = train_local(spec, WORK / "runs" / spec.name)
    install_models(WORK / "runs" / spec.name, destination=WORK / "models")
    return spec.name, "ok" if all(r.ok for r in runs) else "FAILED"


with ThreadPoolExecutor(3) as executor:
    for name, status in executor.map(run, specs):
        print(f"{name:<26} {status}", flush=True)
