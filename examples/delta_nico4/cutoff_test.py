"""Step 4: is the xTB corrections' remaining dissociation error a cutoff problem?

On 74 structures, GFN-xTB + Δ still overestimates the CO pull energy by
0.1–0.2 eV, while MACE-MP-0 + Δ and direct fine-tuning get it within 0.03 eV.
GFN-xTB's overbinding is long-ranged (its curve still rises at Ni–C 4–5 Å), and
the correction sees 5 Å. This retrains both xTB corrections on the same 74
structures with a 7 Å cutoff and evaluates them like the rest.
-> models/delta-gfn{1,2}_rmax7_N74/, results_cutoff.json
"""

import json

from ase.io import read
from common import BATCH, VALID_FRACTION, WORK, epochs
from evaluate import evaluate

from samson_mlip_visualizer.calculators import create_calculator
from samson_mlip_visualizer.delta import delta_e0s, xtb_baseline_card
from samson_mlip_visualizer.training import TrainingSpec, install_models, train_local
from samson_mlip_visualizer.xtb_backend import find_xtb

N = 74
results = {}
for method in ("gfn2", "gfn1"):
    train_file = WORK / "data" / f"delta-{method}_N{N}.extxyz"
    frames = read(train_file, ":")
    spec = TrainingSpec(
        name=f"delta-{method}_rmax7_N{N}", foundation="", train_file=str(train_file),
        mode="scratch", e0s=delta_e0s(frames), architecture={"r_max": 7.0},
        energy_key="DELTA_energy", forces_key="DELTA_forces", lr=0.01, seeds=(1,),
        batch_size=BATCH, valid_fraction=VALID_FRACTION, device="cuda", epochs=epochs(N),
        card={"reference": "PBE0/def2-TZVP (Psi4)", "scope": "Ni(CO)4 cutoff test",
              "delta_baseline": xtb_baseline_card(method, executable=find_xtb())})
    target = WORK / "models" / spec.name / f"{spec.model_name(1)}.model"
    if not target.exists():
        train_local(spec, WORK / "runs" / spec.name)
        install_models(WORK / "runs" / spec.name, destination=WORK / "models")
    entry = evaluate(create_calculator("mace", target, device="cuda"))
    results[spec.name] = entry
    print(f"{spec.name}: scan E {entry['scan']['energy_mae_ev']:.3f} "
          f"md650 E {entry['md650']['energy_mae_ev']:.3f} F {entry['md650']['force_rmse_ev_A']:.3f}"
          f" | pull {entry['pull_ev']:.3f}", flush=True)
(WORK / "results_cutoff.json").write_text(json.dumps(results, indent=1))
