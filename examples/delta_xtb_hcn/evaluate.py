"""Step 3: every model on the held-out test set, plus its own barrier.
-> results.json, learning_curves.png

For each model (and, for comparison, MACE-MP-0, GFN1-xTB, and GFN2-xTB unchanged):

- energies relative to the anchor frame (HCN minimum) against PBE's, per test
  group: mean and largest absolute error;
- force RMSE per group, all components;
- the barrier and reaction energy on the model's own surface: P-RFO from the
  MACE-MP-0 TS (exact Hessian), HCN and HNC relaxed from linear starts.

Δ-models load through ``create_calculator``, which wraps them in their xTB
baseline because their model card says so.
"""

import json
import time

import numpy as np
from ase.io import read
from common import (
    BASELINES,
    FOUNDATION,
    PBE_BARRIER,
    PBE_REACTION,
    TEST_SET,
    WORK,
    hcn,
    hnc,
    ts_guess,
    xtb,
)

from samson_mlip_visualizer.calculators import create_calculator
from samson_mlip_visualizer.engine import relax
from samson_mlip_visualizer.ts import prfo_search

GROUPS = ("near", "ch_stretch", "nh_stretch", "cn_stretch")
test = read(TEST_SET, ":")
anchor = next(a for a in test if a.info["group"] == "anchor")


def test_errors(calc):
    def evaluate(frame):
        atoms = frame.copy()
        atoms.calc = calc
        return atoms.get_potential_energy(), atoms.get_forces()

    e_anchor, _ = evaluate(anchor)
    out = {}
    for group in GROUPS:
        frames = [a for a in test if a.info["group"] == group]
        e_err, f_err = [], []
        for frame in frames:
            energy, forces = evaluate(frame)
            reference = frame.info["PBE_energy"] - anchor.info["PBE_energy"]
            e_err.append((energy - e_anchor) - reference)
            f_err.append(forces - frame.arrays["PBE_forces"])
        e_err = np.abs(e_err)
        out[group] = {"frames": len(frames), "energy_mae_ev": float(e_err.mean()),
                      "energy_max_ev": float(e_err.max()),
                      "force_rmse_ev_A": float(np.sqrt(np.mean(np.square(f_err))))}
    return out


def surface(calc):
    """Barrier and reaction energy on the model's own surface."""
    ends = {}
    for name, start in (("hcn", hcn()), ("hnc", hnc())):
        start.calc = calc
        relax(start, fmax=1e-3, max_steps=500, optimizer="BFGS")
        ends[name] = start.get_potential_energy()
    ts = ts_guess()
    ts.calc = calc
    result = prfo_search(ts, fmax=1e-3, exact_hessian=True)
    return {"barrier_ev": float(ts.get_potential_energy() - ends["hcn"]),
            "reaction_ev": float(ends["hnc"] - ends["hcn"]),
            "ts_converged": bool(result.converged),
            "ts_angle_hcn_deg": float(ts.get_angle(2, 0, 1))}


models = {"MACE-MP-0 small": lambda: create_calculator("mace", FOUNDATION, device="cuda"),
          **{f"{m.upper()}-xTB": (lambda m=m: xtb(m)) for m in BASELINES}}
for folder in sorted((WORK / "models").glob("*_N*")):
    for model in sorted(folder.glob("*.model")):
        models[model.stem] = lambda model=model: create_calculator("mace", model, device="cuda")

results_path = WORK / "results.json"
results = json.loads(results_path.read_text()) if results_path.exists() else {}
for name, build in models.items():
    if name in results:
        continue
    start = time.perf_counter()
    calc = build()
    entry = {"test": test_errors(calc)}
    try:
        entry.update(surface(calc))
    except Exception as exc:  # a failed search is a result too
        entry["surface_error"] = f"{type(exc).__name__}: {exc}"
    entry["seconds"] = round(time.perf_counter() - start, 1)
    results[name] = entry
    results_path.write_text(json.dumps(results, indent=1))
    near, ch = entry["test"]["near"], entry["test"]["ch_stretch"]
    print(f"{name:<22} near E {near['energy_mae_ev']:.3f} F {near['force_rmse_ev_A']:.3f} | "
          f"C-H stretch E {ch['energy_mae_ev']:.3f} | "
          f"barrier {entry.get('barrier_ev', np.nan):.3f} "
          f"(PBE {PBE_BARRIER}) reaction {entry.get('reaction_ev', np.nan):.3f} "
          f"(PBE {PBE_REACTION}) | {entry['seconds']:.0f} s", flush=True)
