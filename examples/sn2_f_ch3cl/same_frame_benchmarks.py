"""Stock MACE-MP-0, the fine-tuned committee, and AIMNet2 on the same frames vs ωB97X-D.

    python same_frame_benchmarks.py

Two paths, 30 frames each (spread by arc length, TS kept): the AIMNet2 IRC
(``aimnet2_irc.extxyz``; these 30 frames are the fine-tune's training frames)
and the committee's own IRC (``finetune/finetuned_irc.extxyz``; not trained on).
ωB97X-D comes from the shared cache (``sn2_common.reference_cache``), so each
frame is computed once for all three models. Writes a report per run to
``same_frame/`` and every summary to ``same_frame/results.json``; finished runs
are skipped when the script is started again.
"""

import json

import numpy as np
from ase.io import read
from sn2_common import (
    KCAL,
    REFERENCE,
    WORK,
    C,
    F,
    aimnet,
    committee,
    reference_cache,
    stock,
    training_set,
)

from samson_mlip_visualizer.benchmark import (
    _COORDINATES,
    benchmark_path,
    select_frames,
    write_report,
)
from samson_mlip_visualizer.finetune import distances_to

OUT = WORK / "same_frame"
PATHS = {
    "aimnet2_irc": (WORK / "aimnet2_irc.extxyz", "the AIMNet2 IRC (training frames)"),
    "finetuned_irc": (WORK / "finetune" / "finetuned_irc.extxyz",
                      "the fine-tuned committee's IRC (not trained on)"),
}
MODELS = {"stock": ("MACE-MP-0 small", stock), "tuned": ("fine-tuned MACE", committee),
          "aimnet2": ("AIMNet2", aimnet)}
LABELS = {"reactant": "F⁻···CH₃Cl", "product": "FCH₃···Cl⁻"}


def main():
    OUT.mkdir(exist_ok=True)
    results_file = OUT / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    reference = reference_cache()
    train = training_set()
    calculators = {}
    for path_key, (path_file, path_name) in PATHS.items():
        frames = read(path_file, ":")
        arcs = [frame.info["irc_arc"] for frame in frames]
        ts = int(np.argmin(np.abs(arcs)))
        # Which end is the reactant complex (F far from C)?
        cf = [np.linalg.norm(f.positions[F] - f.positions[C]) for f in (frames[0], frames[-1])]
        reactant_first = cf[0] > cf[1]
        ends = (LABELS["reactant"], LABELS["product"])
        ends = ends if reactant_first else ends[::-1]
        picked = select_frames(len(frames), 30, keep=(ts,), coordinate=arcs)
        rmsd = distances_to([frames[k] for k in picked], train)
        for model_key, (model_name, factory) in MODELS.items():
            run = f"{path_key}__{model_key}"
            if run in results:
                continue
            if model_key not in calculators:
                calculators[model_key] = factory()
            before = reference.computed
            bench = benchmark_path(
                frames, calculators[model_key], reference, names=(model_name, REFERENCE),
                coordinate=arcs, coordinate_label=_COORDINATES["irc_arc"], points=30,
                keep=(ts,),
                on_progress=lambda done, total, run=run: print(
                    f"{run}: frame {done}/{total}", flush=True),
            )
            write_report(bench, OUT / run, mark=(0.0, "TS"), end_labels=ends,
                         title=f"{model_name} vs {REFERENCE} along {path_name}")
            start = 0 if reactant_first else -1
            barrier = {name: float((bench.relative(name).max() - bench.relative(name)[start])
                                   * KCAL) for name in bench.names}
            summary = bench.summary()
            results[run] = {
                "path": path_key, "model": model_name, "frames": summary["frames"],
                "barrier_model_kcal": barrier[model_name],
                "barrier_reference_kcal": barrier[REFERENCE],
                "energy_error_max_abs_ev": float(np.abs(bench.energy_error).max()),
                "energy_rmse_ev": summary["energy_error_rmse_ev"],
                "force_mae_ev_per_A": summary["force_mae_ev_per_angstrom"],
                "force_rmse_ev_per_A": summary["force_rmse_ev_per_angstrom"],
                "worst_atom_force_error_ev_per_A": summary["force_error_max_ev_per_angstrom"],
                "committee_energy_std_max_ev": summary.get("committee_energy_std_max_ev"),
                "rmsd_to_training_A": {"median": float(np.median(rmsd)),
                                       "max": float(np.max(rmsd))},
                "new_reference_calculations": reference.computed - before,
            }
            results_file.write_text(json.dumps(results, indent=1))
    print("| path | model | barrier (model / ωB97X-D), kcal/mol | max abs dE, eV | "
          "E RMSE, eV | F MAE / RMSE, eV/Å | worst atom, eV/Å | RMSD to training, Å |")
    print("|---|---|---|---|---|---|---|---|")
    for row in results.values():
        print(f"| {row['path']} | {row['model']} | {row['barrier_model_kcal']:.2f} / "
              f"{row['barrier_reference_kcal']:.2f} | {row['energy_error_max_abs_ev']:.3f} | "
              f"{row['energy_rmse_ev']:.3f} | {row['force_mae_ev_per_A']:.3f} / "
              f"{row['force_rmse_ev_per_A']:.3f} | {row['worst_atom_force_error_ev_per_A']:.3f} | "
              f"{row['rmsd_to_training_A']['median']:.3f} (max "
              f"{row['rmsd_to_training_A']['max']:.3f}) |")


if __name__ == "__main__":
    main()
