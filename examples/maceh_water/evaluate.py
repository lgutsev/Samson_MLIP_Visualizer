"""Predict Kohn-Sham matrices with the trained MACE-H and compare them to Psi4.

    PYTHONPATH=../../src python evaluate.py [--device cpu] [--model DIR] [--tag NAME] [--only SETS]

For each set that exists (held-out dimer MD run, held-out trimer runs,
tetramers, and the training sets for reference) this runs MACE-H's
``deephe3-eval.py`` and then, per structure, diagonalizes the predicted H against
the Psi4 overlap. Tetramers are predicted in MACE-H's inference mode: the graph
comes from ``overlaps.h5`` only, as it would for a new structure with no SCF.
Reports:

- matrix-element MAE (meV) over all blocks, and over on-site/off-site blocks;
- orbital-energy errors of the occupied valence orbitals (the O 1s levels, near
  -510 eV, are left out), HOMO, LUMO, and gap (meV).

Writes ``results_<tag>.json`` and ``images/maceh_water_<tag>.png`` (no suffix
without ``--tag``); predictions go to ``eval/<tag>/<set>``.
"""

import json
from pathlib import Path

import numpy as np
from common import EVAL_DIR, GRAPHS, HERE, PROCESSED, WORK, latest_model, run_maceh, write_ini

from samson_mlip_visualizer.hamiltonian import (
    assemble,
    orbital_energies,
    read_blocks,
    read_orbital_types,
)

# name: inference mode (graph from overlaps.h5 only, as for a structure with no SCF)
SETS = {"dimer_test": False, "trimer_test": False, "tetramer_test": True,
        "hexamer_ring_test": True, "hexamer_prism_test": True,
        "dimer_train": False, "trimer_train": False}
COLORS = {"dimer_test": "#1f77b4", "trimer_test": "#d62728", "tetramer_test": "#2ca02c",
          "hexamer_ring_test": "#9467bd", "hexamer_prism_test": "#ff7f0e"}


def predict(name, inference, model, device, out):
    config = write_ini(out / "eval.ini", {
        "basic": {"device": device, "dtype": "float", "trained_model_dir": model.as_posix(),
                  "output_dir": out.as_posix(), "target": "hamiltonian",
                  "inference": inference, "test_only": False},
        "data": {"graph_dir": "", "DFT_data_dir": "",
                 "processed_data_dir": PROCESSED[name].as_posix(),
                 "save_graph_dir": (GRAPHS / f"{name}_eval").as_posix(),
                 "target_data": "hamiltonian", "dataset_name": f"{name}_eval",
                 "get_overlap": False, "radius": -1},
    })
    run_maceh("deephe3-eval.py", config, out / "eval.log")


def compare(folder: Path, predicted: Path):
    types = read_orbital_types(folder)
    info = json.loads((folder / "info.json").read_text())
    reference = read_blocks(folder / "hamiltonians.h5")
    guess = read_blocks(predicted)
    diffs = {"all": [], "onsite": [], "offsite": []}
    for key, block in reference.items():
        d = np.abs(guess[key] - block).ravel()
        diffs["all"].append(d)
        diffs["onsite" if key[3] == key[4] else "offsite"].append(d)
    overlap = assemble(read_blocks(folder / "overlaps.h5"), types)
    eps_ref = orbital_energies(assemble(reference, types), overlap)
    eps_pred = orbital_energies(assemble(guess, types), overlap)
    n = info["n_occupied"]
    core = int((np.loadtxt(folder / "element.dat") == 8).sum())  # one 1s per O
    # The Psi4 eigenvalues check the DeepH round trip (reorder + eV + blocks).
    roundtrip = np.abs(eps_ref - np.asarray(info["eigenvalues_eV"])).max()
    gap_ref, gap_pred = eps_ref[n] - eps_ref[n - 1], eps_pred[n] - eps_pred[n - 1]
    return {
        "blocks": {k: np.concatenate(v) for k, v in diffs.items()},
        "valence_error": eps_pred[core:n] - eps_ref[core:n],
        "core_error": eps_pred[:core] - eps_ref[:core],
        "homo": (eps_ref[n - 1], eps_pred[n - 1]),
        "lumo": (eps_ref[n], eps_pred[n]),
        "gap": (gap_ref, gap_pred),
        "valence_ref": eps_ref[core:n], "valence_pred": eps_pred[core:n],
        "roundtrip": roundtrip,
    }


def summarize(rows):
    mae = lambda x: float(np.mean(np.abs(x)) * 1000)  # noqa: E731 - eV to meV
    pairs = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    homo, lumo, gap = pairs("homo"), pairs("lumo"), pairs("gap")
    return {
        "structures": len(rows),
        "H_MAE_meV": {k: mae(np.concatenate([r["blocks"][k] for r in rows]))
                      for k in ("all", "onsite", "offsite")},
        "valence_MAE_meV": mae(np.concatenate([r["valence_error"] for r in rows])),
        "O1s_MAE_meV": mae(np.concatenate([r["core_error"] for r in rows])),
        "HOMO_MAE_meV": mae(homo[:, 1] - homo[:, 0]),
        "LUMO_MAE_meV": mae(lumo[:, 1] - lumo[:, 0]),
        "gap_MAE_meV": mae(gap[:, 1] - gap[:, 0]),
        "gap_range_eV": [float(gap[:, 0].min()), float(gap[:, 0].max())],
        "max_roundtrip_error_eV": float(max(r["roundtrip"] for r in rows)),
    }


def plot(all_rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    colors = COLORS
    for name, rows in all_rows.items():
        if name not in colors:
            continue
        ref = np.concatenate([r["valence_ref"] for r in rows])
        pred = np.concatenate([r["valence_pred"] for r in rows])
        axes[0].scatter(ref, pred, s=6, alpha=0.5, color=colors[name], label=name)
        gap = np.array([r["gap"] for r in rows])
        axes[1].scatter(gap[:, 0], gap[:, 1], s=12, alpha=0.7, color=colors[name], label=name)
    for ax, title in zip(axes, ("Valence orbital energies (eV)", "HOMO-LUMO gap (eV)"),
                         strict=True):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
        ax.set_xlabel("Psi4 PBE/def2-SVP")
        ax.set_ylabel("MACE-H H, diagonalized with Psi4 S")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", type=Path, help="a MACE-H run folder (default: the latest)")
    parser.add_argument("--only", default="",
                        help="comma-separated sets; merged into an existing results file")
    parser.add_argument("--tag", default="", help="suffix for results/plot/eval folders")
    args = parser.parse_args()
    model = args.model or latest_model()
    suffix = f"_{args.tag}" if args.tag else ""
    print(f"Model: {model}")
    only = [name for name in args.only.split(",") if name]
    results_file = WORK / f"results{suffix}.json"
    results, all_rows = {"model": model.name}, {}
    if only and results_file.is_file():
        results = json.loads(results_file.read_text())
        if results.get("model") != model.name:
            raise SystemExit(f"{results_file} is for {results['model']}, not {model.name}")
    for name, inference in SETS.items():
        if only and name not in only:
            continue
        if not PROCESSED[name].is_dir():
            continue
        out = EVAL_DIR / (args.tag or "latest") / name
        predict(name, inference, model, args.device, out)
        rows = [compare(folder, out / folder.name / "hamiltonians_pred.h5")
                for folder in sorted(PROCESSED[name].iterdir()) if folder.is_dir()]
        all_rows[name] = rows
        results[name] = summarize(rows)
        print(name, json.dumps(results[name], indent=1))
    results_file.write_text(json.dumps(results, indent=1))
    (HERE / f"results{suffix}.json").write_text(json.dumps(results, indent=1))
    image = f"maceh_water{suffix}" + ("_" + "_".join(only) if only else "")
    plot(all_rows, HERE / "images" / f"{image}.png")


if __name__ == "__main__":
    main()
