"""Step 3: every model on the held-out frames, the dissociation curve, and plots.
-> results.json, images/nico4_learning_curves.png, images/nico4_dissociation.png

- held-out errors per group (``scan``: 7 points of the CO pull between the
  training points; ``md650``: an independent 650 K trajectory), energies
  relative to each group's mean, so constant offsets between methods drop out;
- the pull energy E(Ni–C 4.60 Å) − E(Ni–C 1.80 Å), both held-out scan points,
  against PBE0;
- the whole scan (training and held-out points) for the dissociation figure.

The figures also show the 7 Å-cutoff xTB corrections of ``cutoff_test.py``
(``results_cutoff.json``) when they exist: hollow markers at the largest
training set, and a third panel of the dissociation figure.

Δ-models load through ``create_calculator``, which adds their baseline.
"""

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from common import FOUNDATION, HERE, POOL, TEST_SET, WORK, xtb
from matplotlib.ticker import FormatStrFormatter, LogLocator

from samson_mlip_visualizer.calculators import create_calculator

test = read(TEST_SET, ":")
pool = read(POOL, ":")
scan = sorted([f for f in pool + test if f.info["group"] == "scan"], key=lambda f: f.info["r_nic"])
GROUPS = ("scan", "md650")


def energies_forces(calc, frames):
    e, f = [], []
    for frame in frames:
        atoms = frame.copy()
        atoms.calc = calc
        e.append(atoms.get_potential_energy())
        f.append(atoms.get_forces())
    return np.array(e), f


def evaluate(calc):
    out = {}
    for group in GROUPS:
        frames = [a for a in test if a.info["group"] == group]
        e, f = energies_forces(calc, frames)
        ref = np.array([a.info["REF_energy"] for a in frames])
        err = (e - e.mean()) - (ref - ref.mean())
        ferr = np.concatenate([fi - a.arrays["REF_forces"] for fi, a in zip(f, frames,
                                                                             strict=True)])
        out[group] = {"energy_mae_ev": float(np.abs(err).mean()),
                      "energy_max_ev": float(np.abs(err).max()),
                      "force_rmse_ev_A": float(np.sqrt(np.mean(ferr**2)))}
    e_scan, _ = energies_forces(calc, scan)
    r = [f.info["r_nic"] for f in scan]
    near, far = r.index(min(r, key=lambda x: abs(x - 1.80))), r.index(min(r, key=lambda x:
                                                                          abs(x - 4.60)))
    out["pull_ev"] = float(e_scan[far] - e_scan[near])
    out["scan_ev"] = (e_scan - e_scan.min()).tolist()
    return out




def main():
    models = {"MACE-MP-0": lambda: create_calculator("mace", FOUNDATION, device="cuda"),
              "GFN2-xTB": lambda: xtb("gfn2"), "GFN1-xTB": lambda: xtb("gfn1")}
    for folder in sorted((WORK / "models").glob("*_N*")):
        for model in sorted(folder.glob("*.model")):
            models[model.stem] = lambda model=model: create_calculator("mace", model, device="cuda")
    results_file = WORK / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    ref_scan = np.array([f.info["REF_energy"] for f in scan])
    r_scan = [f.info["r_nic"] for f in scan]
    held = [f.info["r_nic"] for f in test if f.info["group"] == "scan"]
    near = r_scan.index(min(r_scan, key=lambda x: abs(x - 1.8)))
    far = r_scan.index(min(r_scan, key=lambda x: abs(x - 4.6)))
    results["PBE0"] = {"scan_ev": (ref_scan - ref_scan.min()).tolist(), "r_nic": r_scan,
                       "held_out_r": held, "pull_ev": float(ref_scan[far] - ref_scan[near])}
    for name, build in models.items():
        if name not in results:
            results[name] = evaluate(build())
            results_file.write_text(json.dumps(results, indent=1))
            entry = results[name]
            print(f"{name:<24} scan E {entry['scan']['energy_mae_ev']:.3f} "
                  f"md650 E {entry['md650']['energy_mae_ev']:.3f} "
                  f"F {entry['md650']['force_rmse_ev_A']:.2f} | pull {entry['pull_ev']:.3f} "
                  f"(PBE0 {results['PBE0']['pull_ev']:.3f})", flush=True)

    # --- figures ----------------------------------------------------------------------
    (HERE / "images").mkdir(exist_ok=True)
    SERIES = {"direct": ("MACE-MP-0 fine-tuned on PBE0", "#2a78d6", "o"),
              "delta-gfn2": ("GFN2-xTB + Δ", "#1baf7a", "D"),
              "delta-gfn1": ("GFN1-xTB + Δ", "#eb6834", "s"),
              "delta-mace": ("MACE-MP-0 + Δ", "#4a3aa7", "^")}
    REFERENCES = {"MACE-MP-0": ":", "GFN2-xTB": "-.", "GFN1-xTB": "--"}
    cutoff_file = WORK / "results_cutoff.json"
    cutoff = json.loads(cutoff_file.read_text()) if cutoff_file.exists() else {}
    curves = defaultdict(dict)
    for name, entry in results.items():
        if "_N" in name:
            method, rest = name.split("_N")
            curves[method][int(rest.split("_")[0])] = entry
    PANELS = [(lambda e: e["md650"]["energy_mae_ev"], "Energy MAE, 650 K MD (eV)"),
              (lambda e: e["md650"]["force_rmse_ev_A"], "Force RMSE, 650 K MD (eV/Å)"),
              (lambda e: e["scan"]["energy_mae_ev"], "Energy MAE, CO pull (eV)"),
              (lambda e: abs(e["pull_ev"] - results["PBE0"]["pull_ev"]),
               "|Pull energy − PBE0| (eV)")]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, PANELS, strict=True):
        for method, (label, color, marker) in SERIES.items():
            sizes = sorted(curves[method])
            ax.plot(sizes, [metric(curves[method][n]) for n in sizes], color=color, marker=marker,
                    markersize=6, linewidth=2, label=label)
            for name, entry in cutoff.items():
                if name.startswith(f"{method}_rmax7_N"):
                    n = int(name.rsplit("_N", 1)[1])
                    ax.plot([n * 1.06], [metric(entry)], marker=marker, markersize=8,
                            markerfacecolor="none", markeredgewidth=2, color=color,
                            linestyle="none", label=f"{label}, 7 Å cutoff")
        for name, style in REFERENCES.items():
            ax.axhline(metric(results[name]), color="#8a8985", linestyle=style, linewidth=1.2,
                       label=f"{name} (unchanged)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([10, 20, 40, 74], labels=["10", "20", "40", "74"])
        ax.minorticks_off()
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_title(title, loc="left")
        ax.set_xlabel("PBE0-labeled training structures", color="#52514e")
        ax.grid(True, color="#e4e3df", linewidth=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Ni(CO)₄: Δ-learning on three baselines vs direct fine-tuning "
                 "(held-out PBE0/def2-TZVP)", x=0.01, ha="left", fontsize=11)
    fig.savefig(HERE / "images" / "nico4_learning_curves.png", dpi=150)

    largest = max(n for n in curves["direct"])
    fig, axes = plt.subplots(1, 3 if cutoff else 2, figsize=(16 if cutoff else 11, 4),
                             constrained_layout=True)
    raw = [("GFN2-xTB", "#1baf7a", "-."), ("GFN1-xTB", "#eb6834", "--"),
           ("MACE-MP-0", "#8a8985", ":")]
    trained = [(f"{method}_N{largest}_seed1", SERIES[method][1], "-") for method in SERIES]
    panels = [("The methods as they come", raw),
              (f"Trained on {largest} PBE0 structures (5 Å cutoff)", trained)]
    if cutoff:
        panels.append((f"xTB corrections, {largest} structures: 5 Å vs 7 Å cutoff", [
            (f"direct_N{largest}_seed1", SERIES["direct"][1], "-"),
            *[(f"delta-{m}_N{largest}_seed1", SERIES[f"delta-{m}"][1], ":") for m in ("gfn2",
                                                                                   "gfn1")],
            *[(f"delta-{m}_rmax7_N{largest}", SERIES[f"delta-{m}"][1], "-") for m in ("gfn2",
                                                                                   "gfn1")]]))
    shown = {**results, **cutoff}
    for ax, (title, names) in zip(axes, panels, strict=True):
        for name, color, style in names:
            if name in shown:
                label = SERIES.get(name.split("_N")[0].replace("_rmax7", ""), (name,))[0]
                if "_rmax7" in name:
                    label += ", 7 Å"
                elif ax is axes[-1] and name.startswith("delta"):
                    label += ", 5 Å"
                ax.plot(r_scan, shown[name]["scan_ev"], color=color, linestyle=style,
                        linewidth=2, label=label)
        train = [r for r in r_scan if r not in held]
        pbe0 = dict(zip(r_scan, results["PBE0"]["scan_ev"], strict=True))
        ax.plot(train, [pbe0[r] for r in train], "o", color="#0b0b0b", ms=5,
                label="PBE0 (training)")
        ax.plot(held, [pbe0[r] for r in held], "o", mfc="white", mec="#0b0b0b", mew=1.6, ms=6,
                label="PBE0 (held out)")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Ni–C distance of the leaving CO (Å), rest relaxed with GFN2-xTB")
        ax.grid(True, color="#e4e3df", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Energy relative to the scan minimum (eV)")
    axes[1].sharey(axes[0])
    if cutoff:  # zoom on the plateau, where the cutoff matters
        axes[2].set_xlim(2.6, 5.1)
        axes[2].set_ylim(0.85, 1.4)
    fig.savefig(HERE / "images" / "nico4_dissociation.png", dpi=150)
    print("figures written")


if __name__ == "__main__":
    main()
