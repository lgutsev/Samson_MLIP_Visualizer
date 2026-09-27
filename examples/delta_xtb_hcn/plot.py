"""Step 4: learning curves from results.json.  -> learning_curves.png

Four panels against the number of training structures: energy and force
errors near the reaction path, the energy error on the bond stretches (all
three groups together), and the barrier error. Seeds are averaged; the unchanged
MACE-MP-0 and xTB methods are drawn as horizontal reference lines, and the
models of the stretch round (87 path structures + 8 bond stretches,
``stretch_round.py``) as hollow markers at 95.
"""

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import PBE_BARRIER, WORK
from matplotlib.ticker import FormatStrFormatter, LogLocator

results = json.loads((WORK / "results.json").read_text())
SERIES = {"direct": ("MACE-MP-0 fine-tuned on PBE", "#2a78d6", "o"),
          "delta-gfn1": ("GFN1-xTB + Δ-MACE", "#eb6834", "s"),
          "delta-gfn2": ("GFN2-xTB + Δ-MACE", "#1baf7a", "D")}
REFERENCES = {"MACE-MP-0 small": ":", "GFN1-xTB": "--", "GFN2-xTB": "-."}
STRETCHES = ("ch_stretch", "nh_stretch", "cn_stretch")


def metrics(entry):
    test = entry["test"]
    frames = sum(test[g]["frames"] for g in STRETCHES)
    return {
        "near_e": test["near"]["energy_mae_ev"],
        "near_f": test["near"]["force_rmse_ev_A"],
        "stretch_e": sum(test[g]["energy_mae_ev"] * test[g]["frames"] for g in STRETCHES) / frames,
        "barrier": abs(entry["barrier_ev"] - PBE_BARRIER) if "barrier_ev" in entry else np.nan,
    }


curves = defaultdict(lambda: defaultdict(list))
for name, entry in results.items():
    if "_N" in name:
        method, rest = name.split("_N")
        curves[method][int(rest.split("_")[0])].append(metrics(entry))

PANELS = [("near_e", "Energy MAE near the path (eV)"),
          ("near_f", "Force RMSE near the path (eV/Å)"),
          ("stretch_e", "Energy MAE on bond stretches (eV)"),
          ("barrier", "|Barrier − PBE| (eV)")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), constrained_layout=True)
for ax, (key, title) in zip(axes.flat, PANELS, strict=True):
    for method, (label, color, marker) in SERIES.items():
        sizes = sorted(curves[method])
        values = [np.nanmean([m[key] for m in curves[method][n]]) for n in sizes]
        ax.plot(sizes, values, color=color, marker=marker, markersize=6, linewidth=2,
                label=label)
        for n, entries in curves.get(f"{method}+stretch", {}).items():
            ax.plot([n], [np.nanmean([m[key] for m in entries])], marker=marker, markersize=8,
                    markerfacecolor="none", markeredgewidth=2, color=color, linestyle="none",
                    label=f"{label}, + 8 stretch labels")
    for name, style in REFERENCES.items():
        if name in results:
            ax.axhline(metrics(results[name])[key], color="#8a8985", linestyle=style,
                       linewidth=1.2, label=f"{name} (unchanged)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 20, 40, 87], labels=["10", "20", "40", "87"])  # hollow markers: 95
    ax.minorticks_off()
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_title(title, loc="left", color="#0b0b0b")
    ax.set_xlabel("PBE-labeled training structures", color="#52514e")
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.tick_params(colors="#52514e")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
fig.suptitle("HCN ⇌ HNC: xTB + Δ-learning vs direct fine-tuning (held-out PBE/def2-TZVP)",
             x=0.01, ha="left", fontsize=11)
out = WORK / "learning_curves.png"
fig.savefig(out, dpi=150)
print(out)
