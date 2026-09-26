"""Learning curves of the AIMNet2 fine-tunes (validation errors against epoch).

    python plot_aimnet2_training.py [work folder] [output png]

Reads ``finetune_aimnet2/log.json`` and, if present, ``log_fragments.json``
(from ``finetune_aimnet2.py [--fragments]``). Two panels, one quantity each:
validation energy RMSE and force RMSE, with the untuned model at epoch 0 and
the kept (best) epoch marked. The two runs have different validation sets (the
fragment run also holds out two pulled-apart complexes), so compare shapes,
not the last digit.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else WORK / "aimnet2_finetune_learning_curve.png"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
# Categorical slots 7 and 8, as in the energy diagrams.
RUNS = [("log.json", "complexes only", "#4a3aa7"),
        ("log_fragments.json", "+ free fragments", "#e34948")]
KEYS = {"energy_rmse_ev": "Validation energy RMSE (meV)",
        "force_rmse_ev_per_A": "Validation force RMSE (meV/Å)"}


def main():
    logs = [(json.loads((WORK / "finetune_aimnet2" / name).read_text()), label, color)
            for name, label, color in RUNS if (WORK / "finetune_aimnet2" / name).exists()]
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True, facecolor=SURFACE)
    for ax, (key, ylabel) in zip(axes, KEYS.items(), strict=True):
        ax.set_facecolor(SURFACE)
        for log, label, color in logs:
            epochs = [0] + [row["epoch"] for row in log["epochs"]]
            values = [1000 * log["before"]["valid"][key]] + [1000 * row[key]
                                                               for row in log["epochs"]]
            best = epochs.index(log["best_epoch"])
            ax.plot(epochs, values, color=color, lw=2,
                    label=f"{label}: {values[0]:.0f} → {values[best]:.1f}")
            ax.plot([0, epochs[best]], [values[0], values[best]], "o", color=color, ms=7,
                    mec=SURFACE, mew=2)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, color=INK2)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        if len(logs) > 1:
            ax.legend(frameon=False, loc="upper right", labelcolor=INK2, fontsize=9)
    axes[-1].set_xlabel("Epoch (full batch)", color=INK2)
    axes[0].set_title("AIMNet2 fine-tunes: validation errors (dots: untuned, kept epoch)",
                      loc="left", color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(OUT, dpi=110, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
