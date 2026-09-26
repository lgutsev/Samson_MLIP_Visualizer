"""Learning curve of the AIMNet2 fine-tune (validation errors against epoch).

    python plot_aimnet2_training.py [work folder] [output png]

Reads ``finetune_aimnet2/log.json`` (from ``finetune_aimnet2.py``). Two panels,
one quantity each: validation energy RMSE (relative energies) and force RMSE,
with the untuned model at epoch 0 and the kept (best) epoch marked.
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
TUNED = "#4a3aa7"  # fine-tuned AIMNet2: categorical slot 7, as in the energy diagrams


def main():
    log = json.loads((WORK / "finetune_aimnet2" / "log.json").read_text())
    before = log["before"]["valid"]
    epochs = [0] + [row["epoch"] for row in log["epochs"]]
    series = {
        "energy_rmse_ev": ("Validation energy RMSE (meV)", before["energy_rmse_ev"]),
        "force_rmse_ev_per_A": ("Validation force RMSE (meV/Å)", before["force_rmse_ev_per_A"]),
    }
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True, facecolor=SURFACE)
    for ax, (key, (label, start)) in zip(axes, series.items(), strict=True):
        values = [1000 * start] + [1000 * row[key] for row in log["epochs"]]
        ax.set_facecolor(SURFACE)
        ax.plot(epochs, values, color=TUNED, lw=2)
        ax.plot([0], [values[0]], "o", color=TUNED, ms=8, mec=SURFACE, mew=2)
        best = epochs.index(log["best_epoch"])
        ax.plot([epochs[best]], [values[best]], "o", color=TUNED, ms=8, mec=SURFACE, mew=2)
        ax.annotate(f"untuned {values[0]:.0f}", (0, values[0]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
        ax.annotate(f"kept, epoch {epochs[best]}: {values[best]:.1f}", (epochs[best], values[best]),
                    xytext=(0, 10), textcoords="offset points", ha="center", color=INK2,
                    fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel(label, color=INK2)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
    axes[-1].set_xlabel("Epoch (full batch)", color=INK2)
    axes[0].set_title(f"Fine-tuning AIMNet2 on {log['train']} ωB97X-D labels "
                      f"({log['valid']} held out for validation)",
                      loc="left", color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(OUT, dpi=110, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
