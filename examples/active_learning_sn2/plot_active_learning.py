"""The active-learning run in one figure: scan error per round, and what was selected.

    python plot_active_learning.py [run folder] [output png]

Reads each round's ``evaluate/scan_vs_reference.csv`` (energy error and
committee spread on the held-out scan frames) and round 0's selection
manifest, and draws the error along r(C-F) for every round with its committee
spread (dashed), the tolerance, and the selected scan frames.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ase.io import read  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           r"D:\MLIP_Work_Folder\sn2_F_CH3Cl\active_learning")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RUN / "active_learning_scan.png"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, one per round


def column(path, name):
    with path.open(newline="", encoding="utf-8") as handle:
        return [float(row[name]) for row in csv.DictReader(handle)]


def main():
    rounds = json.loads((RUN / "rounds.json").read_text())
    tolerance = 1000 * json.loads((RUN / "config.json").read_text())["tolerances"]["scan_max_ev"]
    figure, ax = plt.subplots(figsize=(8.0, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for row, color in zip(rounds, COLORS, strict=False):
        table = RUN / f"round_{row['round']:02d}" / "evaluate" / "scan_vs_reference.csv"
        x = column(table, "scan distance (Å)")
        error = [1000 * e for e in column(table, "energy_error_eV")]
        spread = [1000 * s for s in column(table, "committee_energy_std_eV")]
        ax.plot(x, error, color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1,
                label=f"round {row['round']}: error (max {max(map(abs, error)):.1f} meV)")
        ax.plot(x, spread, color=color, lw=1.2, ls=(0, (4, 2)),
                label=f"round {row['round']}: committee spread")
    ax.axhspan(-tolerance, tolerance, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    selected = json.loads((RUN / "round_00" / "select" / "manifest.json").read_text())
    scan0 = read(RUN / "round_00" / "explore" / "scan.extxyz", ":")
    for frame in selected["notes"]["summary"]["frames"]:
        if frame["source"].startswith("scan"):
            distance = scan0[int(frame["source"].split()[-1])].info["scan_distance"]
            ax.axvline(distance, color=COLORS[0], lw=0.8, ls=":", zorder=0)
    ax.text(0.01, 0.02, f"shaded: tolerance ±{tolerance:.0f} meV; dotted: scan frames selected "
            "and labeled in round 0", transform=ax.transAxes, color=MUTED, fontsize=8.5)
    ax.set_xlabel("r(C–F) scan distance (Å), held-out frames", color=INK2)
    ax.set_ylabel("Model − ωB97X-D, relative (meV)", color=INK2)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2, fontsize=8.5)
    ax.set_title("Active learning, SN2 with an AIMNet2 committee: off-path error by round",
                 loc="left", color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(OUT, dpi=110, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
