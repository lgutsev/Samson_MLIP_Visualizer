"""Energy-level diagram of the stationary points: each method next to the literature.

    python energy_levels.py [work folder] [output png]

Reads ``comparison.json`` (from ``compare.py``). One short level per method at
each stationary point, joined along the reaction by thin lines; the best
literature value (Czakó's CCSD(T)/CBS focal-point analysis) is drawn in ink as
the reference.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else WORK / "sn2_energy_levels.png"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
# Categorical slots 1-4 of the reference palette, in order; the literature in ink.
SERIES = [
    ("Czakó FPA", "CCSD(T)/CBS focal point (literature)", INK),
    ("AIMNet2", "AIMNet2", "#2a78d6"),
    ("wB97X-D/def2-TZVPD", "ωB97X-D/def2-TZVPD", "#eb6834"),
    ("PBE/def2-TZVPD", "PBE/def2-TZVPD", "#1baf7a"),
    ("GFN2-xTB", "GFN2-xTB", "#eda100"),
]
POINTS = [("reactants", "F⁻ + CH₃Cl"), ("reactant_complex", "F⁻···CH₃Cl"),
          ("ts", "Walden TS"), ("product_complex", "FCH₃···Cl⁻"), ("products", "CH₃F + Cl⁻")]


def main():
    data = json.loads((WORK / "comparison.json").read_text())
    figure, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width, slot = 0.13, 0.165
    for n, (key, label, color) in enumerate(SERIES):
        values = {"reactants": 0.0, **data[key]}
        offset = (n - (len(SERIES) - 1) / 2) * slot
        xs = [k + offset for k in range(len(POINTS))]
        ys = [values[name] for name, _ in POINTS]
        heavy = n == 0
        for x, y in zip(xs, ys, strict=True):
            ax.plot([x - width / 2, x + width / 2], [y, y], color=color, lw=3.5 if heavy else 2.5,
                    solid_capstyle="round", zorder=3)
        ax.plot(xs, ys, color=color, lw=0.8, alpha=0.6, zorder=2,
                ls="-" if heavy else (0, (3, 2)))
        ax.plot([], [], color=color, lw=3.5 if heavy else 2.5, label=label)
    # The quantity that matters most: the barrier from the ion–dipole complex.
    # Neighbouring levels crowd the TS: the literature label goes above its
    # level, AIMNet2's below its own.
    for n, (key, _, _) in enumerate(SERIES):
        below = {"Czakó FPA": False, "AIMNet2": True}.get(key)
        if below is not None:
            offset = (n - (len(SERIES) - 1) / 2) * slot
            ax.annotate(f"{data[key]['barrier']:+.1f}", (2 + offset, data[key]["ts"]),
                        xytext=(0, -6 if below else 6), textcoords="offset points",
                        ha="center", va="top" if below else "bottom", color=INK2, fontsize=8.5)
    ax.set_xticks(range(len(POINTS)), [label for _, label in POINTS], color=INK2)
    ax.set_ylabel("Energy relative to F⁻ + CH₃Cl (kcal/mol)", color=INK2)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK2, fontsize=9)
    ax.set_title("F⁻ + CH₃Cl → CH₃F + Cl⁻: stationary points, classical energies "
                 "(numbers at the TS: barrier from F⁻···CH₃Cl)",
                 loc="left", color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(OUT, dpi=110, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
