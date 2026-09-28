"""Figure: stock UMA against ωB97X-D and the fine-tuned AIMNet2 on the SN2 frames.

    python plot_uma.py

Reads the same-frame CSVs that ``evaluate_uma.py`` (and, for the fine-tuned
AIMNet2, ``examples/sn2_f_ch3cl/evaluate_aimnet2_tuned.py --fragments``) wrote,
and draws energies along the fine-tuned MACE IRC and the r(C–F) scan, zeroed at
the F⁻···CH₃Cl end, with the error against ωB97X-D below. Writes
``images/uma_vs_wb97xd.png``.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sn2_f_ch3cl"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sn2_common import KCAL, WORK  # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
MODELS = [  # label, CSV folder and prefix, colour
    ("UMA uma-s-1p1 (stock)", WORK / "uma" / "uma-s-1p1", "", "#2a78d6"),
    ("UMA uma-s-1p2 (stock)", WORK / "uma" / "uma-s-1p2", "", "#1baf7a"),
    ("fine-tuned AIMNet2 + fragments", WORK / "finetune_aimnet2", "_fragments", "#eb6834"),
]
PATHS = [("mace_tuned_irc", "fine-tuned MACE IRC", "IRC coordinate (Å·amu½)"),
         ("scan", "r(C–F) scan, off the IRC", "r(C–F) (Å)")]


def load(folder, key, suffix):
    with open(folder / f"{key}{suffix}_vs_wb97xd.csv", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, data = rows[0], np.array(rows[1:], float)
    model = next(i for i, h in enumerate(header) if h.startswith("dE_") and "ωB97X-D" not in h)
    reference = next(i for i, h in enumerate(header) if h.startswith("dE_") and "ωB97X-D" in h)
    return data[:, 0], data[:, model], data[:, reference]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)


def main():
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.4), facecolor=SURFACE, sharex="col",
                                gridspec_kw={"height_ratios": [2.2, 1]})
    for column, (key, title, xlabel) in enumerate(PATHS):
        top, bottom = axes[0, column], axes[1, column]
        style(top)
        style(bottom)
        reference_drawn = False
        for label, folder, suffix, colour in MODELS:
            x, model, reference = load(folder, key, suffix)
            order = np.argsort(x)
            x, model, reference = x[order], model[order], reference[order]
            # zero at the F⁻···CH₃Cl end, the higher of the two (27 kcal/mol above FCH₃···Cl⁻)
            start = 0 if reference[0] > reference[-1] else len(x) - 1
            model, reference = model - model[start], reference - reference[start]
            if not reference_drawn:
                top.plot(x, reference * KCAL, color=INK, lw=2.6, label="ωB97X-D/def2-TZVPD")
                reference_drawn = True
            top.plot(x, model * KCAL, color=colour, lw=1.6, marker="o", ms=3.5, label=label)
            bottom.plot(x, (model - reference) * 1000, color=colour, lw=1.4, marker="o", ms=3)
        bottom.axhline(0, color=MUTED, lw=0.8)
        top.set_title(title, loc="left", color=INK, fontsize=11)
        top.set_ylabel("E − E(F⁻···CH₃Cl) (kcal/mol)", color=INK2)
        bottom.set_ylabel("error vs ωB97X-D (meV)", color=INK2)
        bottom.set_xlabel(xlabel, color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    figure.suptitle("F⁻ + CH₃Cl: stock UMA (omol task) on frames labeled with ωB97X-D",
                    x=0.01, ha="left", color=INK, fontsize=12)
    figure.tight_layout()
    out = HERE / "images" / "uma_vs_wb97xd.png"
    out.parent.mkdir(exist_ok=True)
    figure.savefig(out, dpi=110, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
