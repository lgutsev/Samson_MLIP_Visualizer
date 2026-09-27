"""Energy-level diagrams of the stationary points: each method next to the literature.

    python energy_levels.py [work folder] [output png]

1. From the separated reactants (``comparison.json``, from ``compare.py``).
2. From the reactant complex, complex -> TS -> complex (``<output>_from_complex.png``):
   each model at its own stationary points (AIMNet2 from ``comparison.json``,
   the fine-tuned committee from ``finetune/result.json``, stock MACE-MP-0 from
   ``stock_mace_stationary.json`` if it has a Walden TS), ωB97X-D at the
   AIMNet2 geometries, and the literature.

One short level per method at each stationary point, joined along the reaction
by thin lines; the best literature value (Czakó's CCSD(T)/CBS focal-point
analysis) is drawn in ink as the reference.
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


def draw_levels(ax, series, points, width, slot):
    """``series``: [(label, color, {point: value})], the first drawn heavier (the reference)."""
    for n, (label, color, values) in enumerate(series):
        offset = (n - (len(series) - 1) / 2) * slot
        xs = [k + offset for k in range(len(points))]
        ys = [values[name] for name, _ in points]
        heavy = n == 0
        for x, y in zip(xs, ys, strict=True):
            ax.plot([x - width / 2, x + width / 2], [y, y], color=color, lw=3.5 if heavy else 2.5,
                    solid_capstyle="round", zorder=3)
        ax.plot(xs, ys, color=color, lw=0.8, alpha=0.6, zorder=2,
                ls="-" if heavy else (0, (3, 2)))
        ax.plot([], [], color=color, lw=3.5 if heavy else 2.5, label=label)
    ax.set_xticks(range(len(points)), [label for _, label in points], color=INK2)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)


TUNED_AIMNET2 = ("fine-tuned AIMNet2, complexes only", "#4a3aa7")  # categorical slot 7
TUNED_AIMNET2_FRAGMENTS = ("fine-tuned AIMNet2 + fragments", "#e34948")  # slot 8


def tuned_aimnet2(suffix=""):
    """A fine-tuned AIMNet2's evaluation (``evaluate_aimnet2_tuned.py``), if run."""
    path = WORK / "finetune_aimnet2" / f"evaluation{suffix}.json"
    return json.loads(path.read_text()) if path.exists() else None


def aimnet2_runs():
    return [(style, evaluation) for style, evaluation in
            ((TUNED_AIMNET2, tuned_aimnet2()),
             (TUNED_AIMNET2_FRAGMENTS, tuned_aimnet2("_fragments"))) if evaluation]


def main():
    data = json.loads((WORK / "comparison.json").read_text())
    figure, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    series = [(label, color, {"reactants": 0.0, **data[key]}) for key, label, color in SERIES]
    # The fine-tuned AIMNet2s keep the charge input, so they alone among the
    # fine-tunes have energies relative to the separated fragments.
    for style, evaluation in aimnet2_runs():
        if "asymptotes" in evaluation:
            series.append((*style, {"reactants": 0.0, **evaluation["asymptotes"]}))
    width, slot = (0.13, 0.165) if len(series) <= 5 else (0.10, 0.125)
    draw_levels(ax, series, POINTS, width, slot)
    # The quantity that matters most: the barrier from the ion–dipole complex.
    # Neighbouring levels crowd the TS: the literature label goes above its
    # level, AIMNet2's below its own.
    for n, (key, _, _) in enumerate(SERIES):
        below = {"Czakó FPA": False, "AIMNet2": True}.get(key)
        if below is not None:
            offset = (n - (len(series) - 1) / 2) * slot
            ax.annotate(f"{data[key]['barrier']:+.1f}", (2 + offset, data[key]["ts"]),
                        xytext=(0, -6 if below else 6), textcoords="offset points",
                        ha="center", va="top" if below else "bottom", color=INK2, fontsize=8.5)
    ax.set_ylabel("Energy relative to F⁻ + CH₃Cl (kcal/mol)", color=INK2)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK2, fontsize=9)
    ax.set_title("F⁻ + CH₃Cl → CH₃F + Cl⁻: stationary points, classical energies "
                 "(numbers at the TS: barrier from F⁻···CH₃Cl)",
                 loc="left", color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(OUT, dpi=110, facecolor=SURFACE)
    print(OUT)
    from_complex(data)


def from_complex(data):
    """Complex -> TS -> complex, each model at its own stationary points."""
    ev = 23.0605  # kcal/mol per eV
    tuned = json.loads((WORK / "finetune" / "result.json").read_text())
    stock_file = WORK / "stock_mace_stationary.json"
    stock = json.loads(stock_file.read_text()) if stock_file.exists() else {}

    def levels(barrier, c2c):
        return {"reactant_complex": 0.0, "ts": barrier, "product_complex": c2c}

    def from_table(key):
        row = data[key]
        return levels(row["barrier"], row["product_complex"] - row["reactant_complex"])

    # Colors follow the entity across both diagrams; the MACE models take slots 5-6.
    series = [
        ("CCSD(T)/CBS focal point (literature)", INK, levels(3.39, -25.95)),
        ("AIMNet2", "#2a78d6", from_table("AIMNet2")),
        ("ωB97X-D/def2-TZVPD (at AIMNet2 geometries)", "#eb6834",
         from_table("wB97X-D/def2-TZVPD")),
        ("fine-tuned MACE (3-model committee)", "#e87ba4",
         levels(tuned["barrier_from_reactant_complex_ev"] * ev,
                tuned["product_complex_minus_reactant_complex_ev"] * ev)),
    ]
    if stock.get("walden_ts_found"):
        series.append(("MACE-MP-0 small", "#008300",
                       levels(stock["barrier_kcal"], stock["complex_to_complex_kcal"])))
    for style, evaluation in aimnet2_runs():
        if "ts" in evaluation:
            series.append((*style, levels(evaluation["ts"]["barrier_kcal"],
                                          evaluation["ts"]["complex_to_complex_kcal"])))
    series = [(f"{label}: barrier {values['ts']:.1f}", color, values)
              for label, color, values in series]
    points = [("reactant_complex", "F⁻···CH₃Cl"), ("ts", "Walden TS"),
              ("product_complex", "FCH₃···Cl⁻")]
    figure, ax = plt.subplots(figsize=(8.5, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    draw_levels(ax, series, points, 0.13, 0.16)
    ax.set_ylabel("Energy relative to F⁻···CH₃Cl (kcal/mol)", color=INK2)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK2, fontsize=9)
    if not stock.get("walden_ts_found"):
        ax.text(0.99, 0.97, "MACE-MP-0 small: no Walden TS of its own (see text)",
                transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    ax.set_title("From the reactant complex: complex → TS → complex, kcal/mol",
                 loc="left", color=INK, fontsize=11)
    figure.tight_layout()
    out = OUT.with_name(OUT.stem + "_from_complex.png")
    figure.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
