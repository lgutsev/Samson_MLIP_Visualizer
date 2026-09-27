"""Step 5: the pictures of the worked example.  -> images/*.png, figures_data.json

1. ``delta_decomposition.png``: along the isomerization path (the 29 IRC frames
   of the training data, by H–C–N angle), PBE = xTB + residual, and what the
   Δ-model learned of the residual.
2. ``delta_stretches.png``: breaking C–H in HCN and N–H in HNC, which no
   training structure contains: PBE (held-out points) against every model.
3. ``delta_stretch_round.png``: the same stretches after one active-learning
   round (``stretch_round.py``: 8 stretched geometries labeled with PBE).
4. The learning curves come from ``plot.py``.

Uses the models trained on all 87 structures (seed 1), and the 95-structure
models of the stretch round.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from common import FOUNDATION, HERE, POOL, TEST_SET, WORK, hcn, hnc, xtb

from samson_mlip_visualizer.calculators import create_calculator

IMAGES = HERE / "images"
IMAGES.mkdir(exist_ok=True)
N = 87
models = {
    "PBE-trained MACE (direct)": create_calculator(
        "mace", WORK / "models" / f"direct_N{N}" / f"direct_N{N}_seed1.model", device="cuda"),
    "GFN1-xTB + Δ": create_calculator(
        "mace", WORK / "models" / f"delta-gfn1_N{N}" / f"delta-gfn1_N{N}_seed1.model",
        device="cuda"),
    "GFN2-xTB + Δ": create_calculator(
        "mace", WORK / "models" / f"delta-gfn2_N{N}" / f"delta-gfn2_N{N}_seed1.model",
        device="cuda"),
    "MACE-MP-0": create_calculator("mace", FOUNDATION, device="cuda"),
    "GFN1-xTB": xtb("gfn1"),
    "GFN2-xTB": xtb("gfn2"),
    "direct + stretch labels": create_calculator(
        "mace", WORK / "models" / "direct+stretch_N95" / "direct+stretch_N95_seed1.model",
        device="cuda"),
    "GFN1-xTB + Δ + stretch labels": create_calculator(
        "mace", WORK / "models" / "delta-gfn1+stretch_N95" / "delta-gfn1+stretch_N95_seed1.model",
        device="cuda"),
    "GFN2-xTB + Δ + stretch labels": create_calculator(
        "mace", WORK / "models" / "delta-gfn2+stretch_N95" / "delta-gfn2+stretch_N95_seed1.model",
        device="cuda"),
}
COLORS = {"PBE": "#0b0b0b", "PBE-trained MACE (direct)": "#2a78d6", "GFN1-xTB + Δ": "#eb6834",
          "GFN2-xTB + Δ": "#1baf7a", "MACE-MP-0": "#8a8985", "GFN1-xTB": "#eb6834",
          "GFN2-xTB": "#1baf7a", "direct + stretch labels": "#2a78d6",
          "GFN1-xTB + Δ + stretch labels": "#eb6834", "GFN2-xTB + Δ + stretch labels": "#1baf7a"}


def energies(calc, frames, part="energy"):
    out = []
    for frame in frames:
        atoms = frame.copy()
        atoms.calc = calc
        atoms.get_potential_energy()
        out.append(calc.results[part])
    return np.array(out)


data = {}

# --- 1. decomposition along the path --------------------------------------------
irc = [a for a in read(POOL, ":") if a.info["tag"] == "irc0"]
angle = np.array([a.get_angle(2, 0, 1) for a in irc])
pbe = np.array([a.info["REF_energy"] for a in irc])
path = {"angle_deg": angle.tolist(), "PBE": (pbe - pbe[0]).tolist()}
for method, name in (("gfn1", "GFN1-xTB + Δ"), ("gfn2", "GFN2-xTB + Δ")):
    calc = models[name]
    base = energies(calc, irc, "baseline_energy")
    corr = energies(calc, irc, "correction_energy")
    path[f"{method}_baseline"] = (base - base[0]).tolist()
    path[f"{method}_residual"] = ((pbe - base) - (pbe - base)[0]).tolist()
    path[f"{method}_learned"] = (corr - corr[0]).tolist()
    path[f"{method}_total"] = ((base + corr) - (base + corr)[0]).tolist()
direct = energies(models["PBE-trained MACE (direct)"], irc)
mp0 = energies(models["MACE-MP-0"], irc)
path["direct"] = (direct - direct[0]).tolist()
path["MACE-MP-0"] = (mp0 - mp0[0]).tolist()
data["path"] = path

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True, sharex=True)
ax = axes[0]
ax.plot(angle, path["PBE"], color=COLORS["PBE"], lw=2, label="PBE (target)")
ax.plot(angle, path["gfn1_baseline"], color="#eb6834", lw=2, ls="--", label="GFN1-xTB")
ax.plot(angle, path["gfn2_baseline"], color="#1baf7a", lw=2, ls="--", label="GFN2-xTB")
ax.plot(angle, path["MACE-MP-0"], color="#8a8985", lw=1.5, ls=":", label="MACE-MP-0")
ax.set_title("① The baseline gets the shape, not the height", loc="left")
ax.set_ylabel("Energy relative to HCN (eV)")
ax = axes[1]
for method, color in (("gfn1", "#eb6834"), ("gfn2", "#1baf7a")):
    ax.plot(angle, path[f"{method}_residual"], color=color, lw=0, marker="o", ms=5,
            label=f"PBE − {method.upper()} (labels)")
    ax.plot(angle, path[f"{method}_learned"], color=color, lw=2,
            label=f"Δ-MACE on {method.upper()}")
ax.axhline(0, color="#c3c2b7", lw=0.8)
ax.set_title("② The correction learns the residual", loc="left")
ax.set_ylabel("Residual relative to HCN (eV)")
ax = axes[2]
ax.plot(angle, path["PBE"], color=COLORS["PBE"], lw=0, marker="o", ms=5, label="PBE")
ax.plot(angle, path["gfn1_total"], color="#eb6834", lw=2, label="GFN1-xTB + Δ")
ax.plot(angle, path["gfn2_total"], color="#1baf7a", lw=2, ls="--", label="GFN2-xTB + Δ")
ax.set_title("③ Baseline + correction ≈ PBE", loc="left")
for ax in axes:
    ax.set_xlabel("∠H–C–N (°): HCN at 180, HNC near 0")
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
axes[0].invert_xaxis()  # shared: every panel reads HCN -> HNC left to right
fig.savefig(IMAGES / "delta_decomposition.png", dpi=150)

# --- 2. bond stretches: held-out PBE points, model curves -------------------------
test = read(TEST_SET, ":")
anchor = next(a for a in test if a.info["group"] == "anchor")
stretch = {}
for group, build, grid, bond in (
    ("ch_stretch", lambda r: hcn(r_ch=r), np.linspace(0.95, 2.1, 47), (0, 2)),
    ("nh_stretch", lambda r: hnc(r_nh=r), np.linspace(0.85, 2.0, 47), (1, 2)),
):
    points = [a for a in test if a.info["group"] == group]
    ref0 = anchor.info["PBE_energy"]
    entry = {"r_pbe": [a.get_distance(*bond) for a in points],
             "PBE": [a.info["PBE_energy"] - ref0 for a in points], "r": grid.tolist()}
    frames = [build(r) for r in grid]
    for name, calc in models.items():
        e0 = energies(calc, [anchor])[0]
        entry[name] = (energies(calc, frames) - e0).tolist()
    stretch[group] = entry
data["stretch"] = stretch

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, (group, title, xlabel) in zip(axes, (
        ("ch_stretch", "Breaking C–H in HCN (not in the training data)", "r(C–H) (Å)"),
        ("nh_stretch", "Breaking N–H in HNC (not in the training data)", "r(N–H) (Å)")),
        strict=True):
    entry = stretch[group]
    for name, style in (("MACE-MP-0", ":"), ("GFN1-xTB", "--"),
                        ("PBE-trained MACE (direct)", "-"), ("GFN1-xTB + Δ", "-"),
                        ("GFN2-xTB + Δ", "-.")):
        ax.plot(entry["r"], entry[name], color=COLORS[name], ls=style, lw=2 if "Δ" in name
                or "direct" in name else 1.3, label=name)
    ax.plot(entry["r_pbe"], entry["PBE"], "o", color=COLORS["PBE"], ms=6, label="PBE (held out)")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy relative to HCN minimum (eV)")
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
fig.savefig(IMAGES / "delta_stretches.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, (group, title, xlabel) in zip(axes, (
        ("ch_stretch", "C–H in HCN, after 8 stretch labels", "r(C–H) (Å)"),
        ("nh_stretch", "N–H in HNC, after 8 stretch labels", "r(N–H) (Å)")), strict=True):
    entry = stretch[group]
    ax.plot(entry["r"], entry["GFN1-xTB + Δ"], color="#eb6834", lw=1.3, ls=":",
            label="GFN1-xTB + Δ, path data only")
    for name, style in (("direct + stretch labels", "-"), ("GFN1-xTB + Δ + stretch labels", "-"),
                        ("GFN2-xTB + Δ + stretch labels", "-.")):
        ax.plot(entry["r"], entry[name], color=COLORS[name], ls=style, lw=2, label=name)
    ax.plot(entry["r_pbe"], entry["PBE"], "o", color=COLORS["PBE"], ms=6, label="PBE (held out)")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy relative to HCN minimum (eV)")
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
fig.savefig(IMAGES / "delta_stretch_round.png", dpi=150)

(WORK / "figures_data.json").write_text(json.dumps(data))
print("figures written to", IMAGES)
