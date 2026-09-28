"""Step 5: the pictures.  -> images/bbvo_<run>.png

Three panels on the held-out frames (pristine 600 K MD and, if labeled, the
doped ``*_test`` frames) and the rigidly scaled cells:

1. forces: every component of the model against the target;
2. stress: every Voigt component (GPa), the same way;
3. energy per atom against the lattice scale (ions at the PBE+U fractional
   positions), the target's points and each model's curve.

Usage: ``figures.py [--dry-run]``. With ``--dry-run`` the target is the synthetic
one of ``fake_labels.py``, and the title says so.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.units import GPa
from common import HERE, held_out, labels, mace_mp0, primitive, rigid_scaled_cells, run_dir

from samson_mlip_visualizer.calculators import create_calculator

DRY = "--dry-run" in sys.argv
run = run_dir(DRY)
labeled = labels(DRY)
test = [f for f in labeled if held_out(f)]
iso = rigid_scaled_cells(labeled)
MODELS = {
    "MACE-MP-0 alone": ("#8a8985", lambda: mace_mp0()),
    "MACE-MP-0 + Δ": ("#eb6834", lambda: create_calculator(
        "mace", sorted((run / "models" / "mace-mp0+delta").glob("*.model")), device="cuda")),
    "MACE-MP-0 fine-tuned": ("#2a78d6", lambda: create_calculator(
        "mace", sorted((run / "models" / "direct").glob("*.model")), device="cuda")),
}
target = "synthetic target" if DRY else "HSE06"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
ref_f = np.concatenate([f.arrays["HSE06_forces"] for f in test]).ravel()
ref_s = np.concatenate([f.info["HSE06_stress"] for f in test]) / GPa
scales = np.array([s for s, _ in iso])
ref_e = np.array([f.info["HSE06_energy"] / len(f) for _, f in iso])
axes[2].plot(scales, 1000 * (ref_e - ref_e.min()), "o", color="#0b0b0b", ms=6, zorder=5,
             label=target)
dense = np.linspace(scales.min(), scales.max(), 41)
for name, (color, build) in MODELS.items():
    calc = build()
    f_pred, s_pred = [], []
    for frame in test:
        atoms = frame.copy()
        atoms.calc = calc
        f_pred.append(atoms.get_forces())
        s_pred.append(atoms.get_stress() / GPa)
    f_pred = np.concatenate(f_pred).ravel()
    s_pred = np.concatenate(s_pred)
    f_rmse = np.sqrt(np.mean((f_pred - ref_f) ** 2))
    s_rmse = np.sqrt(np.mean((s_pred - ref_s) ** 2))
    axes[0].plot(ref_f, f_pred, ".", ms=3, color=color, alpha=0.6,
                 label=f"{name} (RMSE {f_rmse:.3f} eV/Å)")
    axes[1].plot(ref_s, s_pred, "o", ms=4, color=color, alpha=0.8,
                 label=f"{name} (RMSE {s_rmse:.2f} GPa)")
    energies = []
    for scale in dense:
        cell = primitive(scale)
        cell.calc = calc
        energies.append(cell.get_potential_energy() / len(cell))
    energies = np.array(energies)
    # each curve on its own zero: only the shape and the minimum are comparable
    axes[2].plot(dense, 1000 * (energies - energies.min()), color=color, lw=2, label=name)
for ax, (title, unit) in zip(axes[:2], (("Forces, held-out frames", "eV/Å"),
                                          ("Stress, held-out frames", "GPa")), strict=True):
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color="#c3c2b7", lw=1, zorder=0)
    ax.set_xlabel(f"{target} ({unit})")
    ax.set_ylabel(f"model ({unit})")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="#e4e3df", lw=0.6)
axes[2].set_title("Energy vs lattice scale, ions clamped", loc="left")
axes[2].set_xlabel("lattice scale (1 = PBE+U, a = 8.487 Å)")
axes[2].set_ylabel("energy per atom above its minimum (meV)")
axes[2].legend(frameon=False, fontsize=8)
axes[2].grid(True, color="#e4e3df", lw=0.6)
title = "Ba₂BiVO₆, MACE-MP-0 + Δ: " + ("DRY RUN on synthetic labels (pipeline check, not "
                                       "HSE06)" if DRY else "against HSE06 (VASP)")
fig.suptitle(title, x=0.01, ha="left", fontsize=11)
(HERE / "images").mkdir(exist_ok=True)
out = HERE / "images" / f"bbvo_{'dry_run' if DRY else 'hse06'}.png"
fig.savefig(out, dpi=150)
print(out)
