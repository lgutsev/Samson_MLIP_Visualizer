"""Step 6: transferability. How do the HCN models do on molecules they never saw?
-> transfer_set.extxyz, transfer_results.json, images/delta_transfer.png

Only H, C, and N (the Δ-corrections know no other element), graded by how much
they resemble the training data (HCN / HNC along the isomerization, plus C–H and
N–H stretches):

- ``control``: HCN itself;
- ``close``: a C≡N group in a bigger molecule (CH₃CN, CH₃NC, NCCN, acrylonitrile);
- ``medium``: other multiple bonds between C and N (CH₂=NH, pyridine, C₂H₂, N₂);
- ``far``: saturated molecules (CH₄, NH₃, CH₃NH₂) and ethylene.

Each molecule is relaxed with GFN1-xTB, then rattled (6 copies, σ 0.03 Å). Errors
against PBE/def2-TZVP: energies relative to the molecule's own relaxed geometry
(so per-element offsets cancel: the HCN data pin only one composition) and forces.
Plus one reaction none of the models was trained on, CH₃NC → CH₃CN (the methyl
analogue of HNC → HCN): each method relaxes both isomers on its own surface.

The models are the final ones of the HCN example: trained on the 87 path
structures plus the 8 bond stretches (``stretch_round.py``).
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from common import FOUNDATION, HERE, WORK, pbe, xtb

from samson_mlip_visualizer.calculators import create_calculator
from samson_mlip_visualizer.engine import relax

GROUPS = {
    "control": ["HCN"],
    "close": ["CH3CN", "CH3NC", "NCCN", "H2CCHCN"],
    "medium": ["CH2NH", "C5H5N", "C2H2", "N2"],
    "far": ["CH4", "NH3", "H3CNH2", "C2H4"],
}
LABELS = {"HCN": "HCN", "CH3CN": "CH₃CN", "CH3NC": "CH₃NC", "NCCN": "NCCN",
          "H2CCHCN": "acrylonitrile", "CH2NH": "CH₂=NH", "C5H5N": "pyridine",
          "C2H2": "HC≡CH", "N2": "N₂", "CH4": "CH₄", "NH3": "NH₃", "H3CNH2": "CH₃NH₂",
          "C2H4": "H₂C=CH₂"}
RATTLED = 6
SIGMA = 0.03


def build(name):
    if name == "CH3NC":  # methyl isocyanide: C–N≡C, C3v
        return Atoms("CNCHHH", positions=[
            [0, 0, 0], [0, 0, 1.42], [0, 0, 2.60],
            [1.03, 0, -0.36], [-0.515, 0.892, -0.36], [-0.515, -0.892, -0.36]])
    if name == "CH2NH":  # methanimine
        return Atoms("CNHHH", positions=[
            [0, 0, 0], [0, 0, 1.27], [0.94, 0, -0.55], [-0.94, 0, -0.55], [0.87, 0, 1.70]])
    return molecule(name)


def energy_forces(calc, atoms):
    image = atoms.copy()
    image.calc = calc
    return image.get_potential_energy(), image.get_forces()


# --- the test set, labeled with PBE ---------------------------------------------------
test_file = WORK / "transfer_set.extxyz"
if not test_file.exists():
    reference, gfn1 = pbe(), xtb("gfn1")
    rng = np.random.default_rng(7)
    frames, start = [], time.perf_counter()
    for group, names in GROUPS.items():
        for name in names:
            atoms = build(name)
            atoms.calc = gfn1
            relax(atoms, fmax=0.01, max_steps=300, optimizer="BFGS")
            geometries = [atoms.positions.copy()] + [
                atoms.positions + rng.normal(0, SIGMA, atoms.positions.shape)
                for _ in range(RATTLED)]
            for k, positions in enumerate(geometries):
                frame = Atoms(atoms.get_chemical_symbols(), positions=positions)
                energy, forces = energy_forces(reference, frame)
                frame.info.update(molecule=name, group=group, index=k, PBE_energy=energy)
                frame.arrays["PBE_forces"] = forces
                frames.append(frame)
            print(f"{name:<8} labeled ({time.perf_counter() - start:.0f} s)", flush=True)
    write(test_file, frames)
frames = read(test_file, ":")

# --- the methods ---------------------------------------------------------------------
M = WORK / "models"
methods = {
    "MACE-MP-0": lambda: create_calculator("mace", FOUNDATION, device="cuda"),
    "GFN1-xTB": lambda: xtb("gfn1"),
    "GFN2-xTB": lambda: xtb("gfn2"),
    "fine-tuned MACE": lambda: create_calculator(
        "mace", M / "direct+stretch_N95" / "direct+stretch_N95_seed1.model", device="cuda"),
    "GFN1-xTB + Δ": lambda: create_calculator(
        "mace", M / "delta-gfn1+stretch_N95" / "delta-gfn1+stretch_N95_seed1.model",
        device="cuda"),
    "GFN2-xTB + Δ": lambda: create_calculator(
        "mace", M / "delta-gfn2+stretch_N95" / "delta-gfn2+stretch_N95_seed1.model",
        device="cuda"),
}


def isomerization(calc):
    """E(CH₃CN) − E(CH₃NC), each relaxed on the method's own surface."""
    energies = {}
    for name in ("CH3NC", "CH3CN"):
        atoms = next(f for f in frames if f.info["molecule"] == name and f.info["index"] == 0)
        atoms = atoms.copy()
        atoms.calc = calc
        relax(atoms, fmax=0.005, max_steps=400, optimizer="BFGS")
        energies[name] = atoms.get_potential_energy()
    return energies["CH3CN"] - energies["CH3NC"]


results_file = WORK / "transfer_results.json"
results = json.loads(results_file.read_text()) if results_file.exists() else {}
if "PBE" not in results:
    results["PBE"] = {"isomerization_ev": isomerization(pbe())}
for name, build_calc in methods.items():
    if name in results:
        continue
    calc = build_calc()
    per_molecule = {}
    for mol in [m for names in GROUPS.values() for m in names]:
        own = [f for f in frames if f.info["molecule"] == mol]
        e0, _ = energy_forces(calc, own[0])
        e_err, f_err = [], []
        for frame in own:
            energy, forces = energy_forces(calc, frame)
            e_err.append((energy - e0) - (frame.info["PBE_energy"] - own[0].info["PBE_energy"]))
            f_err.append(forces - frame.arrays["PBE_forces"])
        per_molecule[mol] = {
            "energy_mae_ev": float(np.mean(np.abs(e_err[1:]))),
            "force_rmse_ev_A": float(np.sqrt(np.mean(np.concatenate(f_err) ** 2))),
        }
    results[name] = {"molecules": per_molecule, "isomerization_ev": isomerization(calc)}
    results_file.write_text(json.dumps(results, indent=1))
    print(f"{name}: done", flush=True)

# --- report --------------------------------------------------------------------------
names = [m for group in GROUPS.values() for m in group]
print(f"\n{'':<10}" + "".join(f"{n[:14]:>16}" for n in methods))
for key, unit in (("energy_mae_ev", "E MAE"), ("force_rmse_ev_A", "F RMSE")):
    print(unit)
    for mol in names:
        values = [results[n]["molecules"][mol][key] for n in methods]
        print(f"  {mol:<8}" + "".join(f"{v:16.3f}" for v in values))
print("CH3NC -> CH3CN (eV):  PBE " + f"{results['PBE']['isomerization_ev']:.3f}  " +
      "  ".join(f"{n} {results[n]['isomerization_ev']:.3f}" for n in methods))

# --- figure: one row per molecule, grouped by similarity to HCN ------------------------
STYLE = {  # trained models in color, methods used as they come in gray with open markers
    "fine-tuned MACE": dict(color="#2a78d6", marker="o", mfc="#2a78d6"),
    "GFN1-xTB + Δ": dict(color="#eb6834", marker="s", mfc="#eb6834"),
    "GFN2-xTB + Δ": dict(color="#1baf7a", marker="D", mfc="#1baf7a"),
    "MACE-MP-0": dict(color="#8a8985", marker="o", mfc="none"),
    "GFN1-xTB": dict(color="#8a8985", marker="s", mfc="none"),
    "GFN2-xTB": dict(color="#8a8985", marker="D", mfc="none"),
}
rows = []
for group, members in GROUPS.items():
    rows += [(group, m) for m in members]
y = {}
position = 0
for i, (group, mol) in enumerate(rows):
    if i and rows[i - 1][0] != group:
        position += 0.8  # a gap between groups
    y[mol] = position
    position += 1
fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True, constrained_layout=True)
for ax, (key, title) in zip(axes, (("energy_mae_ev", "Energy MAE, rattled geometries (eV)"),
                                   ("force_rmse_ev_A", "Force RMSE (eV/Å)")), strict=True):
    for name, style in STYLE.items():
        values = [results[name]["molecules"][mol][key] for _, mol in rows]
        ax.plot(values, [y[mol] for _, mol in rows], linestyle="none", markersize=7,
                markeredgewidth=1.6, label=name, **style)
    ax.set_xscale("log")
    ax.set_title(title, loc="left")
    ax.grid(True, axis="x", color="#e4e3df", lw=0.6)
    for group in GROUPS:
        members = [y[m] for g, m in rows if g == group]
        ax.axhspan(min(members) - 0.45, max(members) + 0.45,
                   color="#f1f2ef" if list(GROUPS).index(group) % 2 else "none", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks([y[m] for _, m in rows],
                   labels=[f"{LABELS[m]}  ({g})" for g, m in rows])
axes[0].invert_yaxis()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, fontsize=9)
fig.suptitle("HCN-trained models on molecules they never saw (vs PBE/def2-TZVP)",
             x=0.01, ha="left", fontsize=11)
fig.savefig(HERE / "images" / "delta_transfer.png", dpi=150)
print("figure written")
