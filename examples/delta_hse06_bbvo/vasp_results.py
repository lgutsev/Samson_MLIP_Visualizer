"""The first real VASP labels from LONI against MACE-MP-0.  -> vasp_results.json, images/

Reads, without writing into them, two finished smoke tests in
``D:\\MLIP_Work_Folder\\hpc_smoke_tests`` (``SMOKE_DIR``):

- ``05_vasp_bbvo``: PBE+U and HSE06 on the primitive cell, a rattled primitive
  cell, and a 40-atom 300 K MD frame;
- ``08_vasp_bbvo_stability``: the same two levels on the cubic 40-atom cell, the
  cell with the ions relaxed by MACE-MP-0, and the fully relaxed MACE-MP-0
  structure (``singlepoints/``), plus a symmetry-free PBE+U relaxation from the
  rattled cubic cell (``relax/``: ions, then cell and ions).

It reports MACE-MP-0's errors against PBE+U on every frame (what the correction
starts from), the size of HSE06 − PBE+U (what it has to learn), and the energy
ladder of the stability test at all three levels. The labels are also written
to ``WORK/loni_round1.extxyz``.
"""

import json
import os
from pathlib import Path

import numpy as np
from ase.io import read, write
from common import HERE, WORK, mace_mp0

from samson_mlip_visualizer.vasp_labeling import collect_vasp_labels

SMOKE = Path(os.environ.get("SMOKE_DIR", r"D:\MLIP_Work_Folder\hpc_smoke_tests"))
PACKAGES = {"05": SMOKE / "05_vasp_bbvo", "08": SMOKE / "08_vasp_bbvo_stability" / "singlepoints"}
RELAX = SMOKE / "08_vasp_bbvo_stability" / "relax"
GPA = 160.21766208  # eV/Å³ -> GPa
STABILITY = ["cubic (Fm-3m, PBE+U lattice)", "ions relaxed by MACE-MP-0 (cubic cell)",
             "fully relaxed by MACE-MP-0"]


def oszicar_energies(path):
    return [float(line.split("F=")[1].split()[0]) for line in open(path) if " F=" in line]


def main():
    calc = mace_mp0(device="cpu")
    frames, rows = [], []
    for tag, package in PACKAGES.items():
        result = collect_vasp_labels(package, write=False)
        if result.rejected:
            raise SystemExit(f"{package}: rejected {result.rejected}")
        for i, atoms in enumerate(result.labeled):
            atoms.info["package"] = tag
            n, fu = len(atoms), len(atoms) / 10
            mace = atoms.copy()
            mace.calc = calc
            e = mace.get_potential_energy()
            f = mace.get_forces()
            s = mace.get_stress(voigt=True)
            row = {"package": tag, "frame": i, "atoms": n,
                   "group": atoms.info.get("group", ""),
                   "PBEU_eV": atoms.info["PBEU_energy"], "HSE06_eV": atoms.info["HSE06_energy"],
                   "MACE_eV": float(e)}
            for level in ("PBEU", "HSE06"):
                forces = atoms.arrays[f"{level}_forces"]
                stress = atoms.info[f"{level}_stress"]
                row[f"MACE_minus_{level}_meV_per_atom"] = 1000 * (e - row[f"{level}_eV"]) / n
                row[f"MACE_vs_{level}_force_rmse"] = float(np.sqrt(((f - forces) ** 2).mean()))
                row[f"{level}_max_force"] = float(np.linalg.norm(forces, axis=1).max())
                row[f"MACE_vs_{level}_stress_rmse_GPa"] = float(
                    GPA * np.sqrt(((s - np.asarray(stress)) ** 2).mean()))
            row["HSE06_minus_PBEU_meV_per_atom"] = 1000 * (row["HSE06_eV"] - row["PBEU_eV"]) / n
            row["HSE06_vs_PBEU_force_rmse"] = float(np.sqrt(
                ((atoms.arrays["HSE06_forces"] - atoms.arrays["PBEU_forces"]) ** 2).mean()))
            row["per_fu"] = fu
            rows.append(row)
            frames.append(atoms)
    write(WORK / "loni_round1.extxyz", frames)

    # the stability ladder, meV per formula unit relative to cubic
    stab = [r for r in rows if r["package"] == "08"]
    ladder = {}
    for level, key in (("PBE+U", "PBEU_eV"), ("HSE06", "HSE06_eV"), ("MACE-MP-0", "MACE_eV")):
        ladder[level] = {name: 1000 * (r[key] - stab[0][key]) / r["per_fu"]
                         for name, r in zip(STABILITY, stab)}
    # the symmetry-free PBE+U relaxation (no MACE-MP-0 involved)
    ions, cell = oszicar_energies(RELAX / "OSZICAR.1_ions"), oszicar_energies(RELAX / "OSZICAR.2_cell")
    relaxed = read(RELAX / "CONTCAR.2_cell")
    relaxed.calc = calc
    ladder["PBE+U"]["PBE+U relaxation from a 0.05 Å rattle"] = 1000 * (
        cell[-1] - stab[0]["PBEU_eV"]) / 4
    ladder["MACE-MP-0"]["PBE+U relaxation from a 0.05 Å rattle"] = 1000 * (
        relaxed.get_potential_energy() - stab[0]["MACE_eV"]) / 4
    lengths = relaxed.cell.lengths()
    out = {"frames": rows, "stability_meV_per_fu": ladder,
           "pbeu_relaxation": {"ionic_steps": [len(ions), len(cell)],
                               "final_eV": cell[-1], "cell_A": lengths.round(3).tolist(),
                               "angles_deg": relaxed.cell.angles().round(2).tolist(),
                               "volume_A3": float(relaxed.get_volume()),
                               "cubic_volume_A3": float(read(RELAX / "POSCAR").get_volume()),
                               "MACE_max_force_eV_per_A": float(
                                   np.linalg.norm(relaxed.get_forces(), axis=1).max())}}
    (HERE / "vasp_results.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['package']} frame {r['frame']} ({r['atoms']} atoms): MACE−PBE+U "
              f"{r['MACE_minus_PBEU_meV_per_atom']:+.1f} meV/atom, F {r['MACE_vs_PBEU_force_rmse']:.3f},"
              f" σ {r['MACE_vs_PBEU_stress_rmse_GPa']:.2f} GPa; HSE06−PBE+U "
              f"{r['HSE06_minus_PBEU_meV_per_atom']:+.0f} meV/atom, F {r['HSE06_vs_PBEU_force_rmse']:.3f}")
    print(json.dumps(ladder, indent=1))
    print(json.dumps(out["pbeu_relaxation"], indent=1))
    figure(ladder)


def figure(ladder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = STABILITY[:2] + ["PBE+U relaxation from a 0.05 Å rattle", STABILITY[2]]
    short = ["cubic\n(Fm-3m)", "ions relaxed\nby MACE-MP-0", "PBE+U relaxed\n(no symmetry)",
             "fully relaxed\nby MACE-MP-0"]
    colors = {"PBE+U": "#2a78d6", "HSE06": "#eb6834", "MACE-MP-0": "#8a8985"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.26
    for k, (level, color) in enumerate(colors.items()):
        x = np.arange(len(names)) + (k - 1) * (width + 0.02)
        vals = [ladder[level].get(n, np.nan) for n in names]
        ax.bar(x, vals, width, color=color, label=level)
        for xi, v in zip(x, vals):
            if np.isfinite(v) and abs(v) > 0.5:
                ax.text(xi, v - 15, f"{v:.0f}", ha="center", va="top", fontsize=8,
                        color="#52514e")
            elif not np.isfinite(v):
                ax.text(xi, -12, "n/a", ha="center", va="top", fontsize=7, color="#8a8985")
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks(range(len(names)), short, fontsize=9)
    ax.set_ylabel("energy relative to cubic (meV per Ba₂BiVO₆)")
    ax.set_title("Cubic Ba₂BiVO₆ is a saddle point at PBE+U and HSE06 (40-atom cell)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(True, axis="y", color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    (HERE / "images").mkdir(exist_ok=True)
    fig.savefig(HERE / "images" / "bbvo_stability_dft.png", dpi=150)


if __name__ == "__main__":
    main()
