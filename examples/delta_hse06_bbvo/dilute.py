"""Step 1c (laptop): one Nb or Ta in large supercells, the dilute limit.

Caution: MACE-MP-0 finds the cubic host unstable (see ``stability_check.py``). In
the 270- and 1080-atom cells the doped relaxation falls partly into that
distortion while the host reference stays cubic, so their substitution
energies measure the host's instability, not the dopant. Rerun this once the
stability check has settled which host structure is right.
-> dilute.json, dilute_frames.extxyz, images/bbvo_dilute.png

The host is the MACE-MP-0-relaxed primitive cell, repeated: 2×2×2 and 3×3×3
primitive (80 and 270 atoms) and the 40-atom cubic cell repeated 2×2×2 and
3×3×3 (320 and 1080 atoms), so one dopant is x = 12.5, 3.7, 3.1, and 0.9 %.
The 40-atom cell itself (x = 25 %) is the starting point.

For each cell and dopant, one V is replaced and the ions relax at the host's
lattice (the dilute limit keeps the host's lattice). Reported:

- the substitution energy relative to the end members,
  E_sub = E(doped) − E(host) − [E(Ba₂BiMO₆) − E(Ba₂BiVO₆)] per formula unit: the
  energy of moving one M from its own compound into the V host. In the dilute
  limit it is the mixing energy per dopant;
- the M–O bond, the mean V–O of the V sites nearest the dopant, and the largest
  displacement of any O from its host position.

For labeling on LONI (``dilute_frames.extxyz``, all held out): the relaxed 80-atom
cell with one Nb and with one Ta, and a 600 K MACE-MP-0 MD frame of each. They test
whether a correction trained on 40-atom cells transfers to a dilute dopant.
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.build import make_supercell
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE
from common import HERE, WORK, grouped, mace_mp0, primitive

calc = mace_mp0()
CONV = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
CELLS = {"40 (cubic 1×1×1)": (CONV, 1), "80 (primitive 2×2×2)": (np.eye(3, dtype=int) * 2, 0),
         "270 (primitive 3×3×3)": (np.eye(3, dtype=int) * 3, 0),
         "320 (cubic 2×2×2)": (CONV, 2), "1080 (cubic 3×3×3)": (CONV, 3)}


def relaxed_cell(atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-3, steps=500)
    return atoms


def energy(atoms):
    image = atoms.copy()
    image.calc = calc
    return image.get_potential_energy()


t0 = time.perf_counter()
host_prim = relaxed_cell(primitive())
e_v = energy(host_prim)  # one formula unit
end = {}
for dopant in ("Nb", "Ta"):
    cell = primitive()
    cell[[a.index for a in cell if a.symbol == "V"][0]].symbol = dopant
    end[dopant] = energy(relaxed_cell(cell))
bulk_vo = np.sort(host_prim.get_all_distances(mic=True)[2, 4:])[:6].mean()


def supercell(matrix, repeat):
    atoms = make_supercell(host_prim, matrix)
    if repeat:
        atoms = atoms.repeat(repeat)
    return grouped(atoms)


results = {"host_lattice_A": float(host_prim.cell.lengths()[0] * np.sqrt(2)),
           "bulk_V_O_A": float(bulk_vo), "cells": {}}
frames = []
for name, (matrix, repeat) in CELLS.items():
    host = supercell(matrix, repeat)
    n_fu = sum(1 for a in host if a.symbol == "V")
    for dopant in ("Nb", "Ta"):
        doped = host.copy()
        site = [a.index for a in doped if a.symbol == "V"][0]
        doped[site].symbol = dopant
        doped = grouped(doped)
        site = [a.index for a in doped if a.symbol == dopant][0]
        start = doped.positions.copy()
        doped.calc = calc
        FIRE(doped, logfile=None).run(fmax=0.005, steps=2000)
        e_sub = doped.get_potential_energy() - n_fu * e_v - (end[dopant] - e_v)
        d = doped.get_all_distances(mic=True)
        oxygen = [a.index for a in doped if a.symbol == "O"]
        vanadium = [a.index for a in doped if a.symbol == "V"]
        nearest_v = sorted(vanadium, key=lambda i: d[site, i])[:6]
        shift = np.linalg.norm(doped.positions[oxygen] - start[oxygen], axis=1)
        entry = {"atoms": len(doped), "x": 1 / n_fu, "e_sub_meV": 1000 * float(e_sub),
                 "M_O_A": float(np.sort(d[site, oxygen])[:6].mean()),
                 "near_V_O_A": float(np.mean([np.sort(d[i, oxygen])[:6].mean()
                                              for i in nearest_v])) if vanadium else None,
                 "max_O_shift_A": float(shift.max())}
        results["cells"].setdefault(name, {})[dopant] = entry
        print(f"{name:<22} {dopant}: E_sub {entry['e_sub_meV']:+7.1f} meV, M–O "
              f"{entry['M_O_A']:.3f} Å, max O shift {entry['max_O_shift_A']:.3f} Å "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        if name.startswith("80 "):
            relaxed = doped.copy()
            relaxed.calc = None
            relaxed.info = {"group": f"{dopant.lower()}_dilute80_test", "x": 1 / n_fu,
                            "dopant": dopant}
            frames.append(relaxed)
            atoms = doped.copy()
            atoms.calc = calc
            MaxwellBoltzmannDistribution(atoms, temperature_K=600,
                                         rng=np.random.default_rng(7))
            Langevin(atoms, 1.0 * units.fs, temperature_K=600, friction=0.02,
                     rng=np.random.default_rng(8)).run(400)
            hot = atoms.copy()
            hot.calc = None
            hot.info = dict(relaxed.info)
            frames.append(hot)
write(WORK / "dilute_frames.extxyz", frames)
(HERE / "dilute_mace_mp0.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
(WORK / "dilute.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print(f"{len(frames)} frames -> {WORK / 'dilute_frames.extxyz'} ({time.perf_counter() - t0:.0f} s)")

COLORS = {"Nb": "#2a78d6", "Ta": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
sizes = [entry["Nb"]["atoms"] for entry in results["cells"].values()]
for dopant, color in COLORS.items():
    rows = [entry[dopant] for entry in results["cells"].values()]
    axes[0].plot(sizes, [r["e_sub_meV"] for r in rows], "o-", color=color, lw=2,
                 label=f"M = {dopant}")
    axes[1].plot(sizes, [r["M_O_A"] for r in rows], "o-", color=color, lw=2,
                 label=f"{dopant}–O")
    axes[1].plot(sizes, [r["near_V_O_A"] for r in rows], "s--", color=color, lw=1.4,
                 label=f"V–O next to {dopant}")
axes[1].axhline(bulk_vo, color="#8a8985", ls=":", lw=1.2, label="V–O, pristine")
for ax, title, ylabel in zip(axes, ("Substitution energy vs the end members",
                                    "Bonds around the dopant"),
                             ("meV per dopant", "bond to the 6 nearest O (Å)"), strict=True):
    ax.set_xscale("log")
    ax.set_xticks(sizes, labels=[str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("atoms in the cell (one dopant)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
axes[0].axhline(0, color="#c3c2b7", lw=1)
fig.suptitle("One Nb or Ta in Ba₂BiVO₆: toward the dilute limit (MACE-MP-0, host lattice fixed)",
             x=0.01, ha="left", fontsize=11)
fig.savefig(HERE / "images" / "bbvo_dilute.png", dpi=150)
print("figure written")
