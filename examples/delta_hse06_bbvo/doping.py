"""Step 1b (laptop): the V-site substitution series Ba₂Bi(V₁₋ₓMₓ)O₆, M = Nb, Ta.
-> doped.json, doped_frames.extxyz, images/bbvo_doping.png

In the 40-atom cubic cell the four V sites form an fcc sublattice, so each
x = 0.25, 0.5, 0.75, 1 has a single ordering (every choice of sites is
equivalent). For each composition, with MACE-MP-0:

- a full relaxation (cell and ions; the cell may lose its cubic shape);
- the lattice constant (cube root of the volume per 40 atoms) against x;
- the mean B–O bond length for each B-site species, and Bi–O;
- the mixing energy per formula unit relative to the end members,
  ΔE_mix(x) = E(x) − (1 − x) E(Ba₂BiVO₆) − x E(Ba₂BiMO₆): negative means the
  mixed crystal is favored over separate phases (at 0 K, for this ordering).

Frames for labeling (``doped_frames.extxyz``), per composition: the relaxed cell
(``<m><x>_min``), 6 MD frames at 300 and 900 K (``<m><x>_md``), and 2 held-out
frames at 600 K (``<m><x>_test``); plus the relaxed pristine cell
(``pristine_min``), so the labels give the mixing energy on the same footing.

These are MACE-MP-0 (PBE+U-level) numbers. Once the doped campaign has HSE06
labels, the Δ-model gives the same quantities at hybrid level.
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from common import HERE, WORK, conventional, grouped, mace_mp0

DOPANTS = ("Nb", "Ta")
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
calc = mace_mp0()


def substituted(dopant, x):
    cell = conventional()
    sites = [a.index for a in cell if a.symbol == "V"]
    for index in sites[: round(4 * x)]:
        cell[index].symbol = dopant
    return grouped(cell)


def relax(atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-3, steps=500)
    return atoms


def bond_lengths(atoms):
    """Mean distance from each B-site species (and Bi) to its 6 nearest O."""
    d = atoms.get_all_distances(mic=True)
    oxygen = [a.index for a in atoms if a.symbol == "O"]
    out = {}
    for species in ("V", "Nb", "Ta", "Bi"):
        sites = [a.index for a in atoms if a.symbol == species]
        if sites:
            out[species] = float(np.mean([np.sort(d[i, oxygen])[:6].mean() for i in sites]))
    return out


def md(start, temperature, count, seed, group, every=40):
    atoms = start.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(seed))
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=temperature, friction=0.02,
                   rng=np.random.default_rng(seed + 1))
    dyn.run(300)
    out = []
    for _ in range(count):
        dyn.run(every)
        frame = atoms.copy()
        frame.calc = None
        frame.info = {"group": group}
        out.append(frame)
    return out


t0 = time.perf_counter()
pristine = relax(conventional())
results = {"x": list(FRACTIONS), "series": {}}
reference = pristine.copy()
reference.calc = None
reference.info = {"group": "pristine_min", "x": 0.0, "dopant": "none"}
frames = [reference]
for dopant in DOPANTS:
    series = {"lattice_A": [], "energy_per_fu_eV": [], "bonds_A": [], "cell_lengths_A": []}
    relaxed = {}
    for k, x in enumerate(FRACTIONS):
        atoms = pristine if x == 0 else relax(substituted(dopant, x))
        relaxed[x] = atoms
        series["lattice_A"].append(float(atoms.get_volume() ** (1 / 3)))
        series["cell_lengths_A"].append([float(v) for v in atoms.cell.lengths()])
        series["energy_per_fu_eV"].append(float(atoms.get_potential_energy() / 4))
        series["bonds_A"].append(bond_lengths(atoms))
        if x == 0:
            continue  # the pristine frames are in frames.extxyz already
        tag = f"{dopant.lower()}{x:g}"
        minimum = atoms.copy()
        minimum.calc = None
        minimum.info = {"group": f"{tag}_min", "x": x, "dopant": dopant}
        frames.append(minimum)
        seed = 100 + 10 * k + (0 if dopant == "Nb" else 5)
        frames += md(atoms, 300, 3, seed, f"{tag}_md") + md(atoms, 900, 3, seed + 2, f"{tag}_md")
        frames += md(atoms, 600, 2, seed + 4, f"{tag}_test")
        print(f"{dopant} x={x:g}: a = {series['lattice_A'][-1]:.4f} Å, "
              f"bonds {series['bonds_A'][-1]}", flush=True)
    e = np.array(series["energy_per_fu_eV"])
    x = np.array(FRACTIONS)
    series["mixing_meV_per_fu"] = (1000 * (e - ((1 - x) * e[0] + x * e[-1]))).tolist()
    results["series"][dopant] = series
for frame in frames[1:]:
    frame.info.setdefault("x", float(frame.info["group"][2:].split("_")[0]))
    frame.info.setdefault("dopant", frame.info["group"][:2].capitalize())
write(WORK / "doped_frames.extxyz", frames)
results["frames"] = len(frames)
(WORK / "doped.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
(HERE / "doped_mace_mp0.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print(f"{len(frames)} frames -> {WORK / 'doped_frames.extxyz'} ({time.perf_counter() - t0:.0f} s)")

# --- figure ------------------------------------------------------------------------
COLORS = {"Nb": "#2a78d6", "Ta": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for dopant, series in results["series"].items():
    color = COLORS[dopant]
    axes[0].plot(FRACTIONS, series["lattice_A"], "o-", color=color, lw=2, label=f"M = {dopant}")
    for species, style in (("V", "o-"), (dopant, "s--"), ("Bi", "^:")):
        xs = [x for x, b in zip(FRACTIONS, series["bonds_A"], strict=True) if species in b]
        ys = [b[species] for b in series["bonds_A"] if species in b]
        if species == "Bi" or species == "V":
            label = f"{species}–O (M = {dopant})"
        else:
            label = f"{species}–O"
        axes[1].plot(xs, ys, style, color=color, lw=1.6, ms=5, label=label)
    axes[2].plot(FRACTIONS, series["mixing_meV_per_fu"], "o-", color=color, lw=2,
                 label=f"M = {dopant}")
axes[2].axhline(0, color="#c3c2b7", lw=1)
titles = ("Lattice constant (∛V of the 40-atom cell)", "Mean bond to the 6 nearest O",
          "Mixing energy vs the end members")
ylabels = ("a (Å)", "bond length (Å)", "meV per formula unit")
for ax, title, ylabel in zip(axes, titles, ylabels, strict=True):
    ax.set_title(title, loc="left")
    ax.set_xlabel("x in Ba₂Bi(V₁₋ₓMₓ)O₆")
    ax.set_ylabel(ylabel)
    ax.set_xticks(FRACTIONS)
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("V-site substitution in Ba₂BiVO₆ with MACE-MP-0 (PBE+U level, before any HSE06 "
             "correction)", x=0.01, ha="left", fontsize=11)
(HERE / "images").mkdir(exist_ok=True)
fig.savefig(HERE / "images" / "bbvo_doping.png", dpi=150)
print("figure written")
