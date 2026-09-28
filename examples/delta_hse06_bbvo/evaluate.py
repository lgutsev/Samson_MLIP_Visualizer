"""Step 4 (laptop, after training): errors against HSE06, the functional-vs-
baseline split of the residual, and the lattice constant.  -> <run>/results.json

- held-out ``md600`` frames: energy (per atom, relative to the set's mean),
  force, and stress errors against HSE06 of MACE-MP-0 alone, MACE-MP-0 + Δ
  (3-seed committee), and MACE-MP-0 fine-tuned on HSE06 (3-seed committee);
- how much of the force residual HSE06 − MACE-MP-0 is the functional
  (HSE06 − PBE+U) and how much is the baseline's own error (PBE+U − MACE-MP-0);
- the clamped-ion lattice constant: the minimum of a quadratic fit of E against
  the lattice scale over the rigidly scaled primitive cells (the ions stay at the
  PBE+U fractional positions), from the HSE06 labels and from each model on the
  same frames. Single points cannot give the relaxed HSE06 lattice (the ions
  relax with volume, which moved the minimum by 0.03 Å in the dry run); each
  model's fully relaxed lattice is reported too, without a reference;
- with the doped series: errors per held-out composition (``<m><x>_test``), and
  the mixing energy ΔE_mix(x) = E(x) − (1 − x) E(pristine) − x E(end member) per
  formula unit, from the HSE06 labels of the relaxed cells (``pristine_min``,
  ``<m><x>_min``, MACE-MP-0 geometries) and from each model on the same cells.

Usage: ``evaluate.py [--dry-run]``.
"""

import json
import sys

import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.units import GPa
from common import (
    A_PRIM,
    conventional,
    held_out,
    labels,
    mace_mp0,
    rigid_scaled_cells,
    run_dir,
)

from samson_mlip_visualizer.calculators import create_calculator

DRY = "--dry-run" in sys.argv
run = run_dir(DRY)
labeled = labels(DRY)
test = [f for f in labeled if f.info["group"] == "md600"]
sets = {"pristine, 600 K": test}
for frame in labeled:
    if held_out(frame) and frame.info["group"] != "md600":
        sets.setdefault(f"{frame.info['dopant']} x={frame.info['x']:g}", []).append(frame)
minima = {f.info["group"]: f for f in labeled if f.info["group"].endswith("_min")}
DOPED = sorted({f.info["dopant"] for f in minima.values() if f.info["dopant"] != "none"})


def mixing(energy_of):
    """ΔE_mix per formula unit (meV) for each dopant, from ``energy_of(frame)`` (eV)."""
    out = {}
    for dopant in DOPED:
        tag = dopant.lower()
        points = {0.0: minima["pristine_min"], **{
            f.info["x"]: f for g, f in minima.items() if g.startswith(tag)}}
        if 1.0 not in points:
            continue
        e = {x: energy_of(frame) / 4 for x, frame in points.items()}
        out[dopant] = {f"{x:g}": 1000 * (e[x] - (1 - x) * e[0.0] - x * e[1.0])
                       for x in sorted(e)}
    return out
models = {
    "MACE-MP-0": mace_mp0,
    "MACE-MP-0 + Δ": lambda: create_calculator(
        "mace", sorted((run / "models" / "mace-mp0+delta").glob("*.model")), device="cuda"),
    "MACE-MP-0 fine-tuned on HSE06": lambda: create_calculator(
        "mace", sorted((run / "models" / "direct").glob("*.model")), device="cuda"),
}


def errors(calc, frames, prefix="HSE06"):
    e, f_err, s_err = [], [], []
    for frame in frames:
        atoms = frame.copy()
        atoms.calc = calc
        e.append(atoms.get_potential_energy() / len(atoms))
        f_err.append(atoms.get_forces() - frame.arrays[f"{prefix}_forces"])
        s_err.append(atoms.get_stress() - frame.info[f"{prefix}_stress"])
    ref = np.array([fr.info[f"{prefix}_energy"] / len(fr) for fr in frames])
    e = np.array(e)
    de = (e - e.mean()) - (ref - ref.mean())
    return {"energy_mae_mev_per_atom": float(1000 * np.abs(de).mean()),
            "force_rmse_ev_A": float(np.sqrt(np.mean(np.concatenate(f_err) ** 2))),
            "stress_rmse_gpa": float(np.sqrt(np.mean(np.square(s_err))) / GPa)}


results = {"labels": str(run / "labeled.extxyz"), "synthetic": DRY, "test_frames": len(test)}
split = {}
for key, (a, b) in {"HSE06 − PBE+U (functional)": ("HSE06", "PBEU"),
                    "HSE06 − MACE-MP-0 (the residual learned)": ("HSE06", None)}.items():
    if b is None:
        base = mace_mp0()
        diffs = []
        for frame in test:
            atoms = frame.copy()
            atoms.calc = base
            diffs.append(frame.arrays[f"{a}_forces"] - atoms.get_forces())
    else:
        diffs = [frame.arrays[f"{a}_forces"] - frame.arrays[f"{b}_forces"] for frame in test]
    split[key] = float(np.sqrt(np.mean(np.concatenate(diffs) ** 2)))
results["force_residual_rms_ev_A"] = split

# HSE06 lattice from the labels: E(scale) of the unrattled isotropic strain frames
iso = rigid_scaled_cells(labeled)
scales = np.array([s for s, _ in iso])


def clamped_lattice(energies):
    """Minimum of a quadratic E(scale) through the rigidly scaled cells (Å)."""
    fit = np.polyfit(scales, energies, 2)
    return float(2 * A_PRIM * -fit[1] / (2 * fit[0]))


if len(iso) >= 3:
    results["hse06_clamped_lattice_A"] = clamped_lattice([f.info["HSE06_energy"] for _, f in iso])
if "pristine_min" in minima:
    results["hse06_mixing_meV_per_fu"] = mixing(lambda f: f.info["HSE06_energy"])
    results["pbeu_mixing_meV_per_fu"] = mixing(lambda f: f.info["PBEU_energy"])
results["models"] = {}
for name, build in models.items():
    calc = build()
    entry = errors(calc, test)
    entry["sets"] = {key: errors(calc, frames) for key, frames in sets.items()}
    if "pristine_min" in minima:
        def model_energy(frame, calc=calc):
            atoms = frame.copy()
            atoms.calc = calc
            return atoms.get_potential_energy()

        entry["mixing_meV_per_fu"] = mixing(model_energy)
    cell = conventional()
    cell.calc = calc
    BFGS(FrechetCellFilter(cell, hydrostatic_strain=True), logfile=None).run(fmax=1e-3,
                                                                             steps=200)
    entry["relaxed_lattice_A"] = float(np.mean(cell.cell.lengths()))
    if len(iso) >= 3:
        energies = []
        for _, frame in iso:
            atoms = frame.copy()
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
        entry["clamped_lattice_A"] = clamped_lattice(energies)
    results["models"][name] = entry
    clamped = ""
    if "clamped_lattice_A" in entry:
        clamped = f"  clamped a {entry['clamped_lattice_A']:.4f} Å"
    print(f"{name:<32} E {entry['energy_mae_mev_per_atom']:6.2f} meV/atom  "
          f"F {entry['force_rmse_ev_A']:.3f} eV/Å  S {entry['stress_rmse_gpa']:.2f} GPa  "
          f"relaxed a {entry['relaxed_lattice_A']:.4f} Å{clamped}", flush=True)
    for key, value in entry["sets"].items():
        if key != "pristine, 600 K":
            print(f"    {key:<10} E {value['energy_mae_mev_per_atom']:6.2f} meV/atom  "
                  f"F {value['force_rmse_ev_A']:.3f} eV/Å  S {value['stress_rmse_gpa']:.2f} GPa")
    for dopant, values in entry.get("mixing_meV_per_fu", {}).items():
        print(f"    mixing {dopant}: " + ", ".join(f"x={x} {v:+.1f}" for x, v in values.items())
              + " meV/f.u.")
print("force residual RMS:", {k: round(v, 3) for k, v in split.items()})
for level in ("hse06", "pbeu"):
    for dopant, values in results.get(f"{level}_mixing_meV_per_fu", {}).items():
        print(f"{level.upper()} labels, mixing {dopant}: "
              + ", ".join(f"x={x} {v:+.1f}" for x, v in values.items()) + " meV/f.u.")
if "hse06_clamped_lattice_A" in results:
    print(f"HSE06 clamped-ion lattice from the labels: {results['hse06_clamped_lattice_A']:.4f} Å")
(run / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
