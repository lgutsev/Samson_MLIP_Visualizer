"""Evaluate stock UMA (omol task) on F⁻ + CH₃Cl, like the AIMNet2 checks.

    python evaluate_uma.py [CHECKPOINT]

Run with SAMSON's Python (the TS search needs sella), from any folder. UMA runs through the UMA
backend in the fairchem environment (``UMA_PYTHON``, the ``mlip`` env). CHECKPOINT
defaults to ``uma-s-1p1.pt`` in ``UMA_DIR``. Results go to
``<work>/uma/<checkpoint>/evaluation.json``, one section at a time, each skipped
when done:

1. ``ts``: its own P-RFO TS (exact Hessian) from the fine-tuned AIMNet2 TS,
   frequencies, and IRC with relaxed ends.
2. ``same_frame``: against ωB97X-D on frames that are already labeled (the
   fine-tuned MACE IRC and the r(C–F) scan), so no new DFT runs.
3. ``asymptotes``: the stationary points relative to separated F⁻ + CH₃Cl, each
   fragment relaxed at its own charge.
4. ``forgetting``: C–F and C–Cl bonds of neutral CH₃F, CH₃Cl, CH₂F₂.

UMA's omol task was trained on ωB97M-V/def2-TZVPD (OMol25), so the closest
literature row is ωB97M-V, then the CCSD(T) focal point.
"""

import json
import os
import sys
from pathlib import Path

# The SN2 example's shared pieces: paths, atom order, the cached ωB97X-D labels.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sn2_f_ch3cl"))

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.calculators.calculator import CalculationFailed  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from forgetting import MOLECULES, bonds, build  # noqa: E402
from sn2_common import CL, KCAL, REFERENCE, WORK, C, F, reference_cache  # noqa: E402

from samson_mlip_visualizer.benchmark import (  # noqa: E402
    _COORDINATES,
    benchmark_path,
    write_report,
)
from samson_mlip_visualizer.reaction_path import irc  # noqa: E402
from samson_mlip_visualizer.ts import prfo_search  # noqa: E402
from samson_mlip_visualizer.uma_backend import UMACalculator  # noqa: E402
from samson_mlip_visualizer.vibrations import harmonic_frequencies  # noqa: E402

UMA_PYTHON = Path(os.environ.get("UMA_PYTHON", r"D:\MLIP_Work_Folder\envs\mlip\python.exe"))
UMA_DIR = Path(os.environ.get("UMA_DIR", r"D:\MLIP_Downloaded_Models"))
CHECKPOINT = Path(sys.argv[1]) if len(sys.argv) > 1 else UMA_DIR / "uma-s-1p1.pt"
NAME = f"UMA {CHECKPOINT.stem}"
OUT = WORK / "uma" / CHECKPOINT.stem
OUT.mkdir(parents=True, exist_ok=True)


def uma(charge=-1, multiplicity=1):
    return UMACalculator(UMA_PYTHON, model=str(CHECKPOINT), charge=charge,
                         multiplicity=multiplicity)


def geometry(atoms):
    return {"r_CF": float(atoms.get_distance(C, F)), "r_CCl": float(atoms.get_distance(C, CL))}


def section_ts(calc):
    path = read(WORK / "finetune_aimnet2" / "tuned_irc_fragments.extxyz", ":")
    start = path[int(np.argmin(np.abs([f.info["irc_arc"] for f in path])))]
    ts = Atoms(start.numbers, start.positions)
    ts.calc = calc
    search = prfo_search(ts, fmax=0.005, exact_hessian=True)
    frequencies = harmonic_frequencies(ts)
    result = irc(ts, step=0.05, max_steps=400, fmax=0.01, relax_ends=True)
    frames = result.frames(ts.get_positions())
    write(OUT / "irc.extxyz",
          [Atoms(ts.numbers, f.positions, info={"irc_arc": f.arc, "energy_ev": f.energy_ev})
           for f in frames])
    ends = {}
    for side in ("reverse", "forward"):
        end = Atoms(ts.numbers, getattr(result, f"{side}_minimum_positions"))
        ends[side] = {**geometry(end), "energy_ev": float(getattr(result, f"{side}_minimum_ev")),
                      "positions": end.positions.tolist()}
    reactant = max(ends.values(), key=lambda e: e["r_CF"])
    product = min(ends.values(), key=lambda e: e["r_CF"])
    return {
        "converged": bool(search.converged), "steps": int(search.steps), **geometry(ts),
        "energy_ev": float(ts.get_potential_energy()), "positions": ts.positions.tolist(),
        "imaginary_cm": float(frequencies.wavenumbers_cm[0]),
        "n_imaginary": int(frequencies.n_imaginary),
        "irc_ends": ends,
        "barrier_kcal": (float(ts.get_potential_energy()) - reactant["energy_ev"]) * KCAL,
        "complex_to_complex_kcal": (product["energy_ev"] - reactant["energy_ev"]) * KCAL,
    }


def section_same_frame(calc, reference):
    paths = {
        "mace_tuned_irc": (read(WORK / "finetune" / "finetuned_irc.extxyz", ":"), "irc_arc", 30),
        "scan": (read(WORK / "scan" / "scan_frames.extxyz", ":"), "scan_distance", None),
    }
    out = {}
    for key, (frames, coordinate_key, points) in paths.items():
        x = [f.info[coordinate_key] for f in frames]
        keep = (int(np.argmin(np.abs(x))),) if coordinate_key == "irc_arc" else ()
        cf = [f.get_distance(C, F) for f in (frames[0], frames[-1])]
        ends = ("F⁻···CH₃Cl", "FCH₃···Cl⁻") if cf[0] > cf[1] else ("FCH₃···Cl⁻", "F⁻···CH₃Cl")
        before = reference.computed
        bench = benchmark_path(frames, calc, reference, names=(NAME, REFERENCE), coordinate=x,
                               coordinate_label=_COORDINATES[coordinate_key], points=points,
                               keep=keep)
        if reference.computed != before:
            raise RuntimeError(f"{key}: frames missing from the ωB97X-D cache were computed")
        write_report(bench, OUT / f"{key}_vs_wb97xd",
                     mark=(0.0, "TS") if coordinate_key == "irc_arc" else None,
                     end_labels=ends if coordinate_key == "irc_arc" else ("2.8 Å", "1.4 Å"),
                     title=f"{NAME} vs {REFERENCE}: {key.replace('_', ' ')}")
        summary = bench.summary()
        start = 0 if cf[0] > cf[1] else -1
        barrier = {n: float((bench.relative(n).max() - bench.relative(n)[start]) * KCAL)
                   for n in bench.names}
        out[key] = {
            "frames": summary["frames"],
            "barrier_model_kcal": barrier[NAME], "barrier_reference_kcal": barrier[REFERENCE],
            "energy_error_max_abs_ev": float(np.abs(bench.energy_error).max()),
            "energy_rmse_ev": summary["energy_error_rmse_ev"],
            "force_mae_ev_per_A": summary["force_mae_ev_per_angstrom"],
            "force_rmse_ev_per_A": summary["force_rmse_ev_per_angstrom"],
            "worst_atom_force_error_ev_per_A": summary["force_error_max_ev_per_angstrom"],
        }
    return out


def relaxed_energy(atoms, calc, fmax=0.01):
    atoms = atoms.copy()
    atoms.calc = calc
    if len(atoms) > 1:
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=300)
    return float(atoms.get_potential_energy()), atoms


def section_asymptotes(ts_section):
    """Relative to separated F⁻ + CH₃Cl, kcal/mol, UMA at its own geometries."""
    anion, neutral = uma(-1), uma(0)
    fragments = {}
    for name in ("CH3Cl", "CH3F"):
        fragments[name], _ = relaxed_energy(build(MOLECULES[name]), neutral)
    for name, symbol in (("F-", "F"), ("Cl-", "Cl")):
        fragments[name], _ = relaxed_energy(Atoms(symbol), anion)
    ends = ts_section["irc_ends"].values()
    reactant = max(ends, key=lambda e: e["r_CF"])["energy_ev"]
    product = min(ends, key=lambda e: e["r_CF"])["energy_ev"]
    zero = fragments["CH3Cl"] + fragments["F-"]
    anion.close()
    neutral.close()
    return {
        "reactant_complex": (reactant - zero) * KCAL,
        "ts": (ts_section["energy_ev"] - zero) * KCAL,
        "product_complex": (product - zero) * KCAL,
        "products": (fragments["CH3F"] + fragments["Cl-"] - zero) * KCAL,
        "fragments_ev": fragments,
    }


def section_forgetting():
    calc, out = uma(0), {}
    for name, substituents in MOLECULES.items():
        _, atoms = relaxed_energy(build(substituents), calc, fmax=0.005)
        out[name] = bonds(atoms)
    calc.close()
    return out


def main():
    results_file = OUT / "evaluation.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}

    def save():
        results_file.write_text(json.dumps(results, indent=1))

    calc = uma(-1)
    if "ts" not in results:
        results["ts"] = section_ts(calc)
        save()
    if "same_frame" not in results:
        results["same_frame"] = section_same_frame(calc, reference_cache())
        save()
    calc.close()
    if "asymptotes" not in results:
        try:
            results["asymptotes"] = section_asymptotes(results["ts"])
            save()
        except CalculationFailed as exc:
            if "atom_refs" not in str(exc):
                raise
            # A free F⁻ or Cl⁻ is a single atom, which UMA takes from its reference table.
            print(f"asymptotes skipped: put iso_atom_elem_refs.yaml (facebook/UMA, "
                  f"references/) next to {CHECKPOINT.name} and rerun")
    if "forgetting" not in results:
        results["forgetting"] = section_forgetting()
        save()
    brief = {k: v for k, v in results["ts"].items() if k not in ("positions", "irc_ends")}
    print(json.dumps({"model": NAME, "ts": brief, "same_frame": results["same_frame"],
                      "asymptotes": {k: v for k, v in results.get("asymptotes", {}).items()
                                     if k != "fragments_ev"},
                      "forgetting": results["forgetting"]}, indent=1))


if __name__ == "__main__":
    main()
