"""Evaluate the fine-tuned AIMNet2 like the fine-tuned MACE, plus what only it can do.

    python evaluate_aimnet2_tuned.py

Run with SAMSON's Python after ``finetune_aimnet2.py``. The tuned model runs
through the AIMNet2 backend (``aimnet_model`` = the tuned ``.pt``). Sections,
each saved to ``finetune_aimnet2/evaluation.json`` and skipped when done:

1. ``ts``: its own P-RFO TS (exact Hessian), frequencies, and IRC.
2. ``same_frame``: vs ωB97X-D (cached) on the fine-tuned MACE's IRC and the
   r(C-F) scan frames (both already labeled), and on its own IRC.
3. ``asymptotes``: stationary points relative to separated F- + CH3Cl, with
   the fragments relaxed at their own charges: the check MACE cannot do.
4. ``forgetting``: neutral CH3F, CH3Cl, CH2F2 relaxed with the tuned and the
   stock AIMNet2 (compare ``forgetting/results.json`` for ωB97X-D).
"""

import json

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from forgetting import MOLECULES, bonds, build
from sn2_common import (
    AIMNET_MODEL,
    AIMNET_PYTHON,
    CL,
    KCAL,
    REFERENCE,
    WORK,
    C,
    F,
    reference_cache,
    training_set,
)

from samson_mlip_visualizer.aimnet2_backend import AIMNet2Calculator
from samson_mlip_visualizer.benchmark import _COORDINATES, benchmark_path, write_report
from samson_mlip_visualizer.finetune import distances_to
from samson_mlip_visualizer.reaction_path import irc
from samson_mlip_visualizer.ts import prfo_search
from samson_mlip_visualizer.vibrations import harmonic_frequencies

TUNED_MODEL = (r"D:\MLIP_Work_Folder\cache\aimnet\finetuned"
               r"\aimnet2_wb97m_d3_0_SN2-F-CH3Cl_wB97XD-def2TZVPD.pt")
OUT = WORK / "finetune_aimnet2"
NAME = "fine-tuned AIMNet2"


def tuned(charge=-1):
    return AIMNet2Calculator(AIMNET_PYTHON, model=TUNED_MODEL, charge=charge)


def stock(charge=-1):
    return AIMNet2Calculator(AIMNET_PYTHON, model=AIMNET_MODEL, charge=charge)


def geometry(atoms):
    return {"r_CF": float(atoms.get_distance(C, F)), "r_CCl": float(atoms.get_distance(C, CL))}


def section_ts(calc):
    path = read(WORK / "aimnet2_irc.extxyz", ":")
    ts = Atoms(path[0].numbers, path[int(np.argmax([f.info["energy_ev"] for f in path]))].positions)
    ts.calc = calc
    search = prfo_search(ts, fmax=0.005, exact_hessian=True)
    frequencies = harmonic_frequencies(ts)
    result = irc(ts, step=0.05, max_steps=400, fmax=0.01, relax_ends=True)
    frames = result.frames(ts.get_positions())
    write(OUT / "tuned_irc.extxyz", [Atoms(ts.numbers, f.positions, info={"irc_arc": f.arc})
                                     for f in frames])
    ends = {}
    for side in ("reverse", "forward"):
        end = Atoms(ts.numbers, getattr(result, f"{side}_minimum_positions"))
        ends[side] = {**geometry(end), "energy_ev": float(getattr(result, f"{side}_minimum_ev")),
                      "positions": end.positions.tolist()}
    reactant = max(ends.values(), key=lambda e: e["r_CF"])
    product = min(ends.values(), key=lambda e: e["r_CF"])
    return {
        "converged": bool(search.converged), **geometry(ts),
        "energy_ev": float(ts.get_potential_energy()), "positions": ts.positions.tolist(),
        "imaginary_cm": float(frequencies.wavenumbers_cm[0]),
        "n_imaginary": int(frequencies.n_imaginary),
        "irc_ends": ends,
        "barrier_kcal": (float(ts.get_potential_energy()) - reactant["energy_ev"]) * KCAL,
        "complex_to_complex_kcal": (product["energy_ev"] - reactant["energy_ev"]) * KCAL,
    }


def section_same_frame(calc, reference):
    train = training_set()
    paths = {
        "mace_tuned_irc": (read(WORK / "finetune" / "finetuned_irc.extxyz", ":"), "irc_arc", 30),
        "own_irc": (read(OUT / "tuned_irc.extxyz", ":"), "irc_arc", 30),
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
        write_report(bench, OUT / f"{key}_vs_wb97xd",
                     mark=(0.0, "TS") if coordinate_key == "irc_arc" else None,
                     end_labels=ends if coordinate_key == "irc_arc" else ("2.8 Å", "1.4 Å"),
                     title=f"{NAME} vs {REFERENCE}: {key.replace('_', ' ')}")
        summary = bench.summary()
        rmsd = distances_to([frames[k] for k in bench.frame_indices], train)
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
            "rmsd_to_training_A": {"median": float(np.median(rmsd)), "max": float(np.max(rmsd))},
            "new_reference_calculations": reference.computed - before,
        }
    return out


def relaxed_energy(atoms, calc, fmax=0.01):
    atoms = atoms.copy()
    atoms.calc = calc
    if len(atoms) > 1:
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=300)
    return float(atoms.get_potential_energy()), atoms


def section_asymptotes(ts_section):
    """Relative to separated F- + CH3Cl, kcal/mol, the model at its own geometries."""
    anion, neutral = tuned(-1), tuned(0)
    fragments = {}
    for name, substituents, calc in (("CH3Cl", MOLECULES["CH3Cl"], neutral),
                                     ("CH3F", MOLECULES["CH3F"], neutral)):
        fragments[name], _ = relaxed_energy(build(substituents), calc)
    for name, symbol in (("F-", "F"), ("Cl-", "Cl")):
        fragments[name], _ = relaxed_energy(Atoms(symbol), anion)
    ends = ts_section["irc_ends"].values()
    reactant = max(ends, key=lambda e: e["r_CF"])["energy_ev"]
    product = min(ends, key=lambda e: e["r_CF"])["energy_ev"]
    zero = fragments["CH3Cl"] + fragments["F-"]
    return {
        "reactant_complex": (reactant - zero) * KCAL,
        "ts": (ts_section["energy_ev"] - zero) * KCAL,
        "product_complex": (product - zero) * KCAL,
        "products": (fragments["CH3F"] + fragments["Cl-"] - zero) * KCAL,
        "fragments_ev": fragments,
    }


def section_forgetting():
    out = {}
    for label, factory in ((NAME, tuned), ("AIMNet2", stock)):
        calc = factory(0)
        for name, substituents in MOLECULES.items():
            _, atoms = relaxed_energy(build(substituents), calc, fmax=0.005)
            out.setdefault(label, {})[name] = bonds(atoms)
    return out


def main():
    results_file = OUT / "evaluation.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}

    def save():
        results_file.write_text(json.dumps(results, indent=1))

    calc = tuned(-1)
    if "ts" not in results:
        results["ts"] = section_ts(calc)
        save()
    if "same_frame" not in results:
        results["same_frame"] = section_same_frame(calc, reference_cache())
        save()
    if "asymptotes" not in results:
        results["asymptotes"] = section_asymptotes(results["ts"])
        save()
    if "forgetting" not in results:
        results["forgetting"] = section_forgetting()
        save()
    brief = {k: v for k, v in results["ts"].items() if k not in ("positions", "irc_ends")}
    print(json.dumps({"ts": brief, "same_frame": results["same_frame"],
                      "asymptotes": {k: v for k, v in results["asymptotes"].items()
                                     if k != "fragments_ev"},
                      "forgetting": results["forgetting"]}, indent=1))


if __name__ == "__main__":
    main()
