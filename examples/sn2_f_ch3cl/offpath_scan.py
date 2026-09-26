"""Off-path test: a C-F distance scan with the fine-tuned committee, checked by ωB97X-D.

    python offpath_scan.py

Like HCN's r(N-H) scan: from the committee's reactant complex, r(C-F) goes
from 2.8 to 1.4 Å in 15 constrained relaxations (everything else free). The
frames leave the IRC, so they test the model away from its training data.
ωB97X-D (cached) on every frame; each frame's aligned RMSD to the nearest
training structure; geometry jumps flagged. Writes ``scan/`` (frames, report,
``results.json``); the scan is reused when the script is started again.
"""

import json

import numpy as np
from ase import Atoms
from ase.io import read, write
from sn2_common import CL, REFERENCE, WORK, C, F, committee, reference_cache, training_set

from samson_mlip_visualizer.benchmark import _COORDINATES, benchmark_path, write_report
from samson_mlip_visualizer.finetune import distances_to
from samson_mlip_visualizer.reaction_path import scan_jumps, scan_to_ts

OUT = WORK / "scan"
START, STOP, POINTS = 2.8, 1.4, 15


def main():
    OUT.mkdir(exist_ok=True)
    frames_file = OUT / "scan_frames.extxyz"
    calc = committee()
    if frames_file.exists():
        frames = read(frames_file, ":")
        distances = [f.info["scan_distance"] for f in frames]
    else:
        irc = read(WORK / "finetune" / "finetuned_irc.extxyz", ":")
        end = max((irc[0], irc[-1]), key=lambda a: a.get_distance(C, F))  # F-...CH3Cl
        atoms = Atoms(end.numbers, end.positions)
        atoms.calc = calc
        scan = scan_to_ts(atoms, (C, F), start=START, stop=STOP, points=POINTS,
                          relax_fmax=0.01, refine=False)
        distances = scan.distances
        frames = [Atoms(end.numbers, p, info={"scan_distance": d})
                  for p, d in zip(scan.frames, distances, strict=True)]
        write(frames_file, frames)
    jumps = scan_jumps([f.positions for f in frames], distances)
    reference = reference_cache()
    bench = benchmark_path(
        frames, calc, reference, names=("fine-tuned MACE", REFERENCE), coordinate=distances,
        coordinate_label=_COORDINATES["scan_distance"],
    )
    write_report(bench, OUT / "scan_tuned_vs_wb97xd", end_labels=("r(C–F) 2.8 Å", "1.4 Å"),
                 title=f"Fine-tuned MACE vs {REFERENCE}: r(C–F) scan off the IRC")
    rmsd = distances_to(frames, training_set())
    error = bench.energy_error
    rows = [
        {"r_CF": float(d), "r_CCl": float(f.get_distance(C, CL)),
         "energy_error_ev": float(e), "worst_atom_force_error_ev_per_A": float(w),
         "rmsd_to_training_A": float(r), "committee_energy_std_ev": float(s)}
        for d, f, e, w, r, s in zip(distances, frames, error, bench.force_error_max, rmsd,
                                    bench.committee_energy_std, strict=True)
    ]
    summary = bench.summary()
    result = {
        "points": len(frames), "jumps": jumps,
        "energy_error_max_abs_ev": float(np.abs(error).max()),
        "energy_rmse_ev": summary["energy_error_rmse_ev"],
        "force_mae_ev_per_A": summary["force_mae_ev_per_angstrom"],
        "force_rmse_ev_per_A": summary["force_rmse_ev_per_angstrom"],
        "worst_atom_force_error_ev_per_A": summary["force_error_max_ev_per_angstrom"],
        "rmsd_to_training_A": {"median": float(np.median(rmsd)), "max": float(np.max(rmsd))},
        "committee_energy_std_max_ev": summary["committee_energy_std_max_ev"],
        "error_vs_rmsd_correlation": float(np.corrcoef(np.abs(error), rmsd)[0, 1]),
        "frames": rows,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=1))
    print("r(C-F)  r(C-Cl)  dE (meV)  worst F (eV/A)  RMSD (A)  sigma_E (meV)")
    for row in rows:
        print(f"{row['r_CF']:.2f}    {row['r_CCl']:.2f}    {1000 * row['energy_error_ev']:7.1f}"
              f"   {row['worst_atom_force_error_ev_per_A']:.3f}           "
              f"{row['rmsd_to_training_A']:.3f}     {1000 * row['committee_energy_std_ev']:.1f}")


if __name__ == "__main__":
    main()
