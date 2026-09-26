"""Step 3, every HCN test with a model (or committee), plus held-out and forgetting
checks. Usage: tests.py LABEL MODEL [MODEL ...]   (MODEL 'pbe' for Psi4 PBE).

Held-out frames: the model's own IRC (30 frames) and its r(N-H) scan (13 frames),
each with its aligned RMSD to the nearest training structure."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read
from common import HERE, geo, mace, pbe, ts_guess

from samson_mlip_visualizer.benchmark import benchmark_path, bond_angle, write_report
from samson_mlip_visualizer.compat import supported_species
from samson_mlip_visualizer.engine import relax
from samson_mlip_visualizer.reaction_path import aligned_rmsd, irc, scan_to_ts
from samson_mlip_visualizer.ts import prfo_search
from samson_mlip_visualizer.vibrations import harmonic_frequencies

label, models = sys.argv[1], sys.argv[2:]
is_pbe = models == ["pbe"]
calc = pbe() if is_pbe else mace(models)
tag = label.replace(" ", "_").replace("(", "").replace(")", "")
(HERE / "final").mkdir(exist_ok=True)
newest = max(HERE.glob("train_r*.extxyz"), key=lambda path: int(path.stem.split("_r")[1]))
training = [a.positions for a in read(newest, ":")]
out = {"label": label}
start_all = time.perf_counter()


def distance_to_training(frames):
    """Aligned RMSD (Å) from each frame to its nearest training structure."""
    return [min(aligned_rmsd(f.positions, t) for t in training) for f in frames]


def benchmark(frames, coordinate, label_x, name, keep=(), mark=None, ends=None, points=None):
    bench = benchmark_path(
        frames,
        calc,
        pbe(),
        names=(label, "PBE/def2-TZVP"),
        coordinate=coordinate,
        coordinate_label=label_x,
        points=points,
        keep=keep,
        descriptor=("∠H–C–N", bond_angle([f.positions for f in frames], (2, 0, 1))),
    )
    write_report(
        bench,
        HERE / "final" / f"{name}_{tag}",
        mark=mark,
        end_labels=ends,
        title=f"HCN → HNC {name.replace('_', ' ')}: {label} vs PBE",
    )
    summary = bench.summary()
    evaluated = [frames[k] for k in bench.frame_indices]
    distances = distance_to_training(evaluated)
    error = np.abs(bench.energy_error)
    summary.update(
        {
            "nearest_training_rmsd_median_A": float(np.median(distances)),
            "nearest_training_rmsd_max_A": float(np.max(distances)),
        }
    )
    if bench.committee_energy_std is not None:
        summary["committee_sigma_vs_error_correlation"] = (
            float(np.corrcoef(bench.committee_energy_std, error)[0, 1]) if error.std() > 0 else None
        )
        summary["per_frame"] = [
            {
                "x": round(float(x), 3),
                "abs_error_ev": round(float(e), 4),
                "committee_sigma_ev": round(float(s), 4),
                "rmsd_to_training_A": round(float(d), 3),
            }
            for x, e, s, d in zip(
                bench.coordinate, error, bench.committee_energy_std, distances, strict=True
            )
        ]
    return summary


# 1. minima and their frequencies
hcn = Atoms("CNH", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0.05, -1.07]])
hcn.calc = calc
relax(hcn, fmax=1e-3, optimizer="LBFGS", max_steps=1000)
hnc = Atoms("CNH", positions=[[0, 0, 0], [0, 0, 1.17], [0.05, 0, 2.19]])
hnc.calc = calc
relax(hnc, fmax=1e-3, optimizer="LBFGS", max_steps=1000)
for name, atoms in (("HCN", hcn), ("HNC", hnc)):
    freqs = harmonic_frequencies(atoms).wavenumbers_cm
    out[name] = {"geometry": geo(atoms), "frequencies_cm": [round(float(v)) for v in freqs]}
e_hcn, e_hnc = hcn.get_potential_energy(), hnc.get_potential_energy()

# 2. P-RFO TS from the MACE-MP-0 geometry, exact Hessian, frequencies
ts = ts_guess()
ts.calc = calc
search = prfo_search(ts, fmax=1e-3, exact_hessian=True)
freqs = harmonic_frequencies(ts)
out["TS"] = {
    "converged": bool(search.converged),
    "steps": search.steps,
    "geometry": geo(ts),
    "frequencies_cm": [round(float(v)) for v in freqs.wavenumbers_cm],
    "n_imaginary": int(freqs.n_imaginary),
}
out["barrier_ev"] = float(ts.get_potential_energy() - e_hcn)
out["reaction_ev"] = float(e_hnc - e_hcn)

if not is_pbe:
    # 3. IRC connectivity
    path = irc(ts, step=0.05, fmax=0.005, relax_ends=True)
    ends = []
    for name in ("reverse", "forward"):
        end = ts.copy()
        end.positions = getattr(path, f"{name}_minimum_positions")
        ends.append(
            {
                "is": "HCN" if end.get_distance(0, 2) < end.get_distance(1, 2) else "HNC",
                "geometry": geo(end),
            }
        )
    out["irc"] = {"ends": ends, "frames": len(path.forward) + len(path.reverse) + 1}

    # 4. held-out benchmark along this model's own IRC (30 frames)
    frames = path.frames(ts.get_positions())
    positions, arcs = [f.positions for f in frames], [f.arc for f in frames]
    if ends[0]["is"] != "HCN":
        positions, arcs = positions[::-1], [-a for a in arcs[::-1]]
    irc_frames = [Atoms("CNH", positions=p) for p in positions]
    out["benchmark_irc"] = benchmark(
        irc_frames,
        arcs,
        "IRC coordinate (Å·amu½)",
        "along_the_IRC",
        points=30,
        keep=(int(np.argmin(np.abs(arcs))),),
        mark=(0.0, "TS"),
        ends=("HCN", "HNC"),
    )

    # 5. the r(N-H) scan: does it still jump, and how good is the model on its frames
    #    (the squeezed linear branch is far outside the training data)?
    start = hcn.copy()
    start.calc = calc
    start.positions[2, 1] += 0.4
    scan = scan_to_ts(start, (2, 1), stop=1.0, points=13, exact_hessian=True, ts_fmax=1e-3)
    out["scan"] = {
        "jumps": scan.jumps,
        "bracketed": scan.bracketed,
        "ts_converged": bool(scan.ts.converged),
        "ts_geometry": geo(start),
        "ts_energy_ev": float(start.get_potential_energy() - e_hcn),
    }
    scan_frames = [Atoms("CNH", positions=p) for p in scan.frames]
    out["benchmark_scan"] = benchmark(scan_frames, scan.distances, "scan r(N–H) (Å)", "scan_frames")

# 6. forgetting: element coverage, and H/C/N chemistry it was not trained on
species = supported_species(calc)
out["elements"] = (
    "all (basis set)"
    if is_pbe
    else (
        sorted(species) if species is not None and len(species) < 20 else f"{len(species)} elements"
    )
)
nh3_ts = read(Path(__file__).parent.parent / "nh3_ts_guess.xyz")
nh3_ts.calc = calc
prfo_search(nh3_ts, fmax=1e-3, exact_hessian=True)
nh3 = molecule("NH3")
nh3.calc = calc
relax(nh3, fmax=1e-3, optimizer="LBFGS", max_steps=500)
barrier = nh3_ts.get_potential_energy() - nh3.get_potential_energy()
out["nh3_inversion_barrier_ev"] = round(float(barrier), 4)
out["nh3_r_NH"] = round(float(nh3.get_distance(0, 1)), 4)
ch4 = molecule("CH4")
ch4.calc = calc
relax(ch4, fmax=1e-3, optimizer="LBFGS", max_steps=500)
out["ch4_r_CH"] = round(float(ch4.get_distance(0, 1)), 4)
c2h2 = molecule("C2H2")
c2h2.calc = calc
relax(c2h2, fmax=1e-3, optimizer="LBFGS", max_steps=500)
out["c2h2_r_CC"] = round(float(c2h2.get_distance(0, 1)), 4)
out["c2h2_r_CH"] = round(float(min(c2h2.get_distance(0, 2), c2h2.get_distance(0, 3))), 4)
if species is None or "O" in species:
    water = molecule("H2O")
    water.calc = calc
    relax(water, fmax=1e-3, optimizer="LBFGS", max_steps=500)
    out["water"] = {
        "r_OH": round(float(water.get_distance(0, 1)), 4),
        "HOH": round(float(water.get_angle(1, 0, 2)), 2),
    }
else:
    out["water"] = "not supported: the model has no oxygen"
out["seconds"] = round(time.perf_counter() - start_all)
(HERE / "final" / f"tests_{tag}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if not k.startswith("benchmark")}, indent=1))
