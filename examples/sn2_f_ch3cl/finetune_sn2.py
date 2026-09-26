"""Fine-tune MACE-MP-0 small to ωB97X-D/def2-TZVPD for F- + CH3Cl -> CH3F + Cl-.

Run with SAMSON's Python (mace-torch, CUDA torch), Psi4 in its own environment:

    python finetune_sn2.py [work folder]

1. Seed data: 31 frames of the AIMNet2 IRC (``aimnet2_irc.extxyz``) spread by
   arc length (30 requested, plus the TS), and two rattled copies of each
   (σ 0.04 Å), 93 configurations, labeled with
   ωB97X-D/def2-TZVPD at charge -1. MACE-MP-0 itself has no barrier here, so
   its own path cannot seed the data; AIMNet2's geometries match the literature.
2. A committee of three plain fine-tunes (seeds 1-3, in parallel on the GPU).
3. The committee's own TS (P-RFO, exact Hessian), frequencies, and IRC, and
   ωB97X-D on 15 frames of that IRC that were not trained on.

Every configuration is the same anion, so MACE's lack of a charge input does
not matter inside this data; the tuned model is a specialist for this path and
cannot evaluate separated ions. Each step is skipped when its output exists.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from samson_mlip_visualizer.benchmark import benchmark_path, select_frames, write_report
from samson_mlip_visualizer.finetune import distances_to
from samson_mlip_visualizer.paths import foundation_model
from samson_mlip_visualizer.psi4_backend import Psi4Calculator, find_psi4
from samson_mlip_visualizer.reaction_path import irc
from samson_mlip_visualizer.training import TrainingSpec, install_models, train_local
from samson_mlip_visualizer.ts import prfo_search
from samson_mlip_visualizer.vibrations import harmonic_frequencies

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
OUT = WORK / "finetune"
NAME = "SN2-F-CH3Cl_wB97XD-def2TZVPD_from-MACE-MP-0-small"
REFERENCE = "ωB97X-D/def2-TZVPD"
C, CL, F = 0, 4, 5  # atom order: C H H H Cl F


def reference():
    return Psi4Calculator(find_psi4(), method="wb97x-d", basis="def2-tzvpd", charge=-1,
                          threads=8, memory_mb=1900)


def mace(paths):
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=paths, device="cuda", default_dtype="float64")


def timed(times, stage, start):
    times[stage] = round(time.perf_counter() - start, 1)
    (OUT / "times.json").write_text(json.dumps(times, indent=1))


def seed_data(times):
    train_file = OUT / "train.extxyz"
    if train_file.exists():
        return train_file
    start = time.perf_counter()
    path = read(WORK / "aimnet2_irc.extxyz", index=":")
    arcs = [frame.info["irc_arc"] for frame in path]
    ts = int(np.argmin(np.abs(arcs)))
    picked = select_frames(len(path), 30, keep=(ts,), coordinate=arcs)
    seed = [Atoms(path[k].numbers, path[k].positions) for k in picked]
    rng = np.random.default_rng(0)
    rattled = [Atoms(a.numbers, a.positions + rng.normal(0, 0.04, a.positions.shape))
               for a in seed for _ in range(2)]
    calc, foundation = reference(), mace(str(foundation_model()))
    # Psi4's all-electron energies and the foundation model's differ by a constant;
    # one composition, so a single shift keeps every relative energy.
    first = seed[0].copy()
    first.calc = foundation
    e_mace = first.get_potential_energy()
    data = []
    for atoms, tag in [(a, "irc") for a in seed] + [(a, "rattle") for a in rattled]:
        atoms = atoms.copy()
        atoms.calc = calc
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        if not data:
            shift = energy - e_mace
        image = Atoms(atoms.numbers, atoms.positions)
        image.info.update({"REF_energy": energy - shift, "raw_energy": energy, "tag": tag})
        image.arrays["REF_forces"] = forces
        data.append(image)
    write(train_file, data)
    (OUT / "shift.json").write_text(json.dumps({"shift_ev": shift}))
    timed(times, "labeling_s", start)
    return train_file


def committee(train_file, times):
    runs = OUT / "runs"
    # mace-torch also writes <name>_compiled.model next to each model.
    models = sorted(runs.glob(f"seed*/{NAME}_seed?.model"))
    if len(models) == 3:
        return [str(m) for m in models]
    spec = TrainingSpec(
        name=NAME, foundation=str(foundation_model()), train_file=str(train_file),
        card={
            "reference": f"{REFERENCE} (Psi4), charge -1; energies shifted to the foundation scale",
            "trained_on": "93 configurations of [CH3FCl]- along the AIMNet2 IRC of "
            "F- + CH3Cl -> CH3F + Cl- (31 frames + 2 rattled copies each)",
            "scope": "F- + CH3Cl SN2 (Walden) specialist at charge -1; cannot evaluate "
            "separated ions or other charges: MACE has no charge input",
        },
    )
    start = time.perf_counter()
    results = train_local(spec, runs)
    timed(times, "training_wall_s", start)
    failed = [r for r in results if not r.ok]
    if failed:
        raise RuntimeError(f"training failed, see {failed[0].log}")
    return [str(r.model) for r in results]


def evaluate(models, train_file, times):
    start = time.perf_counter()
    calc = mace(models)
    path = read(WORK / "aimnet2_irc.extxyz", index=":")
    ts = Atoms(path[0].numbers, path[int(np.argmax([f.info["energy_ev"] for f in path]))].positions)
    ts.calc = calc
    search = prfo_search(ts, fmax=1e-3, exact_hessian=True)
    frequencies = harmonic_frequencies(ts)
    result = irc(ts, step=0.05, max_steps=400, fmax=0.005, relax_ends=True)
    frames = result.frames(ts.get_positions())
    positions = [f.positions for f in frames]
    images = [Atoms(ts.numbers, p) for p in positions]
    timed(times, "committee_ts_irc_s", start)

    ends = {}
    for side in ("reverse", "forward"):
        x = getattr(result, f"{side}_minimum_positions")
        ends[side] = {"r_CF": float(np.linalg.norm(x[F] - x[C])),
                      "r_CCl": float(np.linalg.norm(x[CL] - x[C])),
                      "energy_ev": float(getattr(result, f"{side}_minimum_ev"))}
    reactant = max(ends.values(), key=lambda e: e["r_CF"])  # F far away: F-...CH3Cl
    product = min(ends.values(), key=lambda e: e["r_CF"])

    start = time.perf_counter()
    arcs = [f.arc for f in frames]
    bench = benchmark_path(
        images, calc, reference(), names=("fine-tuned MACE", REFERENCE), coordinate=arcs,
        coordinate_label="IRC coordinate (Å·amu½)", points=15,
        keep=(int(np.argmin(np.abs(arcs))),),
    )
    timed(times, "validation_s", start)
    # The IRC's reverse end (frame 0) can be either complex.
    names = ("F⁻···CH₃Cl", "FCH₃···Cl⁻")
    if ends["reverse"] is product:
        names = names[::-1]
    written = write_report(bench, OUT / "bench_finetuned_vs_wb97xd", mark=(0.0, "TS"),
                           end_labels=names,
                           title="Fine-tuned MACE vs ωB97X-D along its own IRC")
    train = read(train_file, ":")
    held_out = [images[k] for k in bench.frame_indices]
    rmsd = distances_to(held_out, train)
    row = {
        "ts_converged": bool(search.converged),
        "ts_r_CF": float(ts.get_distance(C, F)), "ts_r_CCl": float(ts.get_distance(C, CL)),
        "imaginary_cm": float(frequencies.wavenumbers_cm[0]),
        "n_imaginary": int(frequencies.n_imaginary),
        "irc_ends": ends,
        "barrier_from_reactant_complex_ev": float(result.ts_energy_ev - reactant["energy_ev"]),
        "product_complex_minus_reactant_complex_ev":
            float(product["energy_ev"] - reactant["energy_ev"]),
        **{f"path_{k}": v for k, v in bench.summary().items()},
        "held_out_rmsd_to_training_A": {"median": float(np.median(rmsd)),
                                        "max": float(np.max(rmsd))},
        "figures": [str(p) for p in written.values()],
    }
    (OUT / "result.json").write_text(json.dumps(row, indent=1))
    write(OUT / "finetuned_irc.extxyz",
          [Atoms(ts.numbers, p, info={"irc_arc": a})
           for p, a in zip(positions, arcs, strict=True)])
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    times_file = OUT / "times.json"
    times = json.loads(times_file.read_text()) if times_file.exists() else {}
    train_file = seed_data(times)
    models = committee(train_file, times)
    row = evaluate(models, train_file, times)
    install_models(OUT / "runs", name=NAME)
    print(json.dumps(row, indent=1))
    print(json.dumps(times, indent=1))


if __name__ == "__main__":
    main()
