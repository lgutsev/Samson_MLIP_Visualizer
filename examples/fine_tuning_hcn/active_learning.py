"""Step 2, IRC-driven active learning for HCN <-> HNC: MACE-MP-0 small fine-tuned on PBE.

Round r: fine-tune a 3-model committee on train_r{r-1}; with the committee find
the TS (P-RFO, exact Hessian) and follow the IRC; benchmark it against PBE on
held-out IRC frames (the ground truth); pick the frames with the largest
committee force spread (the cheap signal), label them with PBE (+1 rattle each),
and add them to the training set. Stop when the barrier and the path agree with
PBE within 0.05 eV.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase import Atoms
from ase.io import read, write
from common import (
    HERE,
    PBE_BARRIER,
    geo,
    label,
    mace,
    pbe,
    record,
    train,
    ts_guess,
    write_model_card,
)

from samson_mlip_visualizer.benchmark import benchmark_path, bond_angle, write_report
from samson_mlip_visualizer.engine import _committee_spread
from samson_mlip_visualizer.reaction_path import irc
from samson_mlip_visualizer.ts import prfo_search
from samson_mlip_visualizer.vibrations import harmonic_frequencies

SEEDS, EPOCHS, PICK, ROUNDS, TOL = (1, 2, 3), 120, 8, 3, 0.05
log_file = open(HERE / "loop.log", "a", encoding="utf-8")


def log(message):
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def committee(round_):
    folder = HERE / f"round{round_}"
    train_file = HERE / f"train_r{round_ - 1}.extxyz"
    models = [folder / f"hcn_r{round_}_s{seed}.model" for seed in SEEDS]
    if all(model.exists() for model in models):
        log(f"round {round_}: committee already trained")
        return [str(model) for model in models]
    configurations = len(read(train_file, ":"))
    log(f"round {round_}: training {len(SEEDS)} models on {configurations} configurations")
    start = time.perf_counter()
    with ThreadPoolExecutor(len(SEEDS)) as pool:
        jobs = [
            pool.submit(train, train_file, f"hcn_r{round_}_s{seed}", seed, folder, EPOCHS)
            for seed in SEEDS
        ]
        results = [job.result() for job in jobs]
    wall = time.perf_counter() - start
    for model, _ in results:
        write_model_card(model, configurations, round_)
    record(f"round{round_}_training_wall_s", wall)
    per_model = float(np.mean([seconds for _, seconds in results]))
    record(f"round{round_}_training_per_model_s", per_model)
    log(f"round {round_}: trained in {wall:.0f} s wall ({per_model:.0f} s per model, in parallel)")
    return [str(model) for model, _ in results]


def evaluate(round_, models, ts_start):
    start = time.perf_counter()
    calc = mace(models)
    ts = ts_start.copy()
    ts.calc = calc
    search = prfo_search(ts, fmax=1e-3, exact_hessian=True)
    frequencies = harmonic_frequencies(ts)
    path = irc(ts, step=0.05, fmax=0.005, relax_ends=True)
    ends = {}
    for name in ("reverse", "forward"):
        end = ts.copy()
        end.positions = getattr(path, f"{name}_minimum_positions")
        ends[name] = (
            "HCN" if end.get_distance(0, 2) < end.get_distance(1, 2) else "HNC",
            getattr(path, f"{name}_minimum_ev"),
        )
    e_hcn = next(e for label_, e in ends.values() if label_ == "HCN")
    e_hnc = next(e for label_, e in ends.values() if label_ == "HNC")
    frames_pos = [f.positions for f in path.frames(ts.get_positions())]
    arcs = [f.arc for f in path.frames(ts.get_positions())]
    frames = [Atoms("CNH", positions=p) for p in frames_pos]
    # The committee spread on every IRC frame: free, no reference calculation.
    sigma_f = []
    for frame in frames:
        frame.calc = calc
        frame.get_forces()
        sigma_f.append(_committee_spread(calc)[1])
    record(f"round{round_}_ts_irc_s", time.perf_counter() - start)

    # Ground truth on held-out frames: the committee vs PBE along its own IRC.
    start = time.perf_counter()
    ts_index = len(path.reverse)
    reference = pbe()
    bench = benchmark_path(
        frames,
        calc,
        reference,
        names=(f"fine-tuned MACE, round {round_}", "PBE/def2-TZVP"),
        coordinate=arcs,
        coordinate_label="IRC coordinate (Å·amu½)",
        points=15,
        keep=(ts_index,),
        descriptor=("∠H–C–N", bond_angle(frames_pos, (2, 0, 1))),
    )
    record(f"round{round_}_validation_pbe_s", time.perf_counter() - start)
    written = write_report(
        bench,
        HERE / f"round{round_}" / f"bench_r{round_}",
        mark=(0.0, "TS"),
        end_labels=("HCN", "HNC") if ends["reverse"][0] == "HCN" else ("HNC", "HCN"),
        title=f"HCN → HNC, round {round_} committee vs PBE along its IRC",
    )
    summary = bench.summary()
    row = {
        "round": round_,
        "ts_converged": bool(search.converged),
        "ts_steps": search.steps,
        "imaginary_cm": float(frequencies.wavenumbers_cm[0]),
        "n_imaginary": int(frequencies.n_imaginary),
        "ts_geometry": geo(ts),
        "irc_ends": [ends["reverse"][0], ends["forward"][0]],
        "barrier_ev": float(path.ts_energy_ev - e_hcn),
        "reaction_ev": float(e_hnc - e_hcn),
        "barrier_error_vs_pbe_ev": float(path.ts_energy_ev - e_hcn - PBE_BARRIER),
        "path_energy_error_max_abs_ev": float(np.abs(bench.energy_error).max()),
        "path_force_error_mean_ev_per_A": summary["force_error_mean_ev_per_angstrom"],
        "path_force_error_max_ev_per_A": summary["force_error_max_ev_per_angstrom"],
        "committee_force_sigma_max_ev_per_A": float(max(sigma_f)),
        "committee_energy_sigma_max_ev": summary.get("committee_energy_std_max_ev"),
        "figures": [str(p) for p in written.values()],
    }
    log(json.dumps(row, indent=1))
    return row, ts, frames, np.array(sigma_f), set(bench.frame_indices)


def select_and_label(round_, frames, sigma_f, held_out):
    """The PICK frames with the largest committee force spread (not held out, at
    least 3 frames apart), each with one rattled copy, labeled with PBE."""
    start = time.perf_counter()
    order = np.argsort(sigma_f)[::-1]
    chosen = []
    for k in order:
        if k in held_out or any(abs(k - c) < 3 for c in chosen):
            continue
        chosen.append(int(k))
        if len(chosen) == PICK:
            break
    rng = np.random.default_rng(round_)
    picked = [frames[k].copy() for k in chosen]
    rattled = [Atoms("CNH", positions=f.positions + rng.normal(0, 0.03, (3, 3))) for f in picked]
    new = label(picked, pbe(), f"irc{round_}")
    new += label(rattled, pbe(), f"rattle{round_}")
    data = read(HERE / f"train_r{round_ - 1}.extxyz", ":") + new
    write(HERE / f"train_r{round_}.extxyz", data)
    record(f"round{round_}_labeling_s", time.perf_counter() - start)
    log(
        f"round {round_}: labeled {len(new)} new configurations (σ_F of picked frames "
        f"{', '.join(f'{sigma_f[k]:.3f}' for k in chosen)} eV/Å); training set now {len(data)}"
    )


def main():
    total = time.perf_counter()
    ts = ts_guess()
    rows = []
    for round_ in range(1, ROUNDS + 1):
        models = committee(round_)
        row, ts, frames, sigma_f, held_out = evaluate(round_, models, ts)
        rows.append(row)
        (HERE / "rounds.json").write_text(json.dumps(rows, indent=1))
        if abs(row["barrier_error_vs_pbe_ev"]) < TOL and row["path_energy_error_max_abs_ev"] < TOL:
            log(f"round {round_}: converged against PBE; stopping")
            break
        if round_ < ROUNDS:
            select_and_label(round_, frames, sigma_f, held_out)
    record("active_learning_total_s", time.perf_counter() - total)
    log(f"total {time.perf_counter() - total:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
