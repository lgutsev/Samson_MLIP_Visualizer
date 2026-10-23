"""Validate the UMA presets through the bridge's own calculator, relaxation and MD.

Run with a Python that has this package (e.g. SAMSON's), pointing at the fairchem
environment::

    python scripts/uma_validate.py D:/MLIP_Work_Folder/envs/uma_nitro/Scripts/python.exe \
        --device cuda --standard D:/MLIP_Foundational_Models/UMA/uma-s-1p2p1.pt \
        --out D:/MLIP_Work_Folder/uma_nitro_validation

These are integration and numerical-consistency checks (Nitro vs high-precision on
the same checkpoint, task, dispersion and coordinates), not accuracy benchmarks:
accuracy needs DFT references at each preset's level of theory. Timings are for this
GPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from samson_mlip_visualizer import uma_backend
from samson_mlip_visualizer.engine import relax
from samson_mlip_visualizer.md import run_md
from samson_mlip_visualizer.uma_backend import UMACalculator
from samson_mlip_visualizer.uma_config import UMAConfig, UMAConfigError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def structures() -> dict[str, Atoms]:
    molecule = read(EXAMPLES / "nh3_ts_guess.xyz")
    molecule.pbc, molecule.cell = False, None
    box = read(EXAMPLES / "water_box_64.xyz")
    # the water pair with the shortest O···O distance in the 64-water box, unwrapped
    oxygens = [i for i, s in enumerate(box.get_chemical_symbols()) if s == "O"]
    distances = box.get_distances(oxygens[0], oxygens[1:], mic=True)
    partner = oxygens[1 + int(np.argmin(distances))]
    indices = [oxygens[0], oxygens[0] + 1, oxygens[0] + 2, partner, partner + 1, partner + 2]
    vectors = box.get_distances(oxygens[0], indices, mic=True, vector=True)
    dimer = Atoms([box[i].symbol for i in indices], positions=box[oxygens[0]].position + vectors)
    cell = read(EXAMPLES / "delta_hse06_bbvo/poscars/Ba2BiVO6_cubic40_PBEU/POSCAR")
    return {"NH3 (molecule)": molecule, "water dimer (noncovalent pair)": dimer,
            "Ba2BiVO6 40-atom cell (periodic, d0)": cell}


def cases(args) -> list[dict]:
    std = {"preset": "standard", "model": args.standard}
    found = [
        {"structure": "NH3 (molecule)", **std, "task": "omol"},
        {"structure": "water dimer (noncovalent pair)", **std, "task": "omol"},
        {"structure": "Ba2BiVO6 40-atom cell (periodic, d0)", **std, "task": "omat"},
        {"structure": "Ba2BiVO6 40-atom cell (periodic, d0)", **std, "task": "omat",
         "dispersion": "pbe-d3bj"},
        {"structure": "NH3 (molecule)", "preset": "d4rev", "model": args.d4rev},
        {"structure": "water dimer (noncovalent pair)", "preset": "d4rev", "model": args.d4rev},
        {"structure": "Ba2BiVO6 40-atom cell (periodic, d0)", "preset": "r2scan",
         "model": args.r2scan},
    ]
    return found


def evaluate_timed(calc: UMACalculator, atoms: Atoms, warm: int) -> dict:
    atoms = atoms.copy()
    atoms.calc = calc
    start = time.perf_counter()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces().copy()
    first_wall = time.perf_counter() - start
    first = dict(calc.timing)
    rng = np.random.default_rng(0)
    base = atoms.positions.copy()
    walls, evals, peaks = [], [], []
    for _ in range(warm):
        atoms.positions = base + rng.normal(scale=2e-3, size=base.shape)
        start = time.perf_counter()
        atoms.get_potential_energy()
        walls.append(time.perf_counter() - start)
        evals.append(calc.timing.get("eval_s", np.nan))
        peaks.append(calc.timing.get("peak_gpu_mb", np.nan))
    return {
        "energy_ev": float(energy), "forces": forces,
        "first_call_wall_s": first_wall, "load_s": first.get("load_s"),
        "first_eval_s": first.get("eval_s"),
        "warm_wall_ms": 1e3 * float(np.median(walls)) if walls else None,
        "warm_eval_ms": 1e3 * float(np.median(evals)) if evals else None,
        "peak_gpu_mb": float(np.nanmax(peaks)) if peaks else first.get("peak_gpu_mb"),
        "effective": (calc.backend_info or {}).get("effective"),
        "tf32": (calc.backend_info or {}).get("tf32"),
        "warnings": list((calc.backend_info or {}).get("warnings", [])),
    }


def run_case(python, case, atoms, args, log) -> dict:
    options = {k: v for k, v in case.items() if k != "structure" and v is not None}
    result = {"case": case, "modes": {}}
    for mode in ("nitro", "high-precision"):
        uma_backend.shutdown_workers()  # cold start: initialization measured separately
        messages = []
        try:
            calc = UMACalculator(python, device=args.device, inference=mode,
                                 on_message=messages.append, **options)
            start = time.perf_counter()
            timed = evaluate_timed(calc, atoms, args.warm)
            timed["init_total_s"] = time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001 - reported
            result["modes"][mode] = {"error": f"{type(exc).__name__}: {exc}",
                                     "messages": messages}
            log(f"  {mode}: FAILED {exc}")
            continue
        timed["messages"] = messages
        timed["metadata"] = calc.run_metadata()
        result["modes"][mode] = timed
        log(f"  {mode}: E = {timed['energy_ev']:.6f} eV, effective {timed['effective']}, "
            f"load {timed['load_s']:.1f} s, first eval {timed['first_eval_s']:.2f} s, "
            f"warm {timed['warm_eval_ms']:.1f} ms, peak {timed['peak_gpu_mb']:.0f} MB")
        if mode == "nitro" and not args.skip_dynamics:
            result["relax"] = relax_case(calc, atoms)
            log(f"  relax ({mode}): {result['relax']}")
            if case["structure"].startswith("water dimer"):
                result["md"] = md_case(calc, atoms)
                log(f"  md ({mode}): {result['md']}")
    a, b = (result["modes"].get(m, {}) for m in ("nitro", "high-precision"))
    if "energy_ev" in a and "energy_ev" in b:
        df = np.abs(a["forces"] - b["forces"])
        result["nitro_vs_high_precision"] = {
            "dE_meV": 1e3 * (a["energy_ev"] - b["energy_ev"]),
            "dE_meV_per_atom": 1e3 * (a["energy_ev"] - b["energy_ev"]) / len(atoms),
            "max_dF_meV_per_A": 1e3 * float(df.max()),
            "rms_dF_meV_per_A": 1e3 * float(np.sqrt((df**2).mean())),
        }
        log(f"  nitro - high-precision: {result['nitro_vs_high_precision']}")
    for mode in result["modes"].values():
        mode.pop("forces", None)
    return result


def relax_case(calc, atoms) -> dict:
    atoms = atoms.copy()
    atoms.calc = calc
    start = time.perf_counter()
    outcome = relax(atoms, fmax=0.05, max_steps=300, optimizer="FIRE", min_distance=0.5)
    return {"converged": outcome.converged, "steps": outcome.steps,
            "fmax": float(outcome.evaluation.max_force_ev_per_angstrom)
            if hasattr(outcome.evaluation, "max_force_ev_per_angstrom") else None,
            "energy_ev": float(outcome.evaluation.energy_ev),
            "wall_s": time.perf_counter() - start}


def md_case(calc, atoms) -> dict:
    atoms = atoms.copy()
    atoms.calc = calc
    start = time.perf_counter()
    outcome = run_md(atoms, ensemble="Langevin", temperature_k=300.0, timestep_fs=0.5,
                     steps=200, seed=1, report_interval=50)
    wall = time.perf_counter() - start
    return {"steps": outcome.steps, "mean_temperature_k": outcome.mean_temperature_k,
            "wall_s": wall, "ms_per_step": 1e3 * wall / max(outcome.steps, 1)}


def rule_checks(python, args) -> list[dict]:
    checks = []

    def expect_rejected(name, **config):
        try:
            UMAConfig(**config).validated()
        except UMAConfigError as exc:
            checks.append({"check": name, "ok": True, "message": str(exc)})
            return
        checks.append({"check": name, "ok": False, "message": "accepted"})

    expect_rejected("d4rev + extra D3 (double dispersion)", preset="d4rev",
                    dispersion="pbe-d3bj")
    expect_rejected("standard omol + PBE-D3 (double dispersion)", task="omol",
                    dispersion="pbe-d3bj")
    expect_rejected("standard omc + PBE-D3 (double dispersion)", task="omc",
                    dispersion="pbe-d3bj")
    expect_rejected("standard oc20 (RPBE) + PBE-D3", task="oc20", dispersion="pbe-d3bj")
    expect_rejected("d4rev with task omat", preset="d4rev", task="omat")
    expect_rejected("r2scan with task omol", preset="r2scan", task="omol")
    expect_rejected("r2scan with magnetic=True", preset="r2scan", magnetic=True)
    expect_rejected("r2scan with multiplicity 3", preset="r2scan", multiplicity=3)
    # magnetic moments on the structure: rejected before the worker is asked
    nio = Atoms("NiO", positions=[[0, 0, 0], [2.08, 0, 0]], cell=[4.17] * 3, pbc=True)
    nio.set_initial_magnetic_moments([1.7, 0])
    nio.calc = UMACalculator(python, preset="r2scan", model=args.r2scan, device=args.device)
    try:
        nio.get_potential_energy()
        checks.append({"check": "r2scan refuses initial magmoms", "ok": False,
                       "message": "accepted"})
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "r2scan refuses initial magmoms",
                       "ok": "magnetic moments" in str(exc), "message": str(exc)})
    # fallback reporting: Nitro requested on the CPU
    uma_backend.shutdown_workers()
    water = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    messages = []
    water.calc = calc = UMACalculator(python, model=args.standard, device="cpu",
                                      on_message=messages.append)
    try:
        water.get_potential_energy()
        effective = calc.backend_info["effective"]
        checks.append({"check": "nitro on CPU reports its fallback",
                       "ok": effective != "nitro" and bool(messages),
                       "message": f"effective {effective}; " + " | ".join(messages)})
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "nitro on CPU reports its fallback", "ok": False,
                       "message": str(exc)})
    # composition change: the predictor is rebuilt, not silently slowed down
    messages.clear()
    ammonia = Atoms("NH3", positions=[[0, 0, 0.1], [0.94, 0, -0.27], [-0.47, 0.81, -0.27],
                                       [-0.47, -0.81, -0.27]])
    ammonia.calc = calc
    try:
        ammonia.get_potential_energy()
        checks.append({"check": "composition change rebuilds the predictor",
                       "ok": any("Rebuilt" in m for m in messages) or not (
                           calc.backend_info["nitro"] or calc.backend_info["merge_mole"]),
                       "message": " | ".join(messages) or "no fast path to rebuild"})
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "composition change rebuilds the predictor", "ok": False,
                       "message": str(exc)})
    uma_backend.shutdown_workers()
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("python", help="the fairchem environment's python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--standard", default="uma-s-1p2p1")
    parser.add_argument("--d4rev", default="uma-s-1p2p1-ft-d4rev")
    parser.add_argument("--r2scan", default="uma-s-1p2p1-ft-r2scan")
    parser.add_argument("--warm", type=int, default=20, help="warm evaluations per mode")
    parser.add_argument("--skip-dynamics", action="store_true")
    parser.add_argument("--only", default=None, help="run cases whose preset is this")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # ω, ² on a cp1252 console
    args.out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    found = structures()
    report = {"cases": [], "checks": []}
    for case in cases(args):
        if args.only and case["preset"] != args.only:
            continue
        log(f"{case['structure']}: {case['preset']} {case.get('task', '')} "
            f"{case.get('dispersion', '')}")
        report["cases"].append(run_case(args.python, case, found[case["structure"]], args, log))
    if not args.only:
        report["checks"] = rule_checks(args.python, args)
        for check in report["checks"]:
            log(f"check {'PASS' if check['ok'] else 'FAIL'}: {check['check']}: "
                f"{check['message'][:200]}")
    (args.out / "uma_validation.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    (args.out / "uma_validation.log").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
