"""Refine failed W2 Gaussian saddle frames on the Fe2 spin-MACE; write G09 TS restarts.

    python refine_ts_guess.py GUESS.extxyz [MORE.extxyz ...] --route-from W2_JOB.gjf --out DIR
    python refine_ts_guess.py GUESS.extxyz --reactant R.extxyz --product P.extxyz \
        --route-from W2_JOB.gjf --out DIR          # QST3 (QST2 without GUESS)

Each frame of the GUESS files (the best index-1 frame of a failed TS run) is
refined on the charge/spin-embedded Fe2 MACE (``examples/fe_spin_ladder/
train_fe2_spin.py``): P-RFO with an exact initial Hessian, or IDPP + CI-NEB +
P-RFO when both IRC endpoints are given. On the result it computes MLIP
frequencies (exactly one imaginary mode wanted), the barrier (from the given
endpoints, or from an MLIP IRC to both sides), and the RMSD from the start frame.

For every guess that passes (or all, with ``--write-failed``) it writes
``DIR/<name>.ts.gjf``: the route of ``--route-from`` with only the Opt keyword
set to ``Opt=(TS,CalcFC,NoEigenTest)`` and Freq ensured, explicit Cartesians,
and the frame's charge/multiplicity. It refuses a template without IOP(5/13=1).
Also ``DIR/<name>.mlip_ts.extxyz`` and ``DIR/report.json``.

Charge and multiplicity come from ``atoms.info`` (``charge``; ``multiplicity``,
``mult`` or ``spin``, where ``spin`` is the multiplicity as in W2) or from
``--charge``/``--multiplicity``. The MLIP is only a guess generator: the
Gaussian Freq run decides whether the saddle is real. It does not run DFT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
warnings.filterwarnings("ignore")

from ase.io import read, write  # noqa: E402

from samson_mlip_visualizer.qm_export import (  # noqa: E402
    gaussian_ts_input_from_template,
    parse_gaussian_template,
    ts_route_from_template,
)
from samson_mlip_visualizer.ts_guess import refine_guess  # noqa: E402

DEFAULT_MODEL = Path(os.environ.get(
    "FE2_SPIN_MODEL", r"D:\MLIP_Work_Folder\fe_spin_ladder\fe2_spin\seed1\fe2_spin_seed1.model"))
TRAINED_ELEMENTS = {"Fe", "H", "N", "O"}


def charge_and_multiplicity(atoms, charge=None, multiplicity=None) -> tuple[int, int]:
    info = atoms.info
    if charge is None:
        charge = info.get("charge")
    if multiplicity is None:
        multiplicity = next((info[k] for k in ("multiplicity", "mult", "spin") if k in info), None)
    if charge is None or multiplicity is None:
        raise SystemExit(f"{info.get('name', 'frame')}: no charge/multiplicity in info; "
                         "pass --charge/--multiplicity")
    return int(charge), int(multiplicity)


def tag(atoms, charge: int, multiplicity: int):
    """Set every info key the spin-MACE (or another spin-aware model) may read."""
    atoms.info.update(charge=charge, multiplicity=multiplicity, spin=multiplicity,
                      total_spin=multiplicity, total_charge=charge)
    return atoms


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "guess"


def load_calculator(model: Path, device: str):
    from mace.calculators import MACECalculator

    if not model.is_file():
        raise SystemExit(f"No MACE model at {model} (set --model or FE2_SPIN_MODEL)")
    return MACECalculator(model_paths=str(model), device=device, default_dtype="float64")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("guess", nargs="*", type=Path, help="extxyz file(s) of TS guess frames")
    parser.add_argument("--reactant", type=Path)
    parser.add_argument("--product", type=Path)
    parser.add_argument("--route-from", type=Path, required=True,
                        help="a W2 Gaussian .gjf whose route/Link 0 lines are copied")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--charge", type=int)
    parser.add_argument("--multiplicity", type=int)
    parser.add_argument("--fmax", type=float, default=0.01, help="eV/Å, MLIP saddle tolerance")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--no-irc", action="store_true", help="skip the MLIP IRC check")
    parser.add_argument("--strategy", choices=("auto", "prfo", "qst"), default="auto",
                        help="with a frame and endpoints: P-RFO first, then QST3 (auto)")
    parser.add_argument("--no-relax-endpoints", action="store_true")
    parser.add_argument("--write-failed", action="store_true",
                        help="write .gjf files also for guesses that fail the MLIP check")
    args = parser.parse_args(argv)

    template_text = args.route_from.read_text(encoding="utf-8", errors="replace")
    template = parse_gaussian_template(template_text)
    ts_route_from_template(template.route)  # fail loudly (no IOP(5/13=1)) before any MLIP work

    jobs = []  # (name, guess or None, reactant or None, product or None)
    frames = [(path, k, atoms) for path in args.guess
              for k, atoms in enumerate(read(path, ":"))]
    if (args.reactant is None) != (args.product is None):
        parser.error("--reactant and --product go together")
    if args.reactant is not None:
        if len(frames) > 1:
            parser.error("with --reactant/--product give at most one guess frame")
        reactant, product = read(args.reactant), read(args.product)
        guess = frames[0][2] if frames else None
        name = (guess or reactant).info.get("name") or args.reactant.stem
        jobs.append((safe_name(str(name).removesuffix("_guess").removesuffix("_reactant")),
                     guess, reactant, product))
    elif not frames:
        parser.error("give at least one guess frame, or --reactant and --product")
    else:
        for path, k, atoms in frames:
            name = atoms.info.get("name") or atoms.info.get("record_id") or \
                (path.stem if k == 0 else f"{path.stem}_{k}")
            jobs.append((safe_name(str(name)), atoms, None, None))

    calc = load_calculator(args.model, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"model": str(args.model), "template": str(args.route_from),
              "template_route": " ".join(template.route), "jobs": {}}
    for name, guess, reactant, product in jobs:
        first = guess if guess is not None else reactant
        charge, multiplicity = charge_and_multiplicity(first, args.charge, args.multiplicity)
        for atoms in (guess, reactant, product):
            if atoms is not None:
                tag(atoms, charge, multiplicity)
        symbols = set(first.get_chemical_symbols())
        notes = []
        if first.get_chemical_symbols().count("Fe") != 2 or not symbols <= TRAINED_ELEMENTS:
            notes.append("outside the Fe2XY (X, Y = H, N, O) training set of this model")
        if template.charge is not None and (template.charge, template.multiplicity) != \
                (charge, multiplicity):
            notes.append(f"template molecule spec is {template.charge} {template.multiplicity}; "
                         f"the frame's {charge} {multiplicity} is written")
        print(f"{name}: {first.get_chemical_formula()} q={charge} M={multiplicity} ...",
              flush=True)
        result = refine_guess(calc, guess, reactant=reactant, product=product, fmax=args.fmax,
                              max_steps=args.max_steps, run_irc=not args.no_irc,
                              relax_endpoints=not args.no_relax_endpoints,
                              strategy=args.strategy)
        result.notes[:0] = notes
        ts = tag(first.copy(), charge, multiplicity)
        ts.set_positions(result.positions)
        ts.info.update(name=f"{name}_mlip_ts", mlip_energy_eV=result.energy_ev,
                       mlip_n_imaginary=result.n_imaginary, mlip_converged=result.converged)
        write(args.out / f"{name}.mlip_ts.extxyz", ts, format="extxyz")
        entry = result.summary()
        if result.ok or args.write_failed:
            gjf = args.out / f"{name}.ts.gjf"
            text, changes = gaussian_ts_input_from_template(
                template, ts, charge=charge, multiplicity=multiplicity,
                title=f"{name} TS guess from Fe2 spin-MACE ({result.method}, "
                      f"{result.n_imaginary} imag on MLIP)",
                chk=f"{name}.ts.chk")
            gjf.write_text(text, encoding="utf-8")
            entry.update(gjf=str(gjf), route_changes=changes)
        report["jobs"][name] = entry
        barrier = ("n/a" if result.barrier_forward_ev is None else
                   f"{result.barrier_forward_ev:.3f} / {result.barrier_reverse_ev:.3f} eV")
        rmsd = "n/a" if result.rmsd_from_start is None else f"{result.rmsd_from_start:.3f} Å"
        print(f"  {result.method}: converged={result.converged} steps={result.steps} "
              f"imaginary={result.imaginary_cm} barrier fwd/rev={barrier} RMSD={rmsd} "
              f"-> {'gjf written' if 'gjf' in entry else 'no gjf (failed MLIP check)'}")
        for note in result.notes:
            print(f"  note: {note}")
    path = args.out / "report.json"  # one file per output folder; jobs of earlier runs kept
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        report["jobs"] = {**previous.get("jobs", {}), **report["jobs"]}
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if all(report["jobs"][name]["ok"] for name, *_ in jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
