"""Headless entry point: evaluate, relax, run MD on, or TS-search a structure file.

This exists so a model can be sanity-checked against a known structure before it
is trusted inside SAMSON. It uses ASE for I/O and shares the calculator and
engine layers with the panel.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .calculators import create_calculator
from .compat import assert_model_covers_structure
from .engine import evaluate, relax
from .md import ENSEMBLES, md_warnings, parse_pairs, run_md
from .provenance import collect_provenance
from .sanity import check_sane
from .ts import dimer_search, prfo_search
from .vibrations import harmonic_frequencies


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samson-mlip",
        description="Run a MACE or DeepMD model, or GFN-xTB, on a structure file (no SAMSON "
        "required).",
    )
    parser.add_argument(
        "structure", type=Path, help="Structure file readable by ASE (xyz, cif, ...)"
    )
    parser.add_argument(
        "model",
        type=Path,
        nargs="+",
        help="Trained model file(s). Several MACE files form an uncertainty committee. "
        "With --backend xtb: the xtb executable, or 'auto' to find it.",
    )
    parser.add_argument("--backend", choices=["mace", "deepmd", "xtb"], default="mace")
    parser.add_argument(
        "--xtb-method", choices=["gfn2", "gfn1", "gfnff"], default="gfn2", help="xTB method"
    )
    parser.add_argument(
        "--solvent", default=None, help="xTB implicit solvent (ALPB), e.g. water or toluene"
    )
    parser.add_argument("--device", default="cpu", help="MACE device, e.g. cpu or cuda")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--relax", action="store_true", help="Relax positions instead of a single point"
    )
    mode.add_argument("--md", action="store_true", help="Run molecular dynamics")
    mode.add_argument(
        "--ts", action="store_true", help="Search for a transition state (P-RFO or dimer)"
    )
    mode.add_argument(
        "--irc",
        action="store_true",
        help="Follow the IRC both ways from a transition state (--trajectory saves all frames)",
    )
    mode.add_argument(
        "--qst",
        type=Path,
        default=None,
        metavar="PRODUCT",
        help="QST2 path search from the structure (reactant) to PRODUCT; with --qst-guess, "
        "QST3 (--trajectory saves the band)",
    )
    mode.add_argument(
        "--scan",
        default=None,
        metavar="I-J:STOP[:POINTS]",
        help="Scan the I-J distance to STOP (A) relaxing everything else, then refine the "
        "highest point with P-RFO (--trajectory saves the scan)",
    )
    parser.add_argument(
        "--freq",
        action="store_true",
        help="Finish with finite-difference frequencies (6 force calls per free atom)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["FIRE", "LBFGS", "BFGS", "PreconLBFGS"],
        default="FIRE",
        help="Local optimizer for --relax (default FIRE)",
    )
    parser.add_argument(
        "--fmax", type=float, default=0.05, help="Force threshold for --relax (eV/A)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=250, help="Maximum optimizer steps for --relax"
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.5,
        help="Abort if two atoms come closer than this (A); 0 disables the check",
    )
    parser.add_argument(
        "--max-drift",
        type=float,
        default=None,
        help="Abort if any atom moves more than this far (A) from its start",
    )
    parser.add_argument(
        "--max-force-std",
        type=float,
        default=None,
        help="Abort if the committee force spread exceeds this (eV/A)",
    )
    md = parser.add_argument_group("molecular dynamics (--md)")
    md.add_argument("--ensemble", choices=ENSEMBLES, default="Langevin")
    md.add_argument("--temperature", type=float, default=300.0, help="Kelvin")
    md.add_argument("--timestep", type=float, default=0.5, help="fs (0.5 with hydrogen)")
    md.add_argument("--md-steps", type=int, default=1000)
    md.add_argument("--friction", type=float, default=0.01, help="Langevin friction (1/fs)")
    md.add_argument(
        "--tdamp", type=float, default=100.0, help="Bussi / Nose-Hoover time constant (fs)"
    )
    md.add_argument("--seed", type=int, default=None, help="Seed for velocities and noise")
    md.add_argument(
        "--fix-distance",
        action="append",
        default=[],
        metavar="I-J[:R]",
        help="Hold the distance between 0-based atoms I and J (optionally at R A); repeatable",
    )
    md.add_argument(
        "--max-temperature", type=float, default=None, help="Abort above this temperature (K)"
    )
    md.add_argument("--report-interval", type=int, default=10, help="Print every N steps")
    ts = parser.add_argument_group("transition-state search (--ts)")
    ts.add_argument(
        "--ts-method",
        choices=["prfo", "dimer"],
        default=None,
        help="P-RFO (Sella; default) or the dimer method (default when --ts-start or "
        "--ts-pair is given)",
    )
    ts.add_argument(
        "--ts-start",
        choices=["hessian", "pair", "random"],
        default=None,
        help="Initial dimer direction: the softest Hessian mode (default), a stretched "
        "atom pair (default when --ts-pair is given), or a random displacement",
    )
    ts.add_argument(
        "--ts-pair",
        default=None,
        metavar="I-J",
        help="0-based atom pair to stretch for --ts-start pair",
    )
    ts.add_argument(
        "--ts-displacement", type=float, default=0.05, help="Initial displacement norm (A)"
    )
    ts.add_argument(
        "--exact-hessian", action="store_true", help="P-RFO from an exact Hessian (CalcFC)"
    )
    ts.add_argument(
        "--recompute-hessian",
        type=int,
        default=None,
        metavar="N",
        help="Recompute the exact Hessian every N P-RFO steps (RecalcFC=N)",
    )
    qm = parser.add_argument_group("quantum-chemistry export")
    qm.add_argument(
        "--export-qm",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write a Gaussian (.gjf/.com) or ORCA (.inp) input from the final structure",
    )
    qm.add_argument(
        "--qm-job",
        choices=["ts", "opt", "irc"],
        default=None,
        help="Job for --export-qm (default: ts after --ts/--scan/--qst, else opt)",
    )
    qm.add_argument("--qm-level", default=None, help="Method and basis for --export-qm")
    qm.add_argument(
        "--charge", type=int, default=0, help="Molecular charge (xTB backend and --export-qm)"
    )
    qm.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (xTB backend and --export-qm)",
    )
    path = parser.add_argument_group("reaction paths (--irc, --qst)")
    path.add_argument("--irc-step", type=float, default=0.1, help="IRC arc step (A amu^1/2)")
    path.add_argument("--qst-guess", type=Path, default=None, help="TS guess for QST3")
    path.add_argument("--images", type=int, default=7, help="Interior images for --qst")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Trajectory output for --relax, --md or --ts (format from the extension)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Write the final structure here"
    )
    return parser


def _print_uncertainty(label: str, evaluation) -> None:
    if evaluation.energy_std_ev is not None:
        print(f"{label} committee energy std: {evaluation.energy_std_ev:.6f} eV")
    if evaluation.max_force_std_ev_per_angstrom is not None:
        print(
            f"{label} committee max force std: "
            f"{evaluation.max_force_std_ev_per_angstrom:.6f} eV/A"
        )


def _run_md(args, atoms) -> None:
    for message in md_warnings(
        atoms, timestep_fs=args.timestep, ensemble=args.ensemble, dtype=args.dtype
    ):
        print("Warning:", message)
    constraints = [pair for text in args.fix_distance for pair in parse_pairs(text)]
    thermostat = f" at {args.temperature:g} K" if args.ensemble != "NVE" else ""
    print(f"{args.ensemble} MD: {args.md_steps} steps x {args.timestep:g} fs{thermostat}")
    result = run_md(
        atoms,
        ensemble=args.ensemble,
        temperature_k=args.temperature,
        timestep_fs=args.timestep,
        steps=args.md_steps,
        friction_per_fs=args.friction,
        tdamp_fs=args.tdamp,
        seed=args.seed,
        distance_constraints=constraints,
        report_interval=args.report_interval,
        min_distance=args.min_distance if args.min_distance > 0 else None,
        max_force_std=args.max_force_std,
        max_temperature_k=args.max_temperature,
        trajectory=args.trajectory,
        trajectory_interval=args.report_interval,
        on_progress=lambda frame: print(
            f"step {frame.step:6d}  t = {frame.time_fs:9.1f} fs  "
            f"Epot = {frame.potential_ev:.6f}  Etot = {frame.total_ev:.6f} eV  "
            f"T = {frame.temperature_k:7.1f} K"
        ),
    )
    print(
        f"Finished after {result.steps} steps ({result.time_fs:g} fs); "
        f"mean T = {result.mean_temperature_k:.1f} K"
    )
    if result.energy_drift_mev_per_atom_ps is not None:
        print(f"NVE energy drift: {result.energy_drift_mev_per_atom_ps:+.3f} meV/atom/ps")
    for summary in result.constraint_forces:
        print(
            f"Constraint {summary.i}-{summary.j} at {summary.distance:.4f} A: mean force "
            f"{summary.mean_force_ev_per_angstrom:+.5f} +/- {summary.std_ev_per_angstrom:.5f} "
            f"eV/A (std, {summary.samples} samples; positive pushes apart)"
        )


def _run_irc(args, atoms):
    from ase.io import write

    from .reaction_path import irc

    print("IRC: frequencies to find the imaginary mode, then both directions")
    result = irc(
        atoms,
        step=args.irc_step,
        max_steps=args.max_steps,
        fmax=args.fmax,
        on_progress=lambda direction, step, energy, fmax, _pos: print(
            f"{direction:7s} step {step:4d}  E = {energy:.8f} eV  Fmax = {fmax:.6f} eV/A"
        )
        if step % 10 == 0
        else None,
    )
    frames = result.frames(atoms.get_positions())
    print(
        f"IRC: {len(result.reverse)} reverse + {len(result.forward)} forward frames "
        f"({len(frames)} with the TS, which is frame {len(result.reverse)})"
    )
    for name in ("reverse", "forward"):
        minimum = getattr(result, f"{name}_minimum_ev")
        if minimum is not None:
            print(f"{name} minimum: {minimum - result.ts_energy_ev:+.6f} eV relative to the TS")
    if args.trajectory is not None:
        images = []
        for frame in frames:
            image = atoms.copy()
            image.set_positions(frame.positions)
            image.info.update({"energy_ev": frame.energy_ev, "irc_arc": frame.arc})
            images.append(image)
        write(args.trajectory, images)
        print(f"Wrote all {len(images)} IRC frames to {args.trajectory}")
    return evaluate(atoms)


def _run_scan(args, atoms):
    from ase.io import write

    from .reaction_path import scan_to_ts

    pair_text, _, rest = args.scan.partition(":")
    stop_text, _, points_text = rest.partition(":")
    try:
        (constraint,) = parse_pairs(pair_text)
        stop = float(stop_text)
        points = int(points_text) if points_text else 11
    except ValueError as exc:
        raise SystemExit(f"--scan expects I-J:STOP[:POINTS], not {args.scan!r}") from exc
    pair = (constraint.i, constraint.j)
    start = atoms.get_distance(*pair, mic=True)
    print(f"Scanning {pair[0]}-{pair[1]} from {start:.3f} to {stop:.3f} A in {points} points")
    result = scan_to_ts(
        atoms,
        pair,
        stop=stop,
        points=points,
        ts_fmax=args.fmax,
        exact_hessian=args.exact_hessian,
        on_progress=lambda index, distance, energy, _pos: print(
            f"point {index:3d}  r = {distance:.4f} A  E = {energy:.8f} eV"
        ),
    )
    peak = result.distances[result.highest]
    bracket = "bracketed" if result.bracketed else "NOT bracketed: extend the range"
    print(f"Highest point r = {peak:.4f} A ({bracket})")
    if args.trajectory is not None:
        images = []
        for positions, energy, distance in zip(
            result.frames, result.energies_ev, result.distances, strict=True
        ):
            image = atoms.copy()
            image.set_positions(positions)
            image.info.update({"energy_ev": energy, "scan_distance": distance})
            images.append(image)
        write(args.trajectory, images)
        print(f"Wrote the {len(images)}-point scan to {args.trajectory}")
    if result.ts is not None:
        state = "converged" if result.ts.converged else "not converged"
        print(f"P-RFO from the highest point {state} after {result.ts.steps} steps")
    return evaluate(atoms)


def _run_qst(args, reactant, calculator):
    from ase.io import read, write

    from .reaction_path import qst

    product = read(args.qst)
    guess = read(args.qst_guess) if args.qst_guess is not None else None
    label = "QST3" if guess is not None else "QST2"
    print(f"{label}: IDPP interpolation, climbing-image NEB, P-RFO refinement")
    result = qst(
        reactant,
        product,
        calculator,
        guess=guess,
        images=args.images,
        fmax=max(args.fmax, 0.05),
        max_steps=args.max_steps,
        on_progress=lambda step, energies, fmax: print(
            f"band step {step:4d}  Fmax = {fmax:.4f} eV/A  "
            f"max dE = {max(energies) - energies[0]:+.4f} eV"
        )
        if step % 5 == 0
        else None,
    )
    print(
        f"Band {'converged' if result.neb_converged else 'not converged'} "
        f"({result.neb_steps} steps); barrier {result.barrier_forward_ev:.6f} eV forward, "
        f"{result.barrier_reverse_ev:.6f} eV reverse"
    )
    if args.trajectory is not None:
        images = []
        for positions, energy in zip(result.images, result.energies_ev, strict=True):
            image = reactant.copy()
            image.set_positions(positions)
            image.info["energy_ev"] = energy
            images.append(image)
        write(args.trajectory, images)
        print(f"Wrote the {len(images)}-image band to {args.trajectory}")
    if result.ts is not None:
        reactant.set_positions(result.ts_positions)
        state = "converged" if result.ts.converged else "not converged"
        print(f"P-RFO refinement {state} after {result.ts.steps} steps; the TS is the output")
    return evaluate(reactant)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from ase.io import read, write

    atoms = read(args.structure)
    xtb = settings = None
    if args.backend == "xtb":
        if [str(path) for path in args.model] == ["auto"]:
            from .xtb_backend import find_xtb

            found = find_xtb()
            if found is None:
                raise SystemExit(
                    "No xtb executable found; install it with `micromamba create -n xtb "
                    "-c conda-forge xtb`, or pass its path or set XTB_EXE."
                )
            args.model = [found]
        xtb = settings = {
            "method": args.xtb_method,
            "charge": args.charge,
            "multiplicity": args.multiplicity,
            "solvent": args.solvent,
        }
    model_arg = args.model[0] if len(args.model) == 1 else args.model
    calculator = create_calculator(
        args.backend,
        model_arg,
        device=args.device,
        dtype=args.dtype,
        xtb=xtb,
    )
    atoms.calc = calculator

    provenance = collect_provenance(
        backend=args.backend,
        model_path=args.model[0],
        device=args.device,
        dtype=args.dtype,
        settings=settings,
    )
    print(provenance.as_text())
    if len(args.model) > 1:
        print(f"committee:      {len(args.model)} models")
    print()

    if args.dtype == "float32" and (args.relax or args.ts) and args.backend == "mace":
        print("Warning: MACE recommends float64 for geometry optimization; float32 "
              "force noise can stall convergence.\n")

    check_sane(atoms, min_distance=args.min_distance)

    supported = assert_model_covers_structure(calculator, atoms)
    if supported is not None:
        print("Model training elements:", ", ".join(supported))

    # Choosing a dimer start implies the dimer method; otherwise P-RFO is the default.
    ts_method = args.ts_method or ("dimer" if args.ts_start or args.ts_pair else "prfo")
    if args.relax:
        result = relax(
            atoms,
            fmax=args.fmax,
            max_steps=args.max_steps,
            optimizer=args.optimizer,
            min_distance=args.min_distance if args.min_distance > 0 else None,
            max_drift=args.max_drift,
            max_force_std=args.max_force_std,
            trajectory=args.trajectory,
            on_progress=lambda step, energy, fmax, _pos: print(
                f"step {step:4d}  E = {energy:.8f} eV  Fmax = {fmax:.6f} eV/A"
            ),
        )
        state = "stopped" if result.stopped else "converged" if result.converged else "step limit"
        evaluation = result.evaluation
        print(f"Finished ({state}) after {result.steps} steps")
    elif args.md:
        _run_md(args, atoms)
        evaluation = evaluate(atoms)
    elif args.ts and ts_method == "prfo":
        print("P-RFO (Sella) transition-state search")
        result = prfo_search(
            atoms,
            fmax=args.fmax,
            max_steps=args.max_steps,
            exact_hessian=args.exact_hessian,
            recompute_every=args.recompute_hessian,
            min_distance=args.min_distance if args.min_distance > 0 else None,
            trajectory=args.trajectory,
            on_progress=lambda step, energy, fmax, curvature, _pos: print(
                f"step {step:4d}  E = {energy:.8f} eV  Fmax = {fmax:.6f} eV/A"
            ),
        )
        state = "converged" if result.converged else "not converged"
        evaluation = result.evaluation
        print(f"Finished ({state}) after {result.steps} steps")
    elif args.irc:
        evaluation = _run_irc(args, atoms)
    elif args.qst is not None:
        evaluation = _run_qst(args, atoms, calculator)
    elif args.scan is not None:
        evaluation = _run_scan(args, atoms)
    elif args.ts:
        start = args.ts_start or ("pair" if args.ts_pair else "hessian")
        pair = None
        if start == "pair":
            if not args.ts_pair:
                raise SystemExit("--ts-start pair needs --ts-pair I-J")
            (constraint,) = parse_pairs(args.ts_pair)
            pair = (constraint.i, constraint.j)
        print(f"Dimer search starting along the {start} direction")
        result = dimer_search(
            atoms,
            fmax=args.fmax,
            max_steps=args.max_steps,
            pair=pair,
            use_hessian=start == "hessian",
            displacement=args.ts_displacement,
            seed=args.seed,
            min_distance=args.min_distance if args.min_distance > 0 else None,
            trajectory=args.trajectory,
            on_progress=lambda step, energy, fmax, curvature, _pos: print(
                f"step {step:4d}  E = {energy:.8f} eV  Fmax = {fmax:.6f} eV/A  "
                f"curvature = {curvature:+.4f} eV/A^2"
            ),
        )
        if result.stopped:
            state = "stopped"
        else:
            state = "converged" if result.converged else "not converged"
        evaluation = result.evaluation
        print(f"Finished ({state}) after {result.steps} steps; curvature {result.curvature:+.4f}")
    else:
        evaluation = evaluate(atoms)

    print(f"Energy: {evaluation.energy_ev:.10f} eV")
    print(f"Max force: {evaluation.max_force_ev_per_angstrom:.6f} eV/A")
    _print_uncertainty("Final", evaluation)

    if args.freq:
        frequencies = harmonic_frequencies(atoms)
        print("Wavenumbers (cm^-1, negative = imaginary):")
        print("  " + " ".join(f"{value:.1f}" for value in frequencies.wavenumbers_cm))
        print("Stationary point:", frequencies.classification())
        hint = frequencies.soft_mode_hint()
        if hint:
            print("Note:", hint)

    if args.output is not None:
        atoms.info.update(provenance.as_dict())
        write(args.output, atoms)
        print(f"Wrote {args.output}")
    if args.export_qm is not None:
        from .qm_export import write_qm_input

        searched = args.ts or args.scan is not None or args.qst is not None
        job = args.qm_job or ("ts" if searched else "opt")
        written = write_qm_input(
            args.export_qm,
            [atoms],
            job=job,
            level=args.qm_level,
            charge=args.charge,
            multiplicity=args.multiplicity,
        )
        print(f"Wrote {job.upper()} input: " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
