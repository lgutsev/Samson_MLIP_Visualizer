"""Headless entry point: evaluate or relax a structure file without SAMSON.

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
from .provenance import collect_provenance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samson-mlip",
        description="Run a MACE or DeepMD model on a structure file (no SAMSON required).",
    )
    parser.add_argument(
        "structure", type=Path, help="Structure file readable by ASE (xyz, cif, ...)"
    )
    parser.add_argument("model", type=Path, help="Trained MACE or DeepMD model file")
    parser.add_argument("--backend", choices=["mace", "deepmd"], default="mace")
    parser.add_argument("--device", default="cpu", help="MACE device, e.g. cpu or cuda")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument(
        "--relax", action="store_true", help="FIRE-relax positions instead of a single point"
    )
    parser.add_argument(
        "--fmax", type=float, default=0.05, help="Force threshold for --relax (eV/A)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=250, help="Maximum FIRE steps for --relax"
    )
    parser.add_argument(
        "--trajectory", type=Path, default=None, help="ASE trajectory output for --relax"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Write the final structure here"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from ase.io import read, write

    atoms = read(args.structure)
    calculator = create_calculator(
        args.backend,
        args.model,
        device=args.device,
        dtype=args.dtype,
    )
    atoms.calc = calculator

    provenance = collect_provenance(
        backend=args.backend, model_path=args.model, device=args.device, dtype=args.dtype
    )
    print(provenance.as_text())
    print()

    supported = assert_model_covers_structure(calculator, atoms)
    if supported is not None:
        print("Model training elements:", ", ".join(supported))

    if args.relax:
        result = relax(
            atoms,
            fmax=args.fmax,
            max_steps=args.max_steps,
            trajectory=args.trajectory,
            on_progress=lambda step, energy, fmax, _pos: print(
                f"step {step:4d}  E = {energy:.8f} eV  Fmax = {fmax:.6f} eV/A"
            ),
        )
        state = "stopped" if result.stopped else "converged" if result.converged else "step limit"
        evaluation = result.evaluation
        print(f"Finished ({state}) after {result.steps} steps")
    else:
        evaluation = evaluate(atoms)

    print(f"Energy: {evaluation.energy_ev:.10f} eV")
    print(f"Max force: {evaluation.max_force_ev_per_angstrom:.6f} eV/A")

    if args.output is not None:
        atoms.info.update(provenance.as_dict())
        write(args.output, atoms)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
