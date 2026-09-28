"""UMA on F⁻ + CH₃Cl, standalone, for a GPU node (copied into the LONI package).

    python run_uma_sn2.py CHECKPOINT OUTDIR [--device cuda]

Needs only fairchem-core, ASE, and sella: no SAMSON, no samson-mlip-visualizer.
Inputs (in ``data/``, written by ``make_loni_package.py``):

- ``labeled_frames.extxyz``: frames that already have ωB97X-D/def2-TZVPD labels on
  the desktop (the fine-tuned MACE IRC and the r(C–F) scan). UMA energies and
  forces on them are saved; the comparison with ωB97X-D is done on the desktop.
- ``starts.extxyz``: starting geometries (from the laptop's uma-s-1p1): the
  reactant complex, the Walden TS, and the product complex.

Computed with the checkpoint's ``omol`` task at charge −1 (singlet):

- the two complexes relaxed (BFGS, fmax 0.005 eV/Å) and the TS refined with
  Sella (order 1, exact Hessian every step), with harmonic frequencies at the TS;
- the barrier and complex-to-complex energy from those;
- the separated fragments at their own charges: CH₃Cl and CH₃F relaxed as
  neutrals, and F⁻ and Cl⁻ from the isolated-atom references (``--atom-refs``,
  or ``iso_atom_elem_refs.yaml`` beside the checkpoint or in ``references/``).

Writes ``OUTDIR/results.json`` and ``OUTDIR/labeled_frames_uma.extxyz``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.vibrations import Vibrations

KCAL = 23.0605
C, CL, F = 0, 4, 5  # atom order: C H H H Cl F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atom-refs", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from omegaconf import OmegaConf

    refs = args.atom_refs
    if refs is None:
        for candidate in (args.checkpoint.with_name("iso_atom_elem_refs.yaml"),
                          args.checkpoint.parent / "references" / "iso_atom_elem_refs.yaml"):
            if candidate.is_file():
                refs = candidate
                break
    t0 = time.perf_counter()
    unit = pretrained_mlip.load_predict_unit(
        str(args.checkpoint), device=args.device,
        atom_refs=OmegaConf.load(refs) if refs else None)
    load_s = time.perf_counter() - t0
    calc = FAIRChemCalculator(unit, task_name="omol")

    def prepare(atoms, charge=-1, spin=1):
        atoms = atoms.copy()
        atoms.info.update(charge=charge, spin=spin)
        atoms.calc = calc
        return atoms

    results = {"checkpoint": args.checkpoint.name, "device": args.device,
               "load_s": load_s, "atom_refs": str(refs) if refs else None}

    # 1. Energies and forces on the frames labeled with ωB97X-D on the desktop.
    frames = read(args.data / "labeled_frames.extxyz", ":")
    t0 = time.perf_counter()
    out = []
    for frame in frames:
        atoms = prepare(Atoms(frame.numbers, frame.positions))
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        image = Atoms(frame.numbers, frame.positions, info=dict(frame.info))
        image.info["uma_energy"] = float(energy)
        image.arrays["uma_forces"] = np.asarray(forces)
        out.append(image)
    write(args.outdir / "labeled_frames_uma.extxyz", out)
    results["seconds_per_frame"] = (time.perf_counter() - t0) / len(frames)

    # 2. Stationary points.
    starts = {s.info["name"]: s for s in read(args.data / "starts.extxyz", ":")}
    points = {}
    for name in ("reactant_complex", "product_complex"):
        atoms = prepare(Atoms(starts[name].numbers, starts[name].positions))
        BFGS(atoms, logfile=None).run(fmax=0.005, steps=500)
        points[name] = atoms
    from sella import Sella

    ts = prepare(Atoms(starts["ts"].numbers, starts["ts"].positions))
    search = Sella(ts, order=1, internal=True, diag_every_n=1, logfile=None)
    converged = bool(search.run(fmax=0.005, steps=300))
    vib = Vibrations(ts, name=str(args.outdir / "vib_ts"), delta=0.005)
    vib.clean()
    vib.run()
    wavenumbers = vib.get_frequencies()  # cm⁻¹; imaginary ones as complex
    vib.clean()
    imaginary = sorted(-abs(w.imag) for w in wavenumbers if abs(w.imag) > 50)
    points["ts"] = ts

    def geometry(atoms):
        return {"r_CF": float(atoms.get_distance(C, F)),
                "r_CCl": float(atoms.get_distance(C, CL)),
                "energy_ev": float(atoms.get_potential_energy()),
                "positions": atoms.positions.tolist()}

    stationary = {name: geometry(atoms) for name, atoms in points.items()}
    stationary["ts"].update(converged=converged, imaginary_cm=imaginary)
    e = {name: point["energy_ev"] for name, point in stationary.items()}
    results["stationary"] = stationary
    results["barrier_kcal"] = (e["ts"] - e["reactant_complex"]) * KCAL
    results["complex_to_complex_kcal"] = (e["product_complex"] - e["reactant_complex"]) * KCAL

    # 3. Separated fragments, each at its own charge.
    fragments = {}
    for name in ("CH3Cl", "CH3F"):
        atoms = prepare(Atoms(starts[name].numbers, starts[name].positions), charge=0)
        BFGS(atoms, logfile=None).run(fmax=0.005, steps=300)
        fragments[name] = float(atoms.get_potential_energy())
    for name, symbol in (("F-", "F"), ("Cl-", "Cl")):
        try:
            fragments[name] = float(prepare(Atoms(symbol)).get_potential_energy())
        except Exception as exc:  # noqa: BLE001 - no references: say so in the results
            fragments[name] = None
            results.setdefault("errors", []).append(f"{name}: {type(exc).__name__}: {exc}")
    results["fragments_ev"] = fragments
    if fragments["F-"] is not None and fragments["Cl-"] is not None:
        zero = fragments["CH3Cl"] + fragments["F-"]
        results["relative_to_reactants_kcal"] = {
            "reactant_complex": (e["reactant_complex"] - zero) * KCAL,
            "ts": (e["ts"] - zero) * KCAL,
            "product_complex": (e["product_complex"] - zero) * KCAL,
            "products": (fragments["CH3F"] + fragments["Cl-"] - zero) * KCAL,
        }
    (args.outdir / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "stationary"}, indent=1))


if __name__ == "__main__":
    main()
