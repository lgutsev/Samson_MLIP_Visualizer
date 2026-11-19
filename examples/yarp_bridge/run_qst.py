"""Take the reactant/product pairs written by ``enumerate_yarp.py`` through the
visualizer's transition-state protocol with an MLIP.

    python run_qst.py runs/acetaldehyde

For each pair: relax both end points, check that each still has the bonding YARP
intended, QST2 (NEB + P-RFO refinement), frequencies of the TS, and an IRC whose
relaxed ends are compared with the two minima. A TS is reported as ``confirmed``
only when it has exactly one imaginary mode and the IRC reaches both minima.

Runs in the visualizer's Python environment. Results go next to each pair
(``ts.xyz``, ``reactant_relaxed.xyz``, ``product_relaxed.xyz``) and to
``results.json`` in the run folder.
"""

import argparse
import json
import warnings
from pathlib import Path

from ase.build import minimize_rotation_and_translation
from ase.data import covalent_radii
from ase.io import read, write

from samson_mlip_visualizer.engine import relax
from samson_mlip_visualizer.paths import foundation_model
from samson_mlip_visualizer.reaction_path import irc, qst
from samson_mlip_visualizer.vibrations import harmonic_frequencies

BONDED = 1.3  # a pair is bonded below this multiple of the summed covalent radii


def bonded(atoms, pair) -> bool:
    i, j = pair
    limit = BONDED * (covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]])
    return atoms.get_distance(i, j) < limit


def identify(atoms, broken, formed) -> str:
    """Which end of the reaction ``atoms`` is, judged by the bonds that change."""
    has_broken = [bonded(atoms, pair) for pair in broken]
    has_formed = [bonded(atoms, pair) for pair in formed]
    if all(has_broken) and not any(has_formed):
        return "reactant"
    if all(has_formed) and not any(has_broken):
        return "product"
    return "other"


def run_pair(folder: Path, entry: dict, calculator, args) -> dict:
    broken = [tuple(pair) for pair in entry["broken"]]
    formed = [tuple(pair) for pair in entry["formed"]]
    out = {"id": entry["id"], "product_smiles": entry["product_smiles"]}

    ends = {}
    for name in ("reactant", "product"):
        atoms = read(folder / f"{name}.xyz")
        atoms.calc = calculator
        result = relax(atoms, fmax=args.fmax_minimum, optimizer="LBFGS", max_steps=1000)
        if not result.converged:
            return out | {"status": f"{name} did not relax"}
        kind = identify(atoms, broken, formed)
        if kind != name:
            return out | {"status": f"{name} relaxed to {kind}"}
        ends[name] = atoms
    reactant, product = ends["reactant"], ends["product"]
    minimize_rotation_and_translation(reactant, product)
    write(folder / "reactant_relaxed.xyz", reactant)
    write(folder / "product_relaxed.xyz", product)
    e_reactant = float(reactant.get_potential_energy())
    out["reaction_ev"] = float(product.get_potential_energy()) - e_reactant

    search = qst(reactant, product, calculator, images=args.images)
    out["neb_converged"] = bool(search.neb_converged)
    out["neb_barrier_ev"] = float(search.barrier_forward_ev)
    if search.ts is None or search.ts_positions is None or not search.ts.converged:
        return out | {"status": "TS refinement did not converge"}

    ts = reactant.copy()
    ts.positions = search.ts_positions
    ts.calc = calculator
    write(folder / "ts.xyz", ts)
    out["barrier_ev"] = float(ts.get_potential_energy()) - e_reactant
    frequencies = harmonic_frequencies(ts)
    out["imaginary_cm"] = [round(float(v), 1) for v in frequencies.imaginary]
    if frequencies.n_imaginary != 1:
        return out | {"status": f"{frequencies.n_imaginary} imaginary modes"}

    path = irc(ts, step=0.05, fmax=0.005, relax_ends=True)
    reached = []
    for name in ("reverse", "forward"):
        end = reactant.copy()
        end.positions = getattr(path, f"{name}_minimum_positions")
        reached.append(identify(end, broken, formed))
    out["irc_ends"] = reached
    confirmed = sorted(reached) == ["product", "reactant"]
    return out | {"status": "confirmed" if confirmed else "IRC reaches " + " and ".join(reached)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="Folder written by enumerate_yarp.py")
    parser.add_argument("--model", default=str(foundation_model()),
                        help="MACE model file (default: MACE-MP-0 small)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--images", type=int, default=7)
    parser.add_argument("--fmax-minimum", type=float, default=1e-3)
    parser.add_argument("--only", nargs="*", help="Candidate ids to run (default: all)")
    parser.add_argument("--threads", type=int, help="Cap on CPU threads (default: all cores)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    from mace.calculators import MACECalculator

    calculator = MACECalculator(model_paths=args.model, device=args.device, default_dtype="float64")
    summary = json.loads((args.run / "candidates.json").read_text())
    results = []
    for entry in summary["candidates"]:
        if not entry["geometry"] or (args.only and entry["id"] not in args.only):
            continue
        try:
            result = run_pair(args.run / entry["id"], entry, calculator, args)
        except Exception as error:  # one bad candidate must not stop the screen
            result = {"id": entry["id"], "product_smiles": entry["product_smiles"],
                      "status": f"failed: {type(error).__name__}: {error}"}
        results.append(result)
        barrier = result.get("barrier_ev")
        print(f"{result['id']}  {result['product_smiles']:<24} "
              f"barrier {'' if barrier is None else f'{barrier:.2f} eV':<8} {result['status']}",
              flush=True)

    report = {
        "reactant_smiles": summary["reactant_smiles"],
        "model": args.model,
        "results": results,
    }
    (args.run / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
