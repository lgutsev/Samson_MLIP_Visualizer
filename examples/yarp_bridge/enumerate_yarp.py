"""Enumerate candidate products of one reactant with YARP and write them as
reactant/product pairs for ``run_qst.py``.

Runs in the YARP environment, not the visualizer's: YARP is imported from a clone
(``--yarp`` or ``YARP_HOME``), never copied here (its license forbids commercial use
without permission). Only YARP's graph enumeration and force-field clean-up are
used; none of its xTB/CREST/ORCA stages.

    python enumerate_yarp.py "CC=O" --n-break 2 --n-form 2 --out runs/acetaldehyde

Each product keeps the reactant's atom order, which is what a QST search needs.
An ``.xyz`` reactant is reordered by YARP's canonicalization, so always take the
reactant from the output folder, not the input file.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _use_env_libraries() -> None:
    """Make an un-activated conda env usable: MKL/Open Babel DLLs and Open Babel data."""
    prefix = Path(sys.prefix)
    library_bin = prefix / "Library" / "bin"
    if library_bin.is_dir():
        os.environ["PATH"] = f"{library_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        os.add_dll_directory(str(library_bin))
    if "BABEL_DATADIR" not in os.environ:
        for found in prefix.rglob("UFF.prm"):
            os.environ["BABEL_DATADIR"] = str(found.parent)
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="Reactant: a SMILES string or an .xyz file")
    parser.add_argument("--n-break", type=int, default=2)
    parser.add_argument("--n-form", type=int, default=2)
    parser.add_argument("--mode", choices=("concerted", "sequential"), default="concerted")
    parser.add_argument("--lewis-cutoff", type=float, default=0.0,
                        help="Keep products whose Lewis score is at most this (YARP default 0)")
    parser.add_argument("--fc-cutoff", type=float, default=2.0,
                        help="Largest total formal charge kept (YARP default 2)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yarp", type=Path, default=os.environ.get("YARP_HOME"),
                        help="Path of the YARP clone (default: YARP_HOME)")
    args = parser.parse_args()
    if args.yarp is None:
        parser.error("give --yarp or set YARP_HOME to the YARP clone")

    _use_env_libraries()
    sys.path.insert(0, str(Path(args.yarp).resolve()))
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source
    if source.lower().endswith(".xyz"):
        source = str(Path(source).resolve())
    os.chdir(args.out)  # YARP's Open Babel helper drops a temporary file in the cwd

    import numpy as np
    from ase import Atoms
    from ase.io import write
    from yarp import yarpecule
    from yarp.reaction.enum import enumerate_products
    from yarp.reaction.filters import filter_enum_products
    from yarp.reaction.generate_rxns import quick_geom_opt

    def atoms_of(molecule) -> Atoms:
        symbols = [element.capitalize() for element in molecule.elements]
        return Atoms(symbols, positions=np.asarray(molecule.geo, float))

    def bonds_of(molecule) -> set[tuple[int, int]]:
        i, j = np.nonzero(np.triu(molecule.adj_mat))
        return {(int(a), int(b)) for a, b in zip(i, j)}

    reactant = yarpecule(source, mode="smiles")
    reactant.get_smiles()
    raw = enumerate_products(reactant, args.n_break, args.n_form, mode=args.mode)
    kept = filter_enum_products(raw, l_cutoff=args.lewis_cutoff, fc_cutoff=args.fc_cutoff)

    reactant_bonds = bonds_of(reactant)
    # The reactant gets the same force-field clean-up as the products.
    cleaned = quick_geom_opt(reactant)
    reactant_atoms = atoms_of(cleaned if cleaned is not None else reactant)
    write("reactant.xyz", reactant_atoms)

    candidates = []
    for number, product in enumerate(kept):
        product.get_smiles()
        product_bonds = bonds_of(product)
        entry = {
            "id": f"rxn_{number:03d}",
            "product_smiles": product.canon_smi,
            "product_mapped_smiles": product.map_smi,
            "lewis_score": float(product.lewis._scores[0]),
            "broken": sorted(reactant_bonds - product_bonds),
            "formed": sorted(product_bonds - reactant_bonds),
        }
        optimized = quick_geom_opt(product)
        entry["geometry"] = optimized is not None
        if optimized is not None:
            folder = Path(entry["id"])
            folder.mkdir(exist_ok=True)
            write(folder / "reactant.xyz", reactant_atoms)
            write(folder / "product.xyz", atoms_of(optimized))
        candidates.append(entry)

    Path(".tmp.mol").unlink(missing_ok=True)
    summary = {
        "source": args.source,
        "reactant_smiles": reactant.canon_smi,
        "reactant_mapped_smiles": reactant.map_smi,
        "symbols": reactant_atoms.get_chemical_symbols(),
        "charge": int(reactant.q),
        "enumeration": {"n_break": args.n_break, "n_form": args.n_form, "mode": args.mode,
                        "lewis_cutoff": args.lewis_cutoff, "fc_cutoff": args.fc_cutoff},
        "raw_products": len(raw),
        "candidates": candidates,
    }
    Path("candidates.json").write_text(json.dumps(summary, indent=2))

    print(f"{reactant.canon_smi}: {len(raw)} enumerated, {len(kept)} after YARP's filters, "
          f"{sum(c['geometry'] for c in candidates)} with a geometry")
    for c in candidates:
        note = "" if c["geometry"] else "  (no force-field geometry, skipped)"
        print(f"  {c['id']}  {c['product_smiles']:<24} broken {c['broken']} formed {c['formed']}{note}")


if __name__ == "__main__":
    main()
