"""Forgetting check: neutral molecules the tuned model never saw, geometries only.

    python forgetting.py

CH3F, CH3Cl and CH2F2 (charge 0) relaxed with stock MACE-MP-0 small, the
fine-tuned committee (BFGS, Fmax 0.005 eV/Å), and ωB97X-D/def2-TZVPD (Fmax
0.01 eV/Å: its forces are too noisy for tighter), then
C-F, C-Cl and C-H bond lengths compared. Energies are not compared: the tuned
model only saw the [CH3FCl]- anion, and MACE has no charge input. Writes
``forgetting/results.json`` after each relaxation; finished ones are skipped.
"""

import json

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from sn2_common import WORK, committee, psi4_reference, stock

OUT = WORK / "forgetting"
TETRAHEDRAL = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
LENGTH = {"H": 1.09, "F": 1.38, "Cl": 1.78}
MOLECULES = {"CH3F": ["F", "H", "H", "H"], "CH3Cl": ["Cl", "H", "H", "H"],
             "CH2F2": ["F", "F", "H", "H"]}


def build(substituents):
    positions = [[0.0, 0.0, 0.0]] + [LENGTH[s] * d for s, d in zip(substituents, TETRAHEDRAL,
                                                                  strict=True)]
    # A small fixed displacement: Psi4 1.11 fails on the exact idealized CH2F2
    # ("Unrecognized point group bits: 48") even with symmetry c1.
    positions = np.array(positions) + np.random.default_rng(0).normal(0, 0.005, (5, 3))
    return Atoms(["C", *substituents], positions=positions)


def bonds(atoms):
    lengths = {}
    for k, symbol in enumerate(atoms.get_chemical_symbols()[1:], 1):
        lengths.setdefault(f"C-{symbol}", []).append(float(atoms.get_distance(0, k)))
    return {bond: float(np.mean(values)) for bond, values in lengths.items()}


def main():
    OUT.mkdir(exist_ok=True)
    results_file = OUT / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    methods = {"MACE-MP-0 small": stock, "fine-tuned MACE": committee,
               "ωB97X-D/def2-TZVPD": lambda: psi4_reference(charge=0)}
    for method, factory in methods.items():
        calc = None
        for name, substituents in MOLECULES.items():
            if name in results.get(method, {}):
                continue
            calc = calc or factory()
            atoms = build(substituents)
            atoms.calc = calc
            # DFT forces are too noisy to reach 0.005 eV/Å (CH3Cl stalled for 300
            # steps with its bonds settled to 0.001 Å); 0.01 is plenty for bond lengths.
            dft = "ωB97X-D" in method
            optimizer = BFGS(atoms, logfile=None)
            converged = optimizer.run(fmax=0.01 if dft else 0.005, steps=100 if dft else 300)
            results.setdefault(method, {})[name] = {
                "bonds_A": bonds(atoms), "converged": bool(converged),
                "steps": optimizer.nsteps, "positions": atoms.positions.tolist(),
            }
            results_file.write_text(json.dumps(results, indent=1))
            print(method, name, results[method][name]["bonds_A"], flush=True)
    reference = results["ωB97X-D/def2-TZVPD"]
    print("| Molecule | Bond | MACE-MP-0 small | fine-tuned MACE | ωB97X-D |")
    print("|---|---|---|---|---|")
    for name in MOLECULES:
        for bond, value in reference[name]["bonds_A"].items():
            cells = [f"{results[m][name]['bonds_A'][bond]:.3f}"
                     for m in ("MACE-MP-0 small", "fine-tuned MACE")]
            print(f"| {name} | {bond} | {cells[0]} | {cells[1]} | {value:.3f} |")


if __name__ == "__main__":
    main()
