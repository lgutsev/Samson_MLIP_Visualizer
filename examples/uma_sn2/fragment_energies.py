"""Isolated F⁻ and Cl⁻ at UMA's level, computed here, and a check that it fits UMA.

    python fragment_energies.py [CHECKPOINT ...]

UMA cannot evaluate a single atom: fairchem looks the energy up in UMA's table of
isolated-atom ωB97M-V/def2-TZVPD energies (``iso_atom_elem_refs.yaml``, computed
with ORCA for OMol25). The two ions this example needs are cheap to compute
directly, so this script does that with Psi4 at the same level:

1. ωB97M-V/def2-TZVPD single points on F⁻ and Cl⁻ (singlets, charge −1), and UMA's
   own table values for them from any checkpoint that carries the table (uma-s-1p2
   does; uma-s-1p1 does not). The table is the same OMol25 atom data for every UMA
   model, so it is preferred; the Psi4 values are the cross-check (they differ by
   9 and 22 meV for these two anions);
2. the check: for each UMA checkpoint (default uma-s-1p1 and uma-s-1p2),
   neutral CH₃Cl and CH₃F relaxed with UMA, then ωB97M-V/def2-TZVPD at those
   geometries. If UMA's absolute energies sit within a few meV of Psi4's, UMA and
   Psi4 share the energy scale, and the Psi4 ion energies can stand in for the
   table.

Run with SAMSON's Python (Psi4 and UMA each run in their own environment through
the backends). Six small DFT jobs, a few minutes on 8 threads. Results are cached
in ``<work>/uma/fragments_wb97mv.json``; ``evaluate_uma.py`` uses them when UMA's
table is missing.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sn2_f_ch3cl"))

from ase import Atoms  # noqa: E402
from ase.calculators.calculator import CalculationFailed  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from forgetting import MOLECULES, build  # noqa: E402
from sn2_common import WORK  # noqa: E402

from samson_mlip_visualizer.psi4_backend import Psi4Calculator, find_psi4  # noqa: E402
from samson_mlip_visualizer.uma_backend import UMACalculator  # noqa: E402

UMA_PYTHON = Path(os.environ.get("UMA_PYTHON", r"D:\MLIP_Work_Folder\envs\mlip\python.exe"))
UMA_DIR = Path(os.environ.get("UMA_DIR", r"D:\MLIP_Downloaded_Models\UMA"))
CHECKPOINTS = [Path(p) for p in sys.argv[1:]] or [UMA_DIR / "uma-s-1p1.pt",
                                                  UMA_DIR / "uma-s-1p2.pt"]
OUT = WORK / "uma" / "fragments_wb97mv.json"
LEVEL = "ωB97M-V/def2-TZVPD (Psi4)"


def psi4(charge):
    return Psi4Calculator(find_psi4(), method="wb97m-v", basis="def2-tzvpd", charge=charge,
                          threads=8, memory_mb=1900)


def main():
    results = json.loads(OUT.read_text()) if OUT.exists() else {"level": LEVEL, "ions": {},
                                                                "check": {}}

    def save():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))

    for name, symbol in (("F-", "F"), ("Cl-", "Cl")):
        if name not in results["ions"]:
            calc = psi4(-1)
            atoms = Atoms(symbol)
            atoms.calc = calc
            results["ions"][name] = float(atoms.get_potential_energy())
            calc.close()
            save()
        print(f"{name:4s} {LEVEL}: {results['ions'][name]:.6f} eV")

    # UMA's own table, from any checkpoint that carries it (uma-s-1p2 does, 1p1 not).
    table = results.setdefault("table", {})
    for checkpoint in CHECKPOINTS:
        if {"F-", "Cl-"} <= set(table):
            break
        anion = UMACalculator(UMA_PYTHON, model=str(checkpoint), charge=-1)
        try:
            for name, symbol in (("F-", "F"), ("Cl-", "Cl")):
                atoms = Atoms(symbol)
                atoms.calc = anion
                table[name] = float(atoms.get_potential_energy())
            table["from"] = checkpoint.stem
            save()
        except CalculationFailed:
            table.clear()  # no table in this checkpoint; try the next one
        anion.close()
    for name in ("F-", "Cl-"):
        if name in table:
            print(f"{name:4s} UMA's table ({table['from']}): {table[name]:.6f} eV; "
                  f"table − Psi4 {(table[name] - results['ions'][name]) * 1000:+.1f} meV")

    neutral_dft = None
    for checkpoint in CHECKPOINTS:
        check = results["check"].setdefault(checkpoint.stem, {})
        uma = None
        for name in ("CH3Cl", "CH3F"):
            if name in check:
                continue
            uma = uma or UMACalculator(UMA_PYTHON, model=str(checkpoint), charge=0)
            atoms = build(MOLECULES[name])
            atoms.calc = uma
            BFGS(atoms, logfile=None).run(fmax=0.005, steps=300)
            e_uma = float(atoms.get_potential_energy())
            neutral_dft = neutral_dft or psi4(0)
            probe = Atoms(atoms.numbers, atoms.positions)
            probe.calc = neutral_dft
            e_dft = float(probe.get_potential_energy())
            check[name] = {"uma_ev": e_uma, "wb97mv_ev": e_dft,
                           "uma_minus_wb97mv_mev": (e_uma - e_dft) * 1000,
                           "positions": atoms.positions.tolist()}
            save()
        if uma is not None:
            uma.close()
        for name, row in check.items():
            print(f"{checkpoint.stem}: {name:5s} UMA − {LEVEL} at UMA's geometry: "
                  f"{row['uma_minus_wb97mv_mev']:+.1f} meV")
    if neutral_dft is not None:
        neutral_dft.close()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
