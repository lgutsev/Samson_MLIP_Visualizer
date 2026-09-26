"""Relative energies of the stationary points, every method next to the literature.

    python compare.py [work folder]

Reads ``stationary_points_mlip.json`` and ``stationary_points_qm.json`` and prints
a Markdown table (kcal/mol, relative to F- + CH3Cl) plus the barrier from the
ion-dipole complex; writes the same numbers to ``comparison.json``. All methods
are evaluated at the AIMNet2 geometries; the literature values are at their own
optimized geometries.
"""

import json
import sys
from pathlib import Path

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
HARTREE_KCAL, EV_KCAL = 627.5095, 23.0605

# Classical energies (no ZPE) relative to F- + CH3Cl, kcal/mol.
LITERATURE = {
    # Szabó, Czakó, Nat. Commun. 6, 5972 (2015); Szabó, Császár, Czakó, Chem. Sci.
    # 4, 4362 (2013): all-electron CCSD(T)/CBS focal-point analysis with
    # post-CCSD(T) and relativistic corrections.
    "Czakó FPA": {"reactant_complex": -15.63, "ts": -12.24, "product_complex": -41.58,
                  "products": -31.87},
    # Parthiban, de Oliveira, Martin, J. Phys. Chem. A 105, 895 (2001): W1'.
    "W1' (Parthiban)": {"reactant_complex": -15.43, "ts": -12.54, "product_complex": -42.16,
                        "products": -32.65},
}
POINTS = ("reactant_complex", "ts", "product_complex", "products")
LABELS = {"reactant_complex": "F⁻···CH₃Cl", "ts": "Walden TS",
          "product_complex": "FCH₃···Cl⁻", "products": "CH₃F + Cl⁻"}


def relative(energies, scale):
    reactants = energies["CH3Cl"] + energies["F-"]
    rel = {name: (energies[name] - reactants) * scale
           for name in ("reactant_complex", "ts", "product_complex")}
    rel["products"] = (energies["CH3F"] + energies["Cl-"] - reactants) * scale
    return rel


def main():
    mlip = json.loads((WORK / "stationary_points_mlip.json").read_text())
    qm = json.loads((WORK / "stationary_points_qm.json").read_text())
    methods = {
        "AIMNet2": relative({k: v["aimnet2_ev"] for k, v in mlip.items()}, EV_KCAL),
        "GFN2-xTB": relative({k: v["xtb_ev"] for k, v in mlip.items()}, EV_KCAL),
        **{level: relative(values, HARTREE_KCAL) for level, values in qm.items()},
        **LITERATURE,
    }
    for values in methods.values():
        values["barrier"] = values["ts"] - values["reactant_complex"]
    columns = [*POINTS, "barrier"]
    header = ["Method", *(LABELS.get(c, "Barrier from F⁻···CH₃Cl") for c in columns)]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name, values in methods.items():
        print(f"| {name} | " + " | ".join(f"{values[c]:.1f}" for c in columns) + " |")
    (WORK / "comparison.json").write_text(json.dumps(methods, indent=1))


if __name__ == "__main__":
    main()
