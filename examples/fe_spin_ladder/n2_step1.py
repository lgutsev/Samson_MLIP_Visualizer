"""Step 1 of N₂ on Fe₂O₄: the stationary points from the SAMSON scan, saved for DFT.

    <mlip env>/python n2_step1.py

In SAMSON (UMA-s-1p2, omol, q = 0, M = 5) a bond scan of r(N···O), from the
relaxed physisorbed N₂ (3.01 Å) to 1.20 Å, bracketed a first-order saddle; P-RFO
with the exact Hessian refined it (one imaginary mode, −557 cm⁻¹) and the IRC
connected it to the reactant on one side and to an Fe–O–N–N intermediate on the
other. SAMSON keeps the IRC as a path, not a model, so this script rebuilds the
intermediate: from the TS (``n2_no/step1_ts.xyz``, copied from SAMSON) it moves
along the O→N direction, relaxes with UMA, and checks that it lands where the IRC
did. Writes ``step1_intermediate.xyz`` and ``step1.json``, and appends the
barrier and the intermediate to ``functional_pairs.json`` so ``functional_check.py
run`` puts both through BPW91 and ωB97M-V.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from common import MEV, WORK  # noqa: E402

FOLDER = WORK / "n2_no"
UMA = Path(os.environ.get("UMA_DIR", r"D:\MLIP_Foundational_Models\UMA")) / "uma-s-1p2.pt"
O_TERMINAL, N_ATTACK = 4, 5  # SAMSON's atom order: Fe Fe O O O N O N
SAMSON_IRC = {"reactant": -79940.00887313928, "intermediate": -79939.09960740934,
              "intermediate_r": 1.2071656292718373}


def main():
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    calc = FAIRChemCalculator(pretrained_mlip.load_predict_unit(str(UMA), device="cpu"),
                              task_name="omol")
    reactant = read(FOLDER / "step1_reactant.xyz")
    ts = read(FOLDER / "step1_ts.xyz")
    energies = {}
    for name, atoms in (("reactant", reactant), ("ts", ts)):
        atoms.info = {"charge": 0, "spin": 5}
        atoms.calc = calc
        energies[name] = float(atoms.get_potential_energy())

    inter = ts.copy()
    inter.info = {"charge": 0, "spin": 5}
    bond = inter.positions[O_TERMINAL] - inter.positions[N_ATTACK]
    inter.positions[N_ATTACK] += 0.15 * bond / np.linalg.norm(bond)
    inter.positions[N_ATTACK + 2] += 0.15 * bond / np.linalg.norm(bond)  # the other N follows
    inter.calc = calc
    BFGS(inter, logfile=None).run(fmax=0.01, steps=2000)
    energies["intermediate"] = float(inter.get_potential_energy())
    r = float(inter.get_distance(O_TERMINAL, N_ATTACK))
    write(FOLDER / "step1_intermediate.xyz", inter)
    d = inter.get_all_distances()
    summary = {
        "model": "UMA-s-1p2 omol, q=0, M=5",
        "energies_rel_reactant_eV": {k: e - energies["reactant"] for k, e in energies.items()},
        "intermediate": {"r_ON": r, "r_NN": float(d[N_ATTACK, N_ATTACK + 2]),
                         "r_FeO": float(d[0, O_TERMINAL]), "r_FeN": float(d[0, N_ATTACK]),
                         "matches_samson_irc_end": bool(abs(r - SAMSON_IRC["intermediate_r"]) < 0.02
                                                        and abs(energies["intermediate"]
                                                                - SAMSON_IRC["intermediate"])
                                                            < 0.02)},
        "ts": {"r_ON": float(ts.get_distance(O_TERMINAL, N_ATTACK)),
               "r_NN": float(ts.get_distance(N_ATTACK, N_ATTACK + 2)),
               "imaginary_cm": -557.0},
    }
    (FOLDER / "step1.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    pairs_path = WORK / "functional_pairs.json"
    pairs = json.loads(pairs_path.read_text())
    symbols = reactant.get_chemical_symbols()
    for key, name, target in (("step1_barrier", "ts", ts),
                              ("step1_intermediate", "intermediate", inter)):
        pairs[key] = {
            "formula": "Fe2N2O4", "chain": f"UMA reactant -> UMA {name} (SAMSON scan)",
            "symbols": symbols, "positions": reactant.positions.tolist(),
            "positions_low": target.positions.tolist(), "M_high": 5, "M_low": 5,
            "UBPW91_gaussian_meV": None,
            "UMA_s_1p2_meV": MEV * (energies[name] - energies["reactant"]),
        }
    pairs_path.write_text(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    main()
