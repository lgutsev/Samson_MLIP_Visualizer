"""Stationary points of F- + CH3Cl -> CH3F + Cl- with AIMNet2 (and GFN2-xTB energies).

Run with the Python of the environment that has aimnet (it also has ASE):

    python stationary_points.py [work folder]

Reads ``aimnet2_irc.extxyz`` (the AIMNet2 IRC written by the SAMSON bridge's irc
job, both ends relaxed) and writes ``stationary_points.extxyz``: the reactant
complex, the TS, and the product complex (IRC first frame, highest frame, last
frame), plus separately relaxed CH3Cl and CH3F and the bare halide anions.
Each frame carries its charge and the AIMNet2 and xTB energies (eV).
"""

import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from samson_mlip_visualizer.aimnet2_backend import AIMNet2Calculator  # noqa: E402
from samson_mlip_visualizer.xtb_backend import XTBCalculator, find_xtb  # noqa: E402

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
MODEL = r"D:\MLIP_Work_Folder\cache\aimnet\aimnet2_wb97m_d3_0.pt"


def aimnet(charge):
    return AIMNet2Calculator(sys.executable, model=MODEL, charge=charge)


def relaxed(atoms, charge):
    atoms = atoms.copy()
    atoms.calc = aimnet(charge)
    BFGS(atoms, logfile=None).run(fmax=0.005, steps=500)
    return atoms


def main():
    irc = read(WORK / "aimnet2_irc.extxyz", index=":")
    energies = [frame.info["energy_ev"] for frame in irc]
    ts = int(np.argmax(energies))
    frames = {
        "reactant_complex": irc[0],
        "ts": irc[ts],
        "product_complex": irc[-1],
        "CH3Cl": relaxed(Atoms("CClH3", [[0, 0, 0], [0, 0, 1.78], [1.03, 0, -0.36],
                                         [-0.515, 0.89, -0.36], [-0.515, -0.89, -0.36]]), 0),
        "CH3F": relaxed(Atoms("CFH3", [[0, 0, 0], [0, 0, 1.38], [1.03, 0, -0.36],
                                       [-0.515, 0.89, -0.36], [-0.515, -0.89, -0.36]]), 0),
        "F-": Atoms("F"),
        "Cl-": Atoms("Cl"),
    }
    charges = {"CH3Cl": 0, "CH3F": 0}
    xtb = find_xtb()
    out, table = [], {}
    for name, atoms in frames.items():
        atoms = Atoms(atoms.numbers, atoms.positions)
        charge = charges.get(name, -1)
        atoms.calc = aimnet(charge)
        e_aimnet = float(atoms.get_potential_energy())
        e_xtb = None
        if xtb is not None:
            atoms.calc = XTBCalculator(xtb, charge=charge)
            e_xtb = float(atoms.get_potential_energy())
        atoms.calc = None
        atoms.info.update({"name": name, "charge": charge, "aimnet2_ev": e_aimnet})
        if e_xtb is not None:
            atoms.info["xtb_ev"] = e_xtb
        out.append(atoms)
        table[name] = {"charge": charge, "aimnet2_ev": e_aimnet, "xtb_ev": e_xtb}
    write(WORK / "stationary_points.extxyz", out)
    (WORK / "stationary_points_mlip.json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
