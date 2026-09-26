"""Does stock MACE-MP-0 small have its own SN2 stationary points?

    python stock_mace_stationary.py

From the AIMNet2 TS: P-RFO with an exact Hessian, frequencies, and (with one
imaginary mode) an IRC both ways. From the two ends of the AIMNet2 IRC: plain
relaxations. Writes ``stock_mace_stationary.json`` with what it found and,
only if the TS is a first-order saddle whose IRC reaches F-...CH3Cl and
FCH3...Cl-, the barrier and complex-to-complex energy for the diagram.
"""

import json

import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS
from sn2_common import CL, KCAL, WORK, C, F, stock

from samson_mlip_visualizer.reaction_path import irc
from samson_mlip_visualizer.ts import prfo_search
from samson_mlip_visualizer.vibrations import harmonic_frequencies


def describe(atoms):
    return {"r_CF": float(atoms.get_distance(C, F)), "r_CCl": float(atoms.get_distance(C, CL))}


def kind(geometry):
    """Which complex a geometry is: F bonded (C-F < 1.6 Å), or Cl bonded with F⁻
    held off at an ion-dipole distance (C-Cl < 2.1 Å and C-F > 2.3 Å; the CCSD(T)
    complex has 1.84 and 2.50 Å). A shoulder next to the TS (seen with stock
    MACE: C-F 2.02, C-Cl 1.92 Å) is neither."""
    if geometry["r_CF"] < 1.6:
        return "FCH3...Cl-"
    if geometry["r_CCl"] < 2.1 and geometry["r_CF"] > 2.3:
        return "F-...CH3Cl"
    return "neither"


def main():
    calc = stock()
    path = read(WORK / "aimnet2_irc.extxyz", ":")
    energies = [f.info["energy_ev"] for f in path]
    out = {}
    for name, frame in (("from_F-...CH3Cl", path[0]), ("from_FCH3...Cl-", path[-1])):
        atoms = Atoms(frame.numbers, frame.positions)
        atoms.calc = calc
        converged = BFGS(atoms, logfile=None).run(fmax=0.005, steps=500)
        geometry = describe(atoms)
        out[name] = {"converged": bool(converged), **geometry, "is": kind(geometry),
                     "energy_ev": float(atoms.get_potential_energy())}
    ts = Atoms(path[0].numbers, path[int(np.argmax(energies))].positions)
    ts.calc = calc
    search = prfo_search(ts, fmax=1e-3, exact_hessian=True, max_steps=300)
    frequencies = harmonic_frequencies(ts)
    out["ts"] = {"converged": bool(search.converged), "steps": search.steps, **describe(ts),
                 "n_imaginary": int(frequencies.n_imaginary),
                 "lowest_cm": float(frequencies.wavenumbers_cm[0]),
                 "energy_ev": float(ts.get_potential_energy())}
    if search.converged and frequencies.n_imaginary == 1:
        result = irc(ts, step=0.05, max_steps=400, fmax=0.005, relax_ends=True)
        ends = {}
        for side in ("reverse", "forward"):
            end = ts.copy()
            end.positions = getattr(result, f"{side}_minimum_positions")
            geometry = describe(end)
            ends[side] = {**geometry, "is": kind(geometry),
                          "energy_ev": float(getattr(result, f"{side}_minimum_ev"))}
        out["irc_ends"] = ends
        found = {e["is"]: e["energy_ev"] for e in ends.values()}
        if set(found) == {"F-...CH3Cl", "FCH3...Cl-"}:
            out["barrier_kcal"] = (out["ts"]["energy_ev"] - found["F-...CH3Cl"]) * KCAL
            out["complex_to_complex_kcal"] = (found["FCH3...Cl-"] - found["F-...CH3Cl"]) * KCAL
    out["walden_ts_found"] = "barrier_kcal" in out
    (WORK / "stock_mace_stationary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
