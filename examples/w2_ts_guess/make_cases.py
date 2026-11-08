"""Build the Fe2 pilot cases from Warehouse 2 key frames (writes ``cases/``).

    python make_cases.py   # needs FE_SPIN_WORK/key_frames.extxyz (fe_spin_ladder/make_frames.py)

No failed-saddle frames have been exported from ClusterMLIP yet, so each pilot
case is a pair of UBPW91 minima from the W2 spin ladders that differ by one
hydrogen shift, at the same charge and multiplicity, plus a stand-in for the
"best index-1 frame" of a failed Gaussian TS run: the IDPP midpoint between
them (geometry only, no model involved). The product is superimposed on the
reactant first so the interpolation carries no rigid rotation.
"""

import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.mep import NEB

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("FE_SPIN_WORK", r"D:\MLIP_Work_Folder\fe_spin_ladder"))
CASES = {
    # name: (reaction, reactant record, product record)
    "fe2h2_m9": ("terminal Fe-H -> Fe-H-Fe bridge",
                 "f856619a89beb62b2168-ladder-m9-m7-s00-rf01__frame000009",
                 "543d76937e436da98d57-ladder-m9-m3-s00__frame000010"),
    "fe2h2o2_m7": ("O-H -> Fe-H (H from a bridging OH to Fe)",
                   "94ba59a4c2a16976d6fe-ladder-m9-m5-s01__frame000033",
                   "268f4a154cc858438c0f-ladder-m9-m1-s01__frame000012"),
    "fe2h2o3_m7": ("O-H -> Fe-H (H from a bridging OH to Fe)",
                   "af9df47483f2b88c03a6-ladder-m9-m1-s01-rf01__frame000037",
                   "e3d9ea4c00a3c26108c1-ladder-m9-m3-s01__frame000044"),
}


def superimpose(mobile, target):
    """``mobile`` rotated/translated onto ``target`` (Kabsch, same atom order)."""
    a = mobile.positions - mobile.positions.mean(0)
    b = target.positions - target.positions.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    out = mobile.copy()
    out.positions = a @ (u @ np.diag([1.0, 1.0, d]) @ vt) + target.positions.mean(0)
    return out


def main():
    frames = {a.info["record_id"]: a for a in read(WORK / "key_frames.extxyz", ":")}
    for name, (reaction, r_id, p_id) in CASES.items():
        reactant, product = frames[r_id].copy(), frames[p_id].copy()
        assert reactant.get_chemical_symbols() == product.get_chemical_symbols()
        assert (reactant.info["charge"], reactant.info["multiplicity"]) == \
            (product.info["charge"], product.info["multiplicity"])
        product = superimpose(product, reactant)
        band = [reactant.copy(), reactant.copy(), product.copy()]
        NEB(band, method="improvedtangent").interpolate(method="idpp")
        guess = band[1]
        folder = HERE / "cases" / name
        folder.mkdir(parents=True, exist_ok=True)
        sources = {"reactant": frames[r_id], "product": frames[p_id]}
        for role, source in (("reactant", reactant), ("product", product), ("guess", guess)):
            atoms = Atoms(source.numbers, positions=source.positions)  # no REF_forces etc.
            atoms.info = {k: frames[r_id].info[k] for k in ("formula", "charge", "multiplicity")}
            atoms.info["name"] = f"{name}_{role}"
            atoms.info["reaction"] = reaction
            if role == "guess":
                atoms.info["source"] = "IDPP midpoint of the two W2 minima (stand-in TS frame)"
            else:
                src = sources[role]
                atoms.info["source"] = "W2 dataset_spin_v0 key frame, UBPW91 minimum"
                atoms.info["record_id"] = src.info["record_id"]
                atoms.info["REF_energy"] = src.info["REF_energy"]
            write(folder / f"{role}.extxyz", atoms, format="extxyz")
        print(name, reaction)


if __name__ == "__main__":
    main()
