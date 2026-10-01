"""The two end points of N₂ + Fe₂O₄ → Fe₂O₂(NO)₂ from the UBPW91 chains, for SAMSON.

    python n2_to_no_endpoints.py

Among the Fe₂N₂O₄ frames of the Warehouse 2 chains, the lowest structures at
M = 5 are N₂ lying next to an Fe₂(μ-O)₂ core with one terminal oxo on each Fe
(N–N 1.107 Å, Fe···N 2.42 Å); another chain relaxed to the same core carrying two
linear nitrosyls (Fe–N 1.67, N–O 1.17 Å), 0.605 eV higher at UBPW91. Both come
from the relaxed end of a chain step, at q = 0, M = 5.

The product's atom order is Fe, Fe, O, O (bridging), N, O (on Fe 0), N, O (on
Fe 1). The reactant is reordered to match: the N nearest Fe 0 and Fe 0's
terminal O go to slots 4 and 5, the other N and Fe 1's terminal O to 6 and 7, so
a path search can interpolate atom by atom. Writes ``WORK/n2_no/reactant.xyz``
and ``product.xyz`` (SAMSON imports XYZ) and ``endpoints.json`` (energies of
every model on the two frames).
"""

import json

import numpy as np
from ase.io import write
from common import WORK, predictions, read_key_frames

OUT = WORK / "n2_no"
REACTANT, PRODUCT = 2736, 2828  # key-frame indices (M = 5 ends of their chain steps)


def reorder_reactant(atoms):
    symbols = np.array(atoms.get_chemical_symbols())
    d = atoms.get_all_distances()
    fe0, fe1 = np.where(symbols == "Fe")[0]
    oxygens, nitrogens = np.where(symbols == "O")[0], np.where(symbols == "N")[0]
    bridging = [o for o in oxygens if d[o, fe0] < 2.0 and d[o, fe1] < 2.0]
    terminal = {fe: next(o for o in oxygens if o not in bridging and d[o, fe] < 1.7)
                for fe in (fe0, fe1)}
    n_near = min(nitrogens, key=lambda n: d[n, fe0])
    n_far = next(n for n in nitrogens if n != n_near)
    order = [fe0, fe1, *bridging, n_near, terminal[fe0], n_far, terminal[fe1]]
    assert sorted(order) == list(range(len(atoms)))
    return atoms[order], order


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = read_key_frames()
    reactant, order = reorder_reactant(frames[REACTANT])
    product = frames[PRODUCT].copy()
    assert reactant.get_chemical_symbols() == product.get_chemical_symbols()
    for name, atoms in (("reactant", reactant), ("product", product)):
        clean = atoms.copy()
        clean.info = {}
        clean.arrays = {k: v for k, v in clean.arrays.items() if k in ("numbers", "positions")}
        write(OUT / f"{name}.xyz", clean)
    ref = {name: float(frames[i].info["REF_energy"]) for name, i in
           (("reactant", REACTANT), ("product", PRODUCT))}
    energies = {"UBPW91": ref["product"] - ref["reactant"]}
    for model in ("uma-s-1p2", "uma-s-1p1", "mace-mpa-0-medium", "mace-mp-0-small"):
        e = predictions(model)["energy"]
        energies[model] = float(e[PRODUCT] - e[REACTANT])
    summary = {"reactant_key_frame": REACTANT, "product_key_frame": PRODUCT,
               "reactant_chain": frames[REACTANT].info["chain"],
               "product_chain": frames[PRODUCT].info["chain"],
               "reactant_order": [int(i) for i in order], "charge": 0, "multiplicity": 5,
               "reaction_energy_eV_at_UBPW91_geometries": energies}
    (OUT / "endpoints.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
