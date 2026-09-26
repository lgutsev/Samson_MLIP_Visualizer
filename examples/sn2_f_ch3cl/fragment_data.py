"""The separated-fragment data the first AIMNet2 fine-tune lacked, labeled with ωB97X-D.

    python fragment_data.py

Run with SAMSON's Python after ``forgetting.py`` (for the ωB97X-D CH3Cl and CH3F
geometries). Writes ``finetune_aimnet2/fragments.extxyz`` (info: ``charge``,
``raw_energy``, ``tag``; arrays: ``REF_forces``):

- the free anions F- and Cl- (charge -1);
- CH3Cl and CH3F at their ωB97X-D minima, plus two rattled copies of each
  (σ 0.03 Å; charge 0);
- the two ion-dipole complexes pulled apart along the C-X axis, everything else
  rigid: F- from the reactant complex to r(C-F) = 3.0, 3.5, 4.0, 5.0, 6.5 Å,
  Cl- from the product complex to r(C-Cl) = 3.6, 4.2, 5.0, 6.5 Å (charge -1).

About 19 ωB97X-D calculations, each once (shared cache, keyed by charge).
"""

import json

import numpy as np
from ase import Atoms
from ase.io import read, write
from sn2_common import CL, WORK, C, CachedReference, F

OUT = WORK / "finetune_aimnet2" / "fragments.extxyz"
PULL_F = (3.0, 3.5, 4.0, 5.0, 6.5)
PULL_CL = (3.6, 4.2, 5.0, 6.5)


def pulled(atoms, halogen, distance):
    """``atoms`` with ``halogen`` moved along the C->halogen axis to ``distance``."""
    atoms = atoms.copy()
    axis = atoms.positions[halogen] - atoms.positions[C]
    atoms.positions[halogen] = atoms.positions[C] + distance * axis / np.linalg.norm(axis)
    return atoms


def main():
    OUT.parent.mkdir(exist_ok=True)
    cache = WORK / "wb97xd_cache.json"
    references = {-1: CachedReference(cache, charge=-1), 0: CachedReference(cache, charge=0)}
    relaxed = json.loads((WORK / "forgetting" / "results.json").read_text())["ωB97X-D/def2-TZVPD"]
    rng = np.random.default_rng(1)
    structures = [(Atoms("F", [[0, 0, 0]]), -1, "anion"), (Atoms("Cl", [[0, 0, 0]]), -1, "anion")]
    molecules = (("CH3Cl", ["C", "Cl", "H", "H", "H"]), ("CH3F", ["C", "F", "H", "H", "H"]))
    for name, symbols in molecules:
        atoms = Atoms(symbols, positions=relaxed[name]["positions"])
        structures.append((atoms, 0, "molecule"))
        structures += [(Atoms(symbols, atoms.positions + rng.normal(0, 0.03, (5, 3))), 0,
                        "molecule_rattle") for _ in range(2)]
    irc = read(WORK / "aimnet2_irc.extxyz", ":")
    reactant = max((irc[0], irc[-1]), key=lambda a: a.get_distance(C, F))
    product = min((irc[0], irc[-1]), key=lambda a: a.get_distance(C, F))
    structures += [(pulled(Atoms(reactant.numbers, reactant.positions), F, d), -1,
                    f"pulled_F_{d}") for d in PULL_F]
    structures += [(pulled(Atoms(product.numbers, product.positions), CL, d), -1,
                    f"pulled_Cl_{d}") for d in PULL_CL]
    images = []
    for atoms, charge, tag in structures:
        probe = atoms.copy()
        probe.calc = references[charge]
        energy, forces = probe.get_potential_energy(), probe.get_forces()
        image = Atoms(atoms.numbers, atoms.positions)
        image.info.update({"charge": charge, "raw_energy": float(energy), "tag": tag})
        image.arrays["REF_forces"] = np.asarray(forces)
        images.append(image)
        print(f"{tag:18s} charge {charge:+d}  E {energy:.6f} eV", flush=True)
    write(OUT, images)
    computed = sum(r.computed for r in references.values())
    print(f"{len(images)} structures, {computed} new ωB97X-D calculations -> {OUT}")


if __name__ == "__main__":
    main()
