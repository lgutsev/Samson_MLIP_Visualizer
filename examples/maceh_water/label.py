"""Psi4 PBE/def2-SVP Kohn-Sham matrices for every frame, as DeepH folders.

    PYTHONPATH=../../src python label.py        (an env with ASE and h5py)

Writes one folder per frame under ``processed/train/{dimer,trimer}`` and
``processed/test/{dimer,trimer,tetramer}`` (dimer MD run ``TEST_RUN`` is test).
Each ``info.json`` also keeps the Psi4 energy, occupied-orbital count, and
orbital energies (eV) for evaluate.py.
Frames already labeled are skipped, so the script can be restarted.
"""

import json
import time

from ase.io import read
from common import BASIS, FRAMES, METHOD, PROCESSED, TEST_RUN, TRAIN_DATA, psi4_calculator

from samson_mlip_visualizer.hamiltonian import write_deeph


def target_set(name, atoms):
    if name == "dimer":
        return "dimer_test" if atoms.info["run"] == TEST_RUN else "dimer_train"
    return {"trimer": "trimer_test", "trimer_train": "trimer_train",
            "tetramer": "tetramer_test", "hexamer_ring": "hexamer_ring_test",
            "hexamer_prism": "hexamer_prism_test"}[name]


def main():
    calc = psi4_calculator()
    start = time.time()
    for name, path in FRAMES.items():
        if not path.is_file():
            print(f"{path.name} missing, skipped (run make_frames.py)")
            continue
        for index, atoms in enumerate(read(path, ":")):
            folder = PROCESSED[target_set(name, atoms)] / f"{name}_{index:04d}"
            if (folder / "hamiltonians.h5").is_file():
                continue
            atoms.calc = calc
            atoms.get_potential_energy()
            r = calc.results
            write_deeph(folder, atoms, r["orbital_types"], hamiltonian=r["hamiltonian"],
                        overlap=r["overlap"], info={
                            "method": f"{METHOD}/{BASIS}", "energy_eV": r["energy"],
                            "n_occupied": r["n_occupied"],
                            "eigenvalues_eV": r["eigenvalues"].tolist()})
            print(f"{folder.name}: {r['energy']:.4f} eV, "
                  f"{time.time() - start:.0f} s so far", flush=True)
    calc.close()
    for name, folder in PROCESSED.items():
        n = sum(1 for _ in folder.glob("*/hamiltonians.h5"))
        print(f"{name}: {n} structures in {folder}")
    (TRAIN_DATA.parent / "labels.json").write_text(
        json.dumps({"method": METHOD, "basis": BASIS}))


if __name__ == "__main__":
    main()
