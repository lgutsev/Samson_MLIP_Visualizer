"""Reference energies at the stationary points, with Psi4.

Run with the Python of the environment that has Psi4:

    python reference_energies.py [work folder] [threads]

Reads ``stationary_points.extxyz`` (from ``stationary_points.py``) and writes
``stationary_points_qm.json``: energies (Eh) at PBE/def2-TZVPD, ωB97M-V/def2-TZVPPD
and CCSD(T)/aug-cc-pVDZ (frozen core). Results are saved after every
calculation and reused on a rerun, so an interrupted run resumes.

Sized for a laptop: a few minutes in all. The small-basis CCSD(T) is a spot
check; the basis-set-limit references come from the literature (see README).
Keep Psi4's memory below 2 GiB on Windows: with Psi4 1.11 from conda-forge, a
larger setting makes the coupled-cluster codes fail with a spurious "not enough
memory" / "No memory left", even for 100 basis functions.
"""

import json
import os
import sys
from pathlib import Path

import psi4

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LEVELS = {
    "PBE/def2-TZVPD": ("pbe", "def2-tzvpd", {}),
    "wB97M-V/def2-TZVPPD": ("wb97m-v", "def2-tzvppd", {}),
    # Analytic gradients (Psi4 has none for wB97M-V's VV10 term, so its forces
    # cost 25 SCFs per six-atom frame): the candidate level for path forces.
    "wB97X-D/def2-TZVPD": ("wb97x-d", "def2-tzvpd", {}),
    "CCSD(T)/aug-cc-pVDZ": (
        "ccsd(t)", "aug-cc-pvdz", {"cc_type": "conv", "scf_type": "pk", "freeze_core": True}
    ),
}


def frames(path):
    """(name, charge, [(symbol, x, y, z)]) from ASE's extxyz, read without ASE."""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        n = int(lines[i])
        info = lines[i + 1]
        name = info.split("name=")[1].split()[0].strip('"')
        charge = int(info.split("charge=")[1].split()[0])
        atoms = [line.split()[:4] for line in lines[i + 2 : i + 2 + n]]
        yield name, charge, atoms
        i += n + 2


def main():
    out_path = WORK / "stationary_points_qm.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    os.chdir(WORK)  # Psi4 writes timer.dat, ijk.dat, ... into the working directory
    psi4.set_num_threads(THREADS)
    psi4.set_memory("1900 MB")
    psi4.core.set_output_file(str(WORK / "reference_energies_psi4.out"), False)
    for level, (method, basis, options) in LEVELS.items():
        for name, charge, atoms in frames(WORK / "stationary_points.extxyz"):
            if name in results.get(level, {}):
                continue
            psi4.core.clean_options()
            psi4.set_options({"basis": basis, "scf_type": "df", "reference": "rhf", **options})
            geometry = "\n".join(" ".join(atom) for atom in atoms)
            molecule = psi4.geometry(
                f"{charge} 1\n{geometry}\nunits angstrom\nsymmetry c1\nno_reorient\nno_com\n"
            )
            energy = psi4.energy(method, molecule=molecule)
            psi4.core.clean()
            results.setdefault(level, {})[name] = energy
            out_path.write_text(json.dumps(results, indent=1))
            print(f"{level:22s} {name:18s} {energy:.8f} Eh", flush=True)


if __name__ == "__main__":
    main()
