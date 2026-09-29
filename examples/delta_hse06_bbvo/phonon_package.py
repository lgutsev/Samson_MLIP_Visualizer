"""PBE+U phonons of cubic Ba₂BiVO₆, and cheap stabilization tests. For LONI; nothing is submitted.
-> phonons/ (also hpc_smoke_tests/14_vasp_bbvo_phonons)

Smoke test 08 found that a symmetry-free PBE+U relaxation from a 0.05 Å rattle
lowers the cubic 40-atom cell by 48.6 meV/f.u. at fixed volume (V and Bi move
off-center, 3 short + 3 long bonds). That is strong evidence that Fm-3m is not a
local minimum, but only phonons show which modes are unstable, at which
wavevector, and how many. This package writes finite-displacement single points
(phonopy, ±0.01 Å) at the same settings as the campaign (MP POTCARs,
U(V) = 3.25 eV, 520 eV, PREC = Accurate, LREAL = .FALSE., ISYM = 0), with
EDIFF = 1e-8:

- ``sc40``: the 40-atom conventional cell (commensurate with Γ and X): polar
  off-centering and all octahedral tilt patterns of a rock-salt double
  perovskite (a⁻a⁻a⁻ and a⁰a⁰c⁻ at Γ, a⁰a⁰c⁺ at X);
- ``sc40k4``: the same displacements on a 4×4×4 mesh (k convergence of the
  unstable frequencies);
- ``sc80``: 2×2×2 primitive cells (Γ, X, L);
- ``p0.99``, ``p0.98``, ``t1.01``: ``sc40`` at the lattice scaled by 0.99, 0.98
  (about 4 and 9 GPa) and 1.01: does pressure remove the instability?

and one relaxation job per composition (``stab/``): Ba₂Bi(V₁₋ₓMₓ)O₆, M = Nb, Ta,
x = 0, 0.25, 0.5, 1 in the 40-atom cell. Each relaxes cell and ions from the
unrattled cubic-derived cell, then rattles by 0.05 Å and relaxes the ions again
with symmetry off. The second step's energy drop is the composition's remaining
instability, to compare with the 48.6 meV/f.u. of the pristine cell.

The cubic lattice is the earlier PBE+U one (8.487 Å). At the campaign's settings
it has P = +2.3 kbar, about 0.05 % in the lattice constant, which is negligible
for the sign of a soft mode. Run in the ``defects`` environment (phonopy):

    PYTHONPATH=../../src micromamba run -n defects python phonon_package.py
"""

import json
import shutil
import sys

import numpy as np
from ase import Atoms
from ase.io import read
from common import KSPACING, WORK, grouped, primitive
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from samson_mlip_visualizer.vasp_labeling import (
    incar,
    kpoint_mesh,
    potcar_names,
    species_order,
    write_poscar,
)

HERE = __import__("pathlib").Path(__file__).parent
COPY = r"D:\MLIP_Work_Folder\hpc_smoke_tests\14_vasp_bbvo_phonons"
ACCOUNT, PARTITION = "loni_perovsk27", "workq"
MODULE, POTCARS = "vasp6/6.5.1-cpu", "/home/lgutsev/pot/potpaw_PBE"
SUPERCELLS = {"sc40": [[-1, 1, 1], [1, -1, 1], [1, 1, -1]], "sc80": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]}
PHONON_TAGS = {"EDIFF": "1E-8", "ADDGRID": ".FALSE.", "NCORE": 4, "KPAR": 4,
               "LWAVE": ".FALSE.", "LCHARG": ".FALSE."}

target = WORK / "phonons"
if target.exists():
    sys.exit(f"{target} exists; remove it to write the package again")


def to_phonopy(atoms):
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:],
                        scaled_positions=atoms.get_scaled_positions())


def to_ase(cell):
    return Atoms(cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)


def write_job(folder, atoms, level_tags, mesh=None, title=""):
    folder.mkdir(parents=True)
    write_poscar(folder / "POSCAR", atoms)
    mesh = mesh or kpoint_mesh(atoms, KSPACING)
    (folder / "KPOINTS").write_text(f"Gamma-centered {title}\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n",
                                    newline="\n")
    (folder / "POTCAR.names").write_text(" ".join(potcar_names(species_order(atoms))) + "\n", newline="\n")
    for name, tags in level_tags.items():
        text = incar("pbe_u", atoms, extra=tags)
        text = text.replace("single point", title or name)
        (folder / f"INCAR{name}").write_text(text, newline="\n")
    return mesh


singles, summary = [], {"sets": {}}
sets = {"sc40": ("sc40", 1.00, None), "sc40k4": ("sc40", 1.00, (4, 4, 4)), "sc80": ("sc80", 1.00, None),
        "p0.99": ("sc40", 0.99, None), "p0.98": ("sc40", 0.98, None), "t1.01": ("sc40", 1.01, None)}
for name, (sc, scale, mesh) in sets.items():
    unit = primitive(scale)  # grouped Ba, V, Bi, O; phonopy keeps the grouping in its supercells
    ph = Phonopy(to_phonopy(unit), supercell_matrix=SUPERCELLS[sc], primitive_matrix=np.eye(3))
    ph.generate_displacements(distance=0.01, is_plusminus=True)
    folder = target / name
    folder.mkdir(parents=True)
    ph.save(folder / "phonopy_disp.yaml")
    perfect = to_ase(ph.supercell)
    assert species_order(perfect) == species_order(grouped(perfect)), "species not grouped"
    used = None
    for i, cell in enumerate(ph.supercells_with_displacements, 1):
        job = folder / f"disp-{i:03d}"
        used = write_job(job, to_ase(cell), {"": PHONON_TAGS}, mesh, f"{name} displacement {i}")
        singles.append(str(job.relative_to(target)).replace("\\", "/"))
    summary["sets"][name] = dict(supercell=sc, lattice_scale=scale, atoms=len(perfect),
                                 displacements=len(ph.supercells_with_displacements), kmesh=used)
    print(f"{name}: {len(perfect)} atoms, {len(ph.supercells_with_displacements)} displacements, mesh {used}")

# stabilization: symmetric relaxation, then rattle and relax the ions with symmetry off
relaxes = []
poscars = HERE / "poscars"
comps = {"x0": "Ba2BiVO6_cubic40_MACE-MP-0",
         "Nb0.25": "Ba2Bi_V0.75Nb0.25_O6_x0.25_MACE-MP-0", "Nb0.5": "Ba2Bi_V0.5Nb0.5_O6_x0.5_MACE-MP-0",
         "Nb1": "Ba2BiNbO6_x1_MACE-MP-0", "Ta0.25": "Ba2Bi_V0.75Ta0.25_O6_x0.25_MACE-MP-0",
         "Ta0.5": "Ba2Bi_V0.5Ta0.5_O6_x0.5_MACE-MP-0", "Ta1": "Ba2BiTaO6_x1_MACE-MP-0"}
relax_tags = {"NSW": 300, "IBRION": 2, "EDIFFG": -0.01, "EDIFF": "1E-6", "NCORE": 4, "KPAR": 4,
              "LWAVE": ".FALSE.", "LCHARG": ".FALSE."}
for name, src in comps.items():
    atoms = grouped(read(poscars / src / "POSCAR"))
    job = target / "stab" / name
    write_job(job, atoms, {".1_sym": {**relax_tags, "ISIF": 3, "ISYM": 2},
                           ".2_rattle": {**relax_tags, "ISIF": 2, "ISYM": 0}}, None, f"{name} relaxation")
    relaxes.append(f"stab/{name}")
summary["stabilization"] = list(comps)

(target / "singlepoints.txt").write_text("\n".join(singles) + "\n", newline="\n")
(target / "relaxations.txt").write_text("\n".join(relaxes) + "\n", newline="\n")
HEADER = f"""#!/bin/bash
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --output=logs/%x_%A_%a.out
# Written by samson-mlip-visualizer (phonon_package.py). Check account, partition, and time before sbatch.
set -eo pipefail
module purge
module load {MODULE}
set -u
export SINGULARITYENV_OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
POTCARS="{POTCARS}"
"""
POTCAR = """: > POTCAR
for name in $(cat POTCAR.names); do cat "$POTCARS/$name/POTCAR" >> POTCAR; done
grep TITEL POTCAR | awk '{print $4}' > POTCAR.used
"""
(target / "run_singlepoints.slurm").write_text(HEADER.replace("#SBATCH --nodes=1", f"""#SBATCH --job-name=bbvo-phon
#SBATCH --array=1-{len(singles)}%20
#SBATCH --time=04:00:00
#SBATCH --nodes=1""") + f"""job=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" singlepoints.txt)
cd "$job"
if grep -q "General timing" OUTCAR 2>/dev/null; then echo "$job done"; exit 0; fi
{POTCAR}cp INCAR INCAR.used
srun vasp_std > vasp.out 2>&1
rm -f WAVECAR CHG CHGCAR POTCAR
""", encoding="utf-8", newline="\n")
(target / "run_relaxations.slurm").write_text(HEADER.replace("#SBATCH --nodes=1", f"""#SBATCH --job-name=bbvo-stab
#SBATCH --array=1-{len(relaxes)}
#SBATCH --time=24:00:00
#SBATCH --nodes=1""") + f"""job=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" relaxations.txt)
cd "$job"
{POTCAR}cp INCAR.1_sym INCAR
srun vasp_std > vasp_1_sym.out 2>&1
cp OUTCAR OUTCAR.1_sym; cp OSZICAR OSZICAR.1_sym; cp CONTCAR CONTCAR.1_sym
# a 0.05 A Gaussian rattle (fixed seed), Cartesian, on the relaxed cell
python3 - <<'PY'
import random
random.seed(1)
lines = open("CONTCAR.1_sym").read().splitlines()
scale = float(lines[1]); cell = [[float(x) * scale for x in lines[i].split()] for i in (2, 3, 4)]
n = sum(int(x) for x in lines[6].split()); start = 8 if lines[7].strip()[0] in "DdCc" else 9
out = lines[:start]
# Cartesian rattle converted to fractional (general cell): f = r A^-1
a, b, c = cell
det = (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]))
def frac(v):
    m = [[b[1]*c[2]-b[2]*c[1], a[2]*c[1]-a[1]*c[2], a[1]*b[2]-a[2]*b[1]],
         [b[2]*c[0]-b[0]*c[2], a[0]*c[2]-a[2]*c[0], a[2]*b[0]-a[0]*b[2]],
         [b[0]*c[1]-b[1]*c[0], a[1]*c[0]-a[0]*c[1], a[0]*b[1]-a[1]*b[0]]]
    return [sum(v[k] * m[k][j] for k in range(3)) / det for j in range(3)]
for line in lines[start:start + n]:
    f = [float(x) for x in line.split()[:3]]
    d = frac([random.gauss(0, 0.05) for _ in range(3)])
    out.append(" ".join(f"{{x + y:.10f}}" for x, y in zip(f, d)))
out[start - 1] = "Direct"
open("POSCAR", "w").write("\\n".join(out) + "\\n")
PY
cp INCAR.2_rattle INCAR
srun vasp_std > vasp_2_rattle.out 2>&1
cp OUTCAR OUTCAR.2_rattle; cp OSZICAR OSZICAR.2_rattle; cp CONTCAR CONTCAR.2_rattle
rm -f WAVECAR CHG CHGCAR POTCAR
""", encoding="utf-8", newline="\n")
(target / "logs").mkdir()
(target / "package.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
n40 = sum(v["displacements"] for k, v in summary["sets"].items() if v["atoms"] == 40)
n80 = summary["sets"]["sc80"]["displacements"]
(target / "README.md").write_text(f"""# PBE+U phonons of cubic Ba2BiVO6, and stabilization tests

Written by `examples/delta_hse06_bbvo/phonon_package.py`. Nothing was submitted.

Settings are the campaign's (MP POTCARs Ba_sv V_pv Bi O Nb_pv Ta_pv, U(V) = 3.25 eV,
520 eV, PREC = Accurate, LREAL = .FALSE., ISYM = 0 for displacements), with
EDIFF = 1e-8 for the forces. The cubic lattice is the earlier PBE+U 8.487 A
(P = +2.3 kbar at these settings).

| Set | Cell | Displacements (+/-0.01 A) | k-mesh | Question |
|---|---|---|---|---|
""" + "\n".join(f"| `{k}` | {v['atoms']} atoms, a x {v['lattice_scale']} | {v['displacements']} | {'x'.join(map(str, v['kmesh']))} | "
                + {"sc40": "unstable modes at Gamma and X", "sc40k4": "k convergence of sc40",
                   "sc80": "adds L", "p0.99": "compression ~4 GPa", "p0.98": "compression ~9 GPa",
                   "t1.01": "tension"}[k] + " |" for k, v in summary["sets"].items()) + f"""

`stab/<composition>`: relax cell and ions of Ba2Bi(V1-xMx)O6 (40 atoms, MACE-MP-0
cubic-derived start) with symmetry on (`INCAR.1_sym`, ISIF 3), then rattle by 0.05 A
and relax the ions with symmetry off (`INCAR.2_rattle`, ISIF 2). E(2) - E(1) per
formula unit is the remaining instability; x0 reproduces smoke test 08 (-48.6 meV/f.u.).
Needs python3 on the compute node (standard library only) for the rattle.

1. Copy the folder to LONI; check account/partition/time in the two .slurm files.
2. `sbatch run_singlepoints.slurm` ({len(singles)} array tasks: {n40} x 40 atoms, about
   2-4 min each; {n80} x 80 atoms, about 15-30 min each; roughly 5-10 node-hours in all).
   Rerunning skips finished tasks.
3. `sbatch run_relaxations.slurm` (7 tasks, about 2-4 h each, 15-30 node-hours).
4. Copy back everything except WAVECARs, then on the desktop:
   `PYTHONPATH=../../src micromamba run -n defects python phonon_analyze.py <this folder>`
   (frequencies at Gamma/X/L, dispersion, imaginary-mode character, modulated
   structures for a frozen-mode scan, and the stabilization table).
""", encoding="utf-8")
if shutil.os.path.exists(COPY):
    sys.exit(f"{COPY} exists; remove it to copy the package again")
shutil.copytree(target, COPY)
print(f"-> {target} and {COPY}: {len(singles)} single points, {len(relaxes)} relaxations")
