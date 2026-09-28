"""Is cubic (Fm-3m) Ba₂BiVO₆ a minimum? A check for LONI.  Nothing is submitted.
-> stability/ (also hpc_smoke_tests/08_vasp_bbvo_stability)

MACE-MP-0 says no: from a 0.02 Å rattle, the 40-atom cell at the PBE+U lattice
relaxes ~143 meV per formula unit lower (atoms move up to ~0.4 Å), and with the
cell free as well it goes much further (−600 meV/f.u., +23 % volume, V–O bonds of
1.73 Å and broken octahedra, like VO₄ tetrahedra). A relaxation that starts from
the symmetric cell never finds this: every force is zero by symmetry. Whether
the instability is real is a question for DFT, so this writes:

1. ``singlepoints/``: PBE+U and HSE06 on three 40-atom structures — cubic at the
   PBE+U lattice (8.487 Å); the same cell with the ions relaxed by MACE-MP-0 from
   a rattle; and the fully relaxed MACE-MP-0 structure. If PBE+U also puts the
   second and third below the first, the instability is real at the DFT level.
2. ``relax/``: one PBE+U relaxation of the rattled cubic cell with symmetry off
   (ISYM = 0; ions first, ISIF = 2; then cell and ions, ISIF = 3). This test does
   not depend on MACE-MP-0 at all.
"""

import shutil
import sys

import numpy as np
from ase.io import read
from ase.optimize import FIRE
from common import KSPACING, WORK, conventional, mace_mp0

from samson_mlip_visualizer.finetune import Selection, write_selection
from samson_mlip_visualizer.vasp_labeling import (
    VaspSlurmSettings,
    incar,
    kpoint_mesh,
    potcar_names,
    species_order,
    write_poscar,
    write_vasp_package,
)

COPY = r"D:\MLIP_Work_Folder\hpc_smoke_tests\08_vasp_bbvo_stability"
target = WORK / "stability"
if target.exists():
    sys.exit(f"{target} exists; remove it to write the check again")
calc = mace_mp0()
rng = np.random.default_rng(0)

cubic = conventional()  # PBE+U lattice and positions
cubic.info = {"group": "cubic"}
fixed_cell = conventional()
fixed_cell.positions += rng.normal(0, 0.02, fixed_cell.positions.shape)
fixed_cell.calc = calc
FIRE(fixed_cell, logfile=None).run(fmax=0.005, steps=3000)
fixed_cell.calc = None
fixed_cell.info = {"group": "distorted_fixed_cell"}
collapsed = read(WORK / "distorted_40_mace.extxyz")
collapsed.calc = None
collapsed.info = {"group": "distorted_relaxed_cell"}
for atoms in (cubic, fixed_cell, collapsed):
    image = atoms.copy()
    image.calc = calc
    atoms.info["mace_mp0_energy_eV"] = float(image.get_potential_energy())
e0 = cubic.info["mace_mp0_energy_eV"]
for atoms in (fixed_cell, collapsed):
    difference = 1000 * (atoms.info["mace_mp0_energy_eV"] - e0) / 4
    print(f"{atoms.info['group']}: MACE-MP-0 {difference:+.1f} meV/f.u. relative to cubic")

frames = [cubic, fixed_cell, collapsed]
selection = Selection()
for index in range(3):
    selection.add(index, "stability-check", None, float("inf"))
frames_path, manifest_path = write_selection(WORK / "_stability_frames", frames, selection,
                                             source="cubic vs MACE-MP-0-distorted Ba2BiVO6",
                                             model="MACE-MP-0 small")
write_vasp_package(target / "singlepoints", frames_path, manifest_path, kspacing=KSPACING,
                   incar_extra={"pbe_u": {"NCORE": 4, "KPAR": 4},
                                "hse06": {"NCORE": 4, "KPAR": 4}},
                   slurm=VaspSlurmSettings(account="loni_perovsk27", partition="workq",
                                           potcar_dir="/home/lgutsev/pot/potpaw_PBE",
                                           tasks_per_node=64,
                                           modules=("vasp6/6.5.1-cpu",),
                                           setup=("export SINGULARITYENV_OMP_NUM_THREADS=1",),
                                           run="srun vasp_std", time="24:00:00",
                                           max_parallel=None))

# the relaxation: rattled cubic cell, symmetry off, ions then cell
relax = target / "relax"
relax.mkdir(parents=True)
start = conventional()
start.positions += np.random.default_rng(1).normal(0, 0.05, start.positions.shape)
write_poscar(relax / "POSCAR", start)
mesh = kpoint_mesh(start, KSPACING)
(relax / "KPOINTS").write_text(f"Gamma-centered, spacing {KSPACING} 1/A\n0\nGamma\n"
                               f"{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n", newline="\n")
(relax / "POTCAR.names").write_text(" ".join(potcar_names(species_order(start))) + "\n",
                                    newline="\n")
for stage, isif in (("1_ions", 2), ("2_cell", 3)):
    text = incar("pbe_u", start, extra={"NSW": 300, "IBRION": 2, "ISIF": isif,
                                        "EDIFFG": -0.01, "NCORE": 4, "KPAR": 4,
                                        "LWAVE": ".FALSE."})
    text = text.replace("single point", f"relaxation, ISIF = {isif}")
    (relax / f"INCAR.{stage}").write_text(text, newline="\n")
(relax / "run_relax.slurm").write_text("""#!/bin/bash
#SBATCH --job-name=bbvo-stability
#SBATCH --account=loni_perovsk27
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=48:00:00
#SBATCH --output=relax_%j.out
# Written by samson-mlip-visualizer. Replace the placeholders before sbatch.
set -euo pipefail
module purge
module load vasp6/6.5.1-cpu
export SINGULARITYENV_OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
POTCARS="/home/lgutsev/pot/potpaw_PBE"
: > POTCAR
for name in $(cat POTCAR.names); do cat "$POTCARS/$name/POTCAR" >> POTCAR; done
cp INCAR.1_ions INCAR
srun vasp_std > vasp_1_ions.out 2>&1
cp OUTCAR OUTCAR.1_ions; cp OSZICAR OSZICAR.1_ions; cp CONTCAR CONTCAR.1_ions
cp CONTCAR POSCAR
cp INCAR.2_cell INCAR
srun vasp_std > vasp_2_cell.out 2>&1
cp OUTCAR OUTCAR.2_cell; cp OSZICAR OSZICAR.2_cell; cp CONTCAR CONTCAR.2_cell
""", encoding="utf-8", newline="\n")
fixed = 1000 * (fixed_cell.info["mace_mp0_energy_eV"] - e0) / 4
full = 1000 * (collapsed.info["mace_mp0_energy_eV"] - e0) / 4
(target / "README.md").write_text(f"""# Is cubic Ba2BiVO6 a minimum?

MACE-MP-0 finds lower-energy distortions of the cubic (Fm-3m) cell that a
symmetric relaxation cannot reach. Two checks at the DFT level:

1. `singlepoints/`: a normal labeling package (PBE+U, then HSE06) on three
   40-atom structures: `frame_0000` cubic at the PBE+U lattice, `frame_0001` the
   same cell with the ions relaxed by MACE-MP-0 from a rattle, `frame_0002` the
   fully relaxed MACE-MP-0 structure (cell 10.02 x 9.67 x 7.83 A). Fill in the
   placeholders in `run_vasp.slurm`, submit, and compare the energies (OSZICAR,
   or `collect_vasp_labels`). If 0001 and 0002 are lower than 0000 at the DFT
   level, the cubic structure is a saddle point.
2. `relax/`: PBE+U relaxation from the cubic cell rattled by 0.05 A, with
   ISYM = 0: first the ions at fixed cell (ISIF = 2), then everything (ISIF = 3).
   Fill in the placeholders in `run_relax.slurm` and submit. If the energy drops
   well below the cubic cell's and the octahedra distort, the instability is
   real; if it returns to cubic, MACE-MP-0 was wrong here.

MACE-MP-0 energies relative to cubic (meV per formula unit): fixed-cell
distortion {fixed:+.0f}, full relaxation {full:+.0f}.
""", encoding="utf-8")
if shutil.os.path.exists(COPY):
    sys.exit(f"{COPY} exists; remove it to copy the check again")
shutil.copytree(target, COPY)
print(f"-> {target} and {COPY}")
