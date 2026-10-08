"""Where is the PBE+U minimum near the MACE-MP-0 structure? A LONI package.  Nothing is submitted.
-> stability_followup/

The first stability check (``stability_check.py``, LONI smoke test 08) showed that
cubic Ba₂BiVO₆ is a saddle point: the fully relaxed MACE-MP-0 structure is
448 meV/f.u. below cubic at PBE+U and 916 at HSE06. But PBE+U's own
symmetry-free relaxation from a rattled cubic cell stopped at only −88 meV/f.u.,
360 meV/f.u. above that structure, so it found a shallower minimum. This package
relaxes at PBE+U from the MACE-MP-0 structure itself (ions, then cell and ions,
ISYM = 0, same settings as the check) and ends with an HSE06 single point on the
result, all in one job (~8 h PBE+U + ~15 h HSE06 on one QB4 node).

Compare its final energies with ``vasp_results.json`` (the cubic single points
of the same settings): that gives the DFT distortion energy at a DFT geometry.
"""

import sys

from ase.io import read
from common import KSPACING, WORK

from samson_mlip_visualizer.vasp_labeling import (
    incar,
    kpoint_mesh,
    potcar_names,
    species_order,
    write_poscar,
)

target = WORK / "stability_followup"
if target.exists():
    sys.exit(f"{target} exists; remove it to write the package again")
target.mkdir(parents=True)
start = read(WORK / "distorted_40_mace.extxyz")  # frame_0002 of the check
write_poscar(target / "POSCAR", start)
mesh = kpoint_mesh(start, KSPACING)
(target / "KPOINTS").write_text(f"Gamma-centered, spacing {KSPACING} 1/A\n0\nGamma\n"
                                f"{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n", newline="\n")
(target / "POTCAR.names").write_text(" ".join(potcar_names(species_order(start))) + "\n",
                                     newline="\n")
# ions, then cell and ions twice (the second pass starts with a fresh plane-wave
# basis at the new volume, against Pulay stress); the last PBE+U stage keeps its
# WAVECAR to start HSE06, as in the labeling packages
for stage, isif in (("1_ions", 2), ("2_cell", 3), ("3_cell", 3)):
    text = incar("pbe_u", start, extra={"NSW": 300, "IBRION": 2, "ISIF": isif,
                                        "EDIFFG": -0.01, "NCORE": 4, "KPAR": 4,
                                        "LWAVE": ".TRUE." if stage == "3_cell" else ".FALSE."})
    (target / f"INCAR.{stage}").write_text(text.replace("single point",
                                                        f"relaxation, ISIF = {isif}"),
                                           newline="\n")
# the HSE06 single point on the relaxed cell, as in the labeling packages
(target / "INCAR.4_hse06").write_text(incar("hse06", start, extra={"NCORE": 4, "KPAR": 4}),
                                      newline="\n")
(target / "run_followup.slurm").write_text("""#!/bin/bash
#SBATCH --job-name=bbvo-followup
#SBATCH --account=loni_perovsk27
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=48:00:00
#SBATCH --output=followup_%j.out
# Written by samson-mlip-visualizer. PBE+U relaxation from the MACE-MP-0
# structure (ions, then cell), then HSE06 at the result. Outputs of each stage
# are kept with a suffix; a stage that is already done is skipped on resubmit.
set -eo pipefail
module purge
# 6.6.1 (personal license): bit-identical to 6.5.1 on 08's cubic frame (smoke 20),
# so the comparison with 08's single points holds
module load vasp6/6.6.1-cpu
set -u
export SINGULARITYENV_OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
POTCARS="/home/lgutsev/pot/potpaw_PBE"
: > POTCAR
for name in $(cat POTCAR.names); do cat "$POTCARS/$name/POTCAR" >> POTCAR; done
[ -f POSCAR.start ] || cp POSCAR POSCAR.start
for stage in 1_ions 2_cell 3_cell 4_hse06; do
  if [ -f "OUTCAR.$stage" ] && grep -q "General timing" "OUTCAR.$stage"; then
    cp "CONTCAR.$stage" POSCAR
    continue
  fi
  cp "INCAR.$stage" INCAR
  srun vasp_std > "vasp_$stage.out" 2>&1 || true
  cp OUTCAR "OUTCAR.$stage"; cp OSZICAR "OSZICAR.$stage"; cp CONTCAR "CONTCAR.$stage"
  grep -q "General timing" "OUTCAR.$stage" || { echo "stage $stage did not finish"; exit 1; }
  cp CONTCAR POSCAR
done
rm -f WAVECAR CHG CHGCAR
""", encoding="utf-8", newline="\n")
(target / "README.md").write_text("""# PBE+U relaxation from the MACE-MP-0 structure, then HSE06

Start: `POSCAR`, the fully relaxed MACE-MP-0 40-atom cell (frame_0002 of smoke
test 08; PBE+U -448 meV/f.u. and HSE06 -916 meV/f.u. relative to cubic).
`sbatch run_followup.slurm` runs four stages in one job: PBE+U ions (ISIF = 2),
PBE+U cell and ions (ISIF = 3) twice, the second from a fresh basis at the new
volume (Pulay stress), then an HSE06 single point on that geometry, started
from the last PBE+U WAVECAR.
Same settings as the labeling packages (520 eV, Materials Project POTCARs and U,
LREAL = .FALSE., ISYM = 0, Gamma-centered 3x3x3 mesh). Resubmitting skips stages
that finished. Bring back OUTCAR.*, OSZICAR.*, CONTCAR.* and the slurm log.
""", encoding="utf-8", newline="\n")
print(f"-> {target} (k-mesh {mesh})")
