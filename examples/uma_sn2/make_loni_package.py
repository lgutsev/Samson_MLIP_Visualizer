"""Write smoke test 12: the large UMA (uma-m-1p1) on F⁻ + CH₃Cl, for a LONI GPU node.

    python make_loni_package.py

uma-m-1p1 is a 10.7 GB checkpoint: too much for the laptop's free memory next to
everything else it runs, so it goes to the cluster (``gpu2``). Run after
``evaluate_uma.py`` for uma-s-1p1 (its stationary points are the starting
geometries, and its frames the control). Writes
``D:\\MLIP_Work_Folder\\hpc_smoke_tests\\12_uma_sn2_large`` with:

- ``data/labeled_frames.extxyz``: the 46 frames with ωB97X-D labels on the desktop
  (31 from the fine-tuned MACE IRC, the 15 r(C–F) scan frames);
- ``data/starts.extxyz``: uma-s-1p1's reactant complex, TS, product complex, and
  CH₃Cl and CH₃F guesses;
- ``run_uma_sn2.py`` (standalone: fairchem, ASE, sella) and ``run_uma.slurm``, an
  array over the checkpoints: task 1 uma-s-1p1 (the control: it must reproduce the
  laptop), task 2 uma-m-1p1;
- ``README.md``. Nothing is submitted from here.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "sn2_f_ch3cl"))

from ase import Atoms  # noqa: E402
from ase.io import read, write  # noqa: E402
from forgetting import MOLECULES, build  # noqa: E402
from sn2_common import WORK  # noqa: E402

OUT = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests\12_uma_sn2_large")
CONTROL = WORK / "uma" / "uma-s-1p1"
CHECKPOINTS = ["uma-s-1p1.pt", "uma-m-1p1.pt"]  # array task 1, 2

SLURM = """#!/bin/bash
#SBATCH --job-name=uma-sn2
#SBATCH --account=loni_perovsk27
#SBATCH --partition=gpu2
#SBATCH --array=1-{n}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by samson-mlip-visualizer (examples/uma_sn2/make_loni_package.py).
# For LONI QB4. Replace <UMA_DIR> (see README.md) before sbatch.
# Array task 1: uma-s-1p1, the control; task 2: uma-m-1p1.
# conda's activation scripts read unset variables: -u only after them
set -eo pipefail
module purge
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate /project/lgutsev/env/uma
set -u
cd "$SLURM_SUBMIT_DIR"
CHECKPOINTS=({checkpoints})
CHECKPOINT=${{CHECKPOINTS[$((SLURM_ARRAY_TASK_ID - 1))]}}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCHDYNAMO_DISABLE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python run_uma_sn2.py "<UMA_DIR>/$CHECKPOINT" "outputs/${{CHECKPOINT%.pt}}" --device cuda
"""

README = """# 12 · UMA uma-m-1p1 on F⁻ + CH₃Cl (GPU)

Written by `examples/uma_sn2/make_loni_package.py`. Runs Meta's large UMA model,
too big for the laptop (a 10.7 GB checkpoint), on the SN2 checks of
`examples/uma_sn2`, next to the small uma-s-1p1 as a control. Nothing here has
been submitted.

| Array task | Checkpoint | Why |
|---|---|---|
| 1 | `uma-s-1p1.pt` | control: must reproduce the laptop run |
| 2 | `uma-m-1p1.pt` | the large model |

Each task takes a few minutes on one GPU (`gpu2`, 64 GB memory for the large
checkpoint): energies and forces on 46 frames, two relaxations, a Sella TS
refinement with a finite-difference frequency check, and the fragments.

## Before `sbatch run_uma.slurm`

1. **An environment with fairchem-core, ASE, and sella**, next to the MACE env.
   Once, on a login node:

   ```bash
   source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
   conda create -p /project/lgutsev/env/uma python=3.12 -y
   conda activate /project/lgutsev/env/uma
   pip install fairchem-core sella
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   On the login node the last line may print `False`; it has to be `True` on a
   `gpu2` node, as the job's log shows. If it isn't, install the CUDA build of
   torch that matches the node's driver.
2. **One placeholder in `run_uma.slurm`:** `<UMA_DIR>`, the folder on the
   cluster holding `uma-s-1p1.pt` and `uma-m-1p1.pt` (you already copied UMA
   there). The rest is set for QB4: account `loni_perovsk27`, partition `gpu2`,
   conda from `/home/lgutsev/miniforge3`, and the environment above.
3. **Optional: `iso_atom_elem_refs.yaml`** (facebook/UMA, `references/`) in
   `<UMA_DIR>` or `<UMA_DIR>/references/`. uma-m-1p1 may not carry UMA's
   isolated-atom table, and without it the job can't evaluate a free F⁻ or Cl⁻ and
   `results.json` lists the error. That doesn't matter: the desktop check fills
   in those two ions from UMA's table, taken from uma-s-1p2
   (`examples/uma_sn2/fragment_energies.py`).

Then `sbatch run_uma.slurm`.

## Afterwards

Copy `outputs/` back into this folder on the desktop, plus `logs/` if a task
failed. Then, in SAMSON's Python, from `examples/uma_sn2`:
`python check_loni_results.py`. It prints:

- **PASS or FAIL for the control.** uma-s-1p1 on the GPU must match the laptop:
  energies on the 46 frames within 5 meV (float32 on a GPU against a CPU), and the
  barrier within 0.1 kcal/mol.
- **The uma-m-1p1 numbers** beside uma-s-1p1, uma-s-1p2, ωB97X-D and CCSD(T):
  the barrier, geometries, TS mode, errors against ωB97X-D on the labeled frames,
  and (with the references file) energies relative to F⁻ + CH₃Cl.
"""


def labeled_frames():
    """The frames the laptop compared against ωB97X-D, with their path and index."""
    images = []
    for key, source, coordinate in (
        ("mace_tuned_irc", WORK / "finetune" / "finetuned_irc.extxyz", "irc_arc"),
        ("scan", WORK / "scan" / "scan_frames.extxyz", "scan_distance"),
    ):
        frames = read(source, ":")
        with open(CONTROL / f"{key}_vs_wb97xd.csv", encoding="utf-8") as handle:
            indices = [int(row["frame"]) for row in csv.DictReader(handle)]
        for index in indices:
            frame = frames[index]
            images.append(Atoms(frame.numbers, frame.positions, info={
                "path": key, "frame": index, coordinate: float(frame.info[coordinate])}))
    return images


def starts():
    control = json.loads((CONTROL / "evaluation.json").read_text())
    ts = control["ts"]
    ends = ts["irc_ends"].values()
    reactant = max(ends, key=lambda e: e["r_CF"])
    product = min(ends, key=lambda e: e["r_CF"])
    numbers = [6, 1, 1, 1, 17, 9]
    images = [Atoms(numbers, point["positions"], info={"name": name})
              for name, point in (("reactant_complex", reactant), ("ts", ts),
                                  ("product_complex", product))]
    for name in ("CH3Cl", "CH3F"):
        molecule = build(MOLECULES[name])
        molecule.info["name"] = name
        images.append(molecule)
    return images


def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    # SLURM does not create logs/, and an empty folder can get lost in the copy
    (OUT / "logs" / "README.txt").write_text("SLURM writes one log per array task here.\n",
                                             newline="\n")
    frames = labeled_frames()
    write(OUT / "data" / "labeled_frames.extxyz", frames)
    write(OUT / "data" / "starts.extxyz", starts())
    shutil.copy(HERE / "loni" / "run_uma_sn2.py", OUT / "run_uma_sn2.py")
    slurm = SLURM.format(n=len(CHECKPOINTS), checkpoints=" ".join(CHECKPOINTS))
    (OUT / "run_uma.slurm").write_text(slurm, newline="\n")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(f"wrote {OUT} ({len(frames)} labeled frames)")


if __name__ == "__main__":
    main()
