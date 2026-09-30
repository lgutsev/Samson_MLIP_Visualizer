# MACE-H on water clusters: learning the Kohn-Sham Hamiltonian from Psi4

[MACE-H](https://github.com/maurergroup/MACE-H) ([arXiv:2508.15108](https://arxiv.org/abs/2508.15108))
uses MACE's many-body messages to predict the Kohn-Sham Hamiltonian matrix in an
atomic-orbital basis, one block per atom pair, instead of an energy.
Diagonalizing the predicted H against the overlap S gives orbital energies,
gaps, and band structures without an SCF. MACE-H reads the DeepH-E3 data layout,
and its own converters cover OpenMX, ABACUS, and FHI-aims. None of those runs on
this laptop, so [`hamiltonian.py`](../../src/samson_mlip_visualizer/hamiltonian.py)
adds a Psi4 → DeepH converter, and this example trains on it.

**Status (2026-09-30): the pipeline works end to end and is verified; the
model shown is undertrained.** The run shared the 8 GB GPU with another job and
stalled after about 50 epochs. The numbers below are from the epoch-30
checkpoint, so they test the plumbing, not what MACE-H can do.

## Pipeline

| step | script | env | output (`D:\MLIP_Work_Folder\maceh_water`) |
|---|---|---|---|
| 1. frames | `make_frames.py` | `defects` (mace-torch) | MACE-MP-0 Langevin MD at 400 K: 208 H-bonded dimers (4 runs), 27 trimers |
| 2. labels | `label.py` | `envs\maceh` | Psi4 PBE/def2-SVP KS matrix + overlap per frame, ~1 s each → `processed\{dimer_train,dimer_test,trimer}` |
| 3. train | `train.py` | `envs\maceh` | MACE-H default network (3 blocks, ν = 3, l_max 4), 8 Å cutoff → `train\<date>_water_dimer` |
| 4. evaluate | `evaluate.py [--device cpu]` | `envs\maceh` | predicted H, diagonalized with the Psi4 S → `results.json`, `images/maceh_water.png` |

Run steps 2–4 with `PYTHONPATH=../../src`. The test sets are dimer MD run 3,
held out as a whole (80 frames), and the trimers, a composition never trained
on. Trimers are predicted in MACE-H's inference mode, with the graph built from
`overlaps.h5` alone, which is how a new structure without an SCF would be
handled.

## What was checked

- **The orbital order is correct.** Psi4 orders pure shells m = 0, +1, −1, …;
  DeepH uses OpenMX order (p: x y z; d: z², x²−y², xy, xz, yz). After the
  permutation, the H and S blocks of a rotated water molecule match the
  rotated blocks of the original to 1e-8 eV. Two independent checks show this:
  MACE-H's own `Rotate` code, and a Wigner-D fit in
  `tests/test_hamiltonian.py`. The fit covers the O d shell.
- **The DeepH round trip is exact.** Eigenvalues of the assembled
  `hamiltonians.h5` against `overlaps.h5` reproduce Psi4's orbital energies to
  3e-12 eV.

## Epoch-30 checkpoint (undertrained)

| MAE | train dimers | test dimers | trimers |
|---|---|---|---|
| H elements (meV) | 104 | 129 | 409 |
| valence orbital energies (meV) | 396 | 531 | 9 857 |
| HOMO / LUMO (meV) | 502 / 1 452 | 662 / 2 329 | 1 905 / 6 018 |
| HOMO–LUMO gap (meV) (reference gaps 4.0–6.4 eV) | 1 218 | 1 769 | 4 112 |

![Orbital energies and gaps, epoch-30 checkpoint](images/maceh_water.png)

Errors of about 0.1 eV per matrix element become errors of 1 eV or more in the
eigenvalues. def2-SVP on H-bonded clusters has near-linearly-dependent
functions, so S has small eigenvalues, and diagonalizing against S amplifies
errors in H. The spurious low LUMOs and the scattered trimer levels come from
this. Useful eigenvalues need H errors of a few meV. The training loss (MSE
over raw elements) is also dominated by the O 1s diagonal near −510 eV.

## Next steps

- Finish the 400-epoch training on a free GPU (or LONI `gpu2`), then rerun
  `evaluate.py`.
- Add more dimer frames (128 is small) and some trimers, if the goal is transfer
  to larger clusters.
- Consider a smaller or less diffuse basis, or subtracting the on-site
  isolated-molecule blocks, so the model learns a smaller residual.

## Environment notes

MACE-H needs e3nn 0.5, but mace-torch 0.3.16 pins 0.4.4. `envs\maceh` is
therefore a venv with `--system-site-packages` on top of `defects`: it reuses
that env's CUDA torch 2.11 and has its own e3nn 0.5.1, torch_geometric, pathos,
tensorboard, and wandb. No `torch_scatter` wheel exists for torch 2.11, so a
small shim that uses `index_add` sits in the venv's site-packages (an earlier
`scatter_reduce` version crashed with a CUDA illegal instruction). Two MACE-H
settings have to be passed explicitly: `[basic] inference = False` in the
train config, and no `verbose=` in the scheduler parameters (torch ≥ 2.7).

## Paused run (2026-09-30 11:08)

Training reached epoch 208 of 400 (best: epoch 207, val MSE 0.012 eV², LR
halved to 1e-3; the epoch-30 checkpoint evaluated above had 0.096). It stopped
on a CUDA error (`CUBLAS_STATUS_EXECUTION_FAILED`) while the GPU was shared with
four other MACE trainings. Continue with (add `--device cpu` if the GPU is busy;
about 60 s per epoch on 32 free threads):

```bash
PYTHONPATH=../../src python train.py --resume D:/MLIP_Work_Folder/maceh_water/train/2026-09-30_08-48-15_water_dimer/model.pkl
```
