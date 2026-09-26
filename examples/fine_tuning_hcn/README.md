# Fine-tuning MACE-MP-0 for HCN ⇌ HNC

The scripts behind [`docs/fine_tuning.md`](../../docs/fine_tuning.md), which
describes the method, the timing, and the results:

1. `seed_data.py`: the MACE-MP-0 IRC, 29 frames of it and 58 rattled copies
   labeled with PBE/def2-TZVP (Psi4) -> `train_r0.extxyz`.
2. `active_learning.py`: fine-tunes a 3-model committee, finds the TS and IRC
   with it, checks it against PBE on held-out IRC frames, and adds the frames
   with the largest committee spread if another round is needed.
3. `tests.py LABEL MODEL...`: minima, TS, IRC, scan, benchmarks on held-out
   frames, and molecules the model was not trained on.

Outputs (data, models, reports) go to this folder, or to `FINETUNE_DIR`.
