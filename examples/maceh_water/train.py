"""Train MACE-H on the dimer training frames.

    python train.py [--epochs N] [--resume RUN/model.pkl]

Writes ``train/train.ini`` and runs MACE-H's ``deephe3-train.py`` in the MACE-H
Python; the run lands in ``train/<date>_water_dimer`` (``best_model.pkl``,
``result.txt``). 10 % of the training frames are MACE-H's validation split;
the held-out MD run and the trimers are only used by evaluate.py.
``--resume`` continues a run from its ``model.pkl`` (epoch, optimizer, and LR
schedule are restored) into a new run folder.

The network is MACE-H's default (three MACE blocks, correlation order 3,
hidden 64x0e+64x1o+64x2e, l_max 4, which O d x O d blocks need) except for an
8 Å cutoff, which covers every atom pair of a hydrogen-bonded trimer.
"""

import argparse
from pathlib import Path

from common import GRAPHS, PROCESSED, SEED, TRAIN_DIR, run_maceh, write_ini


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--resume", type=Path, help="model.pkl of the run to continue")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    args = parser.parse_args()
    config = write_ini(args.train_dir / "train.ini", {
        "basic": {"device": args.device, "dtype": "float", "save_dir": args.train_dir.as_posix(),
                  "additional_folder_name": "water_dimer", "simplified_output": True,
                  "seed": SEED, "use_new_hypp": False,
                  "checkpoint_dir": args.resume.as_posix() if args.resume else "",
                  "inference": False},  # read but missing from MACE-H's train defaults
        "data": {"graph_dir": "", "DFT_data_dir": "",
                 "processed_data_dir": PROCESSED["dimer_train"].as_posix(),
                 "save_graph_dir": (GRAPHS / "dimer_train").as_posix(),
                 "target_data": "hamiltonian", "dataset_name": "dimer_train",
                 "get_overlap": False, "radius": -1},
        "train": {"num_epoch": args.epochs, "batch_size": 4, "extra_validation": "[]",
                  "extra_val_test_only": True, "train_ratio": 0.9, "val_ratio": 0.1,
                  "test_ratio": 0, "min_lr": 3e-5},
        "hyperparameters": {"learning_rate": 0.002,
                            "scheduler_params": "(factor=0.5, cooldown=20, patience=40, "
                                                "threshold=0.05)"},  # no verbose= in torch>=2.7
        "network": {"cutoff_radius": 8.0},
    })
    print(f"Training MACE-H ({args.epochs} epochs), log in {args.train_dir / 'train.log'}")
    run_maceh("deephe3-train.py", config, args.train_dir / "train.log")


if __name__ == "__main__":
    main()
