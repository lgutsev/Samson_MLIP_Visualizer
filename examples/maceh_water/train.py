"""Train MACE-H on the training frames.

    python train.py [--sets dimer,trimer] [--epochs N] [--init RUN | --resume RUN/model.pkl]

Writes ``train/train.ini`` and runs MACE-H's ``deephe3-train.py`` in the MACE-H
Python; the run lands in ``train/<date>_water_<sets>`` (``best_model.pkl``,
``result.txt``). 10 % of the training frames are MACE-H's validation split;
the held-out sets are only used by evaluate.py.

- ``--sets dimer`` is round 1 (dimers only); the default ``dimer,trimer``
  trains on everything under ``processed/train``.
- ``--init RUN`` starts from the weights of a finished run with a fresh
  optimizer, epoch 0, and no best loss to beat (a fine-tune on new data).
- ``--resume RUN/model.pkl`` continues an interrupted run (epoch, optimizer, LR
  schedule, and best loss restored) into a new run folder.

The network is MACE-H's default (three MACE blocks, correlation order 3,
hidden 64x0e+64x1o+64x2e, l_max 4, which O d x O d blocks need) except for an
8 Å cutoff, which covers the atom pairs of hydrogen-bonded trimers.
"""

import argparse
import json
import shutil
from pathlib import Path

from common import (
    GRAPHS,
    MACEH_PYTHON,
    PROCESSED,
    SEED,
    TRAIN_DATA,
    TRAIN_DIR,
    run_maceh,
    write_ini,
)


def warm_start(run: Path, target: Path) -> Path:
    """A checkpoint folder holding ``run``'s best weights as epoch 0 with an
    infinite best loss, so MACE-H fine-tunes from it as a new training."""
    import subprocess

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run / "src", target / "src")
    (target / "tensorboard").mkdir(parents=True)
    (target / "tensorboard" / "info.json").write_text(json.dumps({"global_step": 0}))
    # torch lives in the MACE-H Python, not necessarily in this one.
    script = (
        "import sys, torch\n"
        "c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
        "c.update(epoch=0, val_loss=float('inf'))\n"
        "for name in ('model.pkl', 'best_model.pkl'):\n"
        "    torch.save(c, sys.argv[2] + '/' + name)\n"
    )
    subprocess.run([str(MACEH_PYTHON), "-c", script, str(run / "best_model.pkl"), str(target)],
                   check=True)
    return target / "model.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="dimer,trimer")
    parser.add_argument("--epochs", type=int, default=400)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--init", type=Path, help="a finished run to fine-tune from")
    start.add_argument("--resume", type=Path, help="model.pkl of the run to continue")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="default 2e-3, or 1e-3 with --init")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    args = parser.parse_args()
    sets = args.sets.split(",")
    if sets == ["dimer"]:
        data, name = PROCESSED["dimer_train"], "dimer_train"  # round 1
    elif sorted(sets) == ["dimer", "trimer"]:
        data, name = TRAIN_DATA, "dimer_trimer_train"
    else:
        raise SystemExit(f"--sets must be dimer or dimer,trimer, not {args.sets}")
    checkpoint = ""
    if args.init:
        target = args.train_dir / f"init_from_{args.init.name}"
        checkpoint = warm_start(args.init, target).as_posix()
    elif args.resume:
        checkpoint = args.resume.as_posix()
    lr = args.learning_rate or (1e-3 if args.init else 2e-3)
    config = write_ini(args.train_dir / "train.ini", {
        "basic": {"device": args.device, "dtype": "float", "save_dir": args.train_dir.as_posix(),
                  "additional_folder_name": "water_" + "_".join(sets), "simplified_output": True,
                  "seed": SEED, "use_new_hypp": bool(args.init), "checkpoint_dir": checkpoint,
                  "inference": False},  # read but missing from MACE-H's train defaults
        "data": {"graph_dir": "", "DFT_data_dir": "",
                 "processed_data_dir": data.as_posix(),
                 "save_graph_dir": (GRAPHS / name).as_posix(),
                 "target_data": "hamiltonian", "dataset_name": name,
                 "get_overlap": False, "radius": -1},
        "train": {"num_epoch": args.epochs, "batch_size": 4, "extra_validation": "[]",
                  "extra_val_test_only": True, "train_ratio": 0.9, "val_ratio": 0.1,
                  "test_ratio": 0, "min_lr": 3e-5},
        "hyperparameters": {"learning_rate": lr,
                            "scheduler_params": "(factor=0.5, cooldown=20, patience=40, "
                                                "threshold=0.05)"},  # no verbose= in torch>=2.7
        "network": {"cutoff_radius": 8.0},
    })
    print(f"Training MACE-H on {data} ({args.epochs} epochs), log in "
          f"{args.train_dir / 'train.log'}")
    run_maceh("deephe3-train.py", config, args.train_dir / "train.log")


if __name__ == "__main__":
    main()
