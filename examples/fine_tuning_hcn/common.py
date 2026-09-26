"""Shared pieces of the HCN <-> HNC fine-tune (MACE-MP-0 small -> PBE/def2-TZVP).

Paths come from environment variables, with the defaults used for the run
documented in ``docs/fine_tuning.md``:

- ``MACE_FOUNDATION``: the foundation model (default: MACE-MP-0 small in ~/.cache/mace);
- ``MACE_RUN_TRAIN``: mace-torch's training script (default: next to this Python);
- ``FINETUNE_DIR``: where data, models, and reports go (default: this folder).
"""

import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms

warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("FINETUNE_DIR", Path(__file__).parent))
FOUNDATION = os.environ.get(
    "MACE_FOUNDATION",
    str(Path.home() / ".cache" / "mace" / "20231210mace128L0_energy_epoch249model"),
)
_SCRIPTS = Path(sys.executable).parent / ("Scripts" if os.name == "nt" else "")
TRAIN = (
    os.environ.get("MACE_RUN_TRAIN")
    or shutil.which("mace_run_train")
    or str(_SCRIPTS / ("mace_run_train.exe" if os.name == "nt" else "mace_run_train"))
)
C, N, H = 0, 1, 2  # atom order in every structure here
PBE_BARRIER, PBE_REACTION = 1.997, 0.662  # Psi4 PBE/def2-TZVP, optimized minima and TS
TIMES = HERE / "times.json"


def mace(paths, device="cuda"):
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=paths, device=device, default_dtype="float64")


def pbe():
    from samson_mlip_visualizer.psi4_backend import Psi4Calculator, find_psi4

    return Psi4Calculator(find_psi4(), method="pbe", basis="def2-tzvp")


def ts_guess():
    """The MACE-MP-0 transition state, rounded: the starting point for every TS search."""
    t = np.radians(68.3)
    return Atoms(
        "CNH", positions=[[0, 0, 0], [0, 0, 1.204], [0, 1.208 * np.sin(t), 1.208 * np.cos(t)]]
    )


def shift():
    """E_PBE − E_MACE-MP-0 at the MACE HCN minimum. Psi4's all-electron energies and
    the foundation model's VASP-referenced ones differ by a constant; subtracting it
    puts the PBE labels on the foundation model's scale (one composition, so every
    relative energy is unchanged)."""
    return json.loads((HERE / "shift.json").read_text())["shift_ev"]


def labeled(atoms, energy, forces, tag):
    image = Atoms(atoms.get_chemical_symbols(), positions=atoms.positions)
    image.info["REF_energy"] = float(energy) - shift()
    image.info["tag"] = tag
    image.arrays["REF_forces"] = np.asarray(forces, float)
    return image


def label(frames, calc, tag):
    out = []
    for frame in frames:
        atoms = frame.copy()
        atoms.calc = calc
        out.append(labeled(atoms, atoms.get_potential_energy(), atoms.get_forces(), tag))
    return out


def record(stage, seconds):
    times = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    times[stage] = round(seconds, 1)
    TIMES.write_text(json.dumps(times, indent=1))


def train(train_file, name, seed, workdir, epochs=120):
    """Fine-tune the foundation model on ``train_file``: plain (single-head)
    fine-tuning, no replay of foundation data. Returns (model path, seconds)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        TRAIN,
        f"--name={name}",
        f"--foundation_model={FOUNDATION}",
        "--multiheads_finetuning=False",
        f"--train_file={train_file}",
        "--valid_fraction=0.1",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        "--E0s=foundation",
        "--loss=weighted",
        "--energy_weight=10",
        "--forces_weight=100",
        "--lr=0.005",
        "--batch_size=4",
        "--valid_batch_size=8",
        f"--max_num_epochs={epochs}",
        "--ema",
        "--ema_decay=0.99",
        "--default_dtype=float64",
        "--device=cuda",
        f"--seed={seed}",
        f"--model_dir={workdir}",
        f"--checkpoints_dir={workdir / 'checkpoints'}",
        f"--results_dir={workdir / 'results'}",
        f"--log_dir={workdir / 'logs'}",
        "--save_cpu",
    ]
    start = time.perf_counter()
    run = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    seconds = time.perf_counter() - start
    (workdir / f"{name}.stdout").write_text(run.stdout + run.stderr, encoding="utf-8")
    model = workdir / f"{name}.model"
    if run.returncode != 0 or not model.exists():
        tail = "\n".join((run.stdout + run.stderr).splitlines()[-25:])
        raise RuntimeError(f"training {name} failed:\n{tail}")
    return model, seconds


def write_model_card(model, configurations, round_):
    """``<model>.json``: recorded in the provenance of every run with the model."""
    Path(str(model) + ".json").write_text(
        json.dumps(
            {
                "fine_tuned_from": "MACE-MP-0 small (2023-12-10)",
                "fine_tuning": "plain (single head, no foundation replay), mace-torch",
                "trained_on": f"{configurations} HCN/HNC configurations along the "
                f"isomerization IRC (active-learning round {round_})",
                "reference": "PBE/def2-TZVP (Psi4), energies shifted to the foundation scale",
                "elements": "H, C, N only",
                "scope": "HCN <-> HNC specialist; not a general-purpose model",
            },
            indent=1,
        )
    )


def geo(a):
    return (
        f"r(C-H) {a.get_distance(C, H):.3f}  r(N-H) {a.get_distance(N, H):.3f}  "
        f"r(C-N) {a.get_distance(C, N):.3f}  H-C-N {a.get_angle(H, C, N):5.1f}"
    )
