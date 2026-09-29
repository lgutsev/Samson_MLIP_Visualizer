"""UMA (omol task, charge- and spin-aware) on the key frames.

    <mlip env>/python run_uma.py [uma-s-1p2 ...]

Runs in the fairchem environment (``D:\\MLIP_Work_Folder\\envs\\mlip``), not
SAMSON's Python, and calls fairchem directly: thousands of small single points go
faster in one process than through the backend's worker protocol. Each frame gets
its own charge and multiplicity. Checkpoints come from ``UMA_DIR``
(``D:\\MLIP_Foundational_Models\\UMA``); results go to
``WORK/predictions/<checkpoint>.npz``, saved every 200 frames so an interrupted run
resumes.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np  # noqa: E402

from common import WORK, read_key_frames, save_predictions  # noqa: E402

UMA_DIR = Path(os.environ.get("UMA_DIR", r"D:\MLIP_Foundational_Models\UMA"))


def run(name: str, frames) -> None:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    partial = WORK / "predictions" / f"{name}.partial.npz"
    energy, forces = [], []
    if partial.is_file():
        data = np.load(partial, allow_pickle=True)
        energy, forces = list(data["energy"]), list(data["forces"])
        print(f"{name}: resuming at frame {len(energy)}")
    unit = pretrained_mlip.load_predict_unit(str(UMA_DIR / f"{name}.pt"), device="cpu")
    calc = FAIRChemCalculator(unit, task_name="omol")
    start = time.time()
    for i in range(len(energy), len(frames)):
        atoms = frames[i].copy()
        atoms.info = {"charge": int(frames[i].info["charge"]),
                      "spin": int(frames[i].info["multiplicity"])}
        atoms.calc = calc
        energy.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), float))
        if (i + 1) % 200 == 0:
            partial.parent.mkdir(parents=True, exist_ok=True)
            np.savez(partial, energy=np.array(energy),
                     forces=np.array(forces, dtype=object))
            print(f"{name}: {i + 1}/{len(frames)} ({time.time() - start:.0f} s)", flush=True)
    save_predictions(name, energy, forces, {
        "model": name, "task": "omol", "checkpoint": str(UMA_DIR / f"{name}.pt"),
        "device": "cpu", "frames": len(frames), "seconds": round(time.time() - start, 1)})
    partial.unlink(missing_ok=True)
    print(f"{name}: done in {time.time() - start:.0f} s")


def main():
    names = sys.argv[1:] or ["uma-s-1p2", "uma-s-1p1"]
    frames = read_key_frames()
    for name in names:
        if (WORK / "predictions" / f"{name}.npz").is_file():
            print(f"{name}: already done")
            continue
        run(name, frames)


if __name__ == "__main__":
    main()
