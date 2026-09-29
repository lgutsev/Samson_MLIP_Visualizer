"""MACE models on the key frames: the spin-blind foundation models, or a trained model.

    <defects env>/python run_mace.py                 # MACE-MP-0 small, MACE-MPA-0 medium
    <defects env>/python run_mace.py NAME PATH.model # any MACE model file

Runs in float64 (Fe16 totals are ~-5.5e5 eV; float32 rounds them to 0.06 eV,
the size of the gaps being compared), on CUDA when there is one. A model trained
with a total-spin embedding (``train_fe2_spin.py``) reads ``charge`` and
``total_spin`` from ``atoms.info``, so both are set from each frame; the
foundation models ignore them and give one energy per geometry.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

from common import WORK, read_key_frames, save_predictions

warnings.filterwarnings("ignore")

FOUNDATION = {
    "mace-mp-0-small": Path.home() / ".cache/mace/20231210mace128L0_energy_epoch249model",
    "mace-mpa-0-medium": Path(r"D:\MLIP_Foundational_Models\mace\mace-mpa-0-medium.model"),
}


def run(name: str, model: Path, frames) -> None:
    import torch
    from mace.calculators import MACECalculator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(model_paths=str(model), device=device, default_dtype="float64")
    energy, forces = [], []
    start = time.time()
    for source in frames:
        atoms = source.copy()
        atoms.info = {"charge": int(source.info["charge"]),
                      "total_spin": int(source.info["multiplicity"]),
                      "spin": int(source.info["multiplicity"])}
        atoms.calc = calc
        energy.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), float))
    save_predictions(name, energy, forces, {
        "model": name, "path": str(model), "dtype": "float64", "device": device,
        "frames": len(frames), "seconds": round(time.time() - start, 1)})
    print(f"{name}: {len(frames)} frames in {time.time() - start:.0f} s")


def main():
    frames = read_key_frames()
    if len(sys.argv) == 3:
        jobs = {sys.argv[1]: Path(sys.argv[2])}
    else:
        jobs = FOUNDATION
    for name, model in jobs.items():
        if (WORK / "predictions" / f"{name}.npz").is_file() and len(sys.argv) != 3:
            print(f"{name}: already done")
            continue
        run(name, model, frames)


if __name__ == "__main__":
    main()
