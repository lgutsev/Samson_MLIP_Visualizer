"""Water dimer, trimer, and tetramer frames from MACE-MP-0 Langevin MD at 400 K.

    PYTHONPATH=../../src python make_frames.py      (an env with mace-torch)

MACE-MP-0 only supplies plausible thermal geometries here; the labels are Psi4
PBE (label.py). Writes ``dimer_frames.extxyz`` (4 runs x 80 frames),
``trimer_frames.extxyz`` (2 runs x 20, the trimer test set),
``trimer_train_frames.extxyz`` (4 more runs x 60, other seeds), and
``tetramer_frames.extxyz`` (2 runs x 30, never trained on), keeping only frames
whose molecules are still hydrogen-bonded (every water within 3.4 Å O-O of
another). Files that already exist are kept.
"""

import numpy as np
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from common import FRAMES, SEED, WORK, water_cluster
from mace.calculators import MACECalculator

from samson_mlip_visualizer.paths import foundation_model


def bonded(atoms):
    oxygens = atoms.positions[atoms.numbers == 8]
    distances = np.linalg.norm(oxygens[:, None] - oxygens[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return bool((distances.min(axis=1) < 3.4).all())


# name: (waters, MD runs, frames per run, seed offset of the runs)
RUNS = {"dimer": (2, 4, 80, 0), "trimer": (3, 2, 20, 0),
        "trimer_train": (3, 4, 60, 10), "tetramer": (4, 2, 30, 0)}


def run(n_waters, runs, frames, every, calc, offset=0):
    kept = []
    for r in range(offset, offset + runs):
        atoms = water_cluster(n_waters, seed=SEED + r)
        atoms.calc = calc
        FIRE(atoms, logfile=None).run(fmax=0.05, steps=300)
        MaxwellBoltzmannDistribution(atoms, temperature_K=400, rng=np.random.default_rng(r))
        md = Langevin(atoms, 0.5 * units.fs, temperature_K=400, friction=0.02,
                      rng=np.random.default_rng(100 + r))
        md.run(1000)  # equilibrate 0.5 ps
        for _ in range(frames):
            md.run(every)
            if bonded(atoms):
                frame = atoms.copy()
                frame.info.update(run=r, waters=n_waters)
                kept.append(frame)
    return kept


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    calc = MACECalculator(model_paths=str(foundation_model()), device="cuda",
                          default_dtype="float64")
    for name, (n, runs, frames, offset) in RUNS.items():
        if FRAMES[name].is_file():
            continue
        kept = run(n, runs, frames, every=50, calc=calc, offset=offset)
        write(FRAMES[name], kept)
        print(f"{name}: {len(kept)} of {runs * frames} frames kept -> {FRAMES[name]}")


if __name__ == "__main__":
    main()
