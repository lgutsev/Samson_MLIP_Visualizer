"""Shared pieces of the MACE-H water example: a Kohn-Sham Hamiltonian model for
water dimers, tested on held-out dimers and on water trimers it never saw.

Paths come from environment variables:

- ``MACEH_WATER_DIR``: data, models, and results (default
  ``D:\\MLIP_Work_Folder\\maceh_water``);
- ``MACEH_DIR``: the MACE-H checkout (default ``D:\\MLIP_Work_Folder\\maceh\\MACE-H``);
- ``MACEH_PYTHON``: the Python with MACE-H (default the ``envs\\maceh`` venv).
"""

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
WORK = Path(os.environ.get("MACEH_WATER_DIR", r"D:\MLIP_Work_Folder\maceh_water"))
MACEH_DIR = Path(os.environ.get("MACEH_DIR", r"D:\MLIP_Work_Folder\maceh\MACE-H"))
MACEH_PYTHON = Path(os.environ.get(
    "MACEH_PYTHON", r"D:\MLIP_Work_Folder\envs\maceh\Scripts\python.exe"))
# Round 1 trained on dimers only; round 2 adds trimers from four more MD runs.
# Held out: dimer MD run TEST_RUN, the two original trimer runs, and tetramers.
FRAMES = {name: WORK / f"{name}_frames.extxyz"
          for name in ("dimer", "trimer", "trimer_train", "tetramer",
                       "hexamer_ring", "hexamer_prism")}
TRAIN_DATA = WORK / "processed" / "train"
PROCESSED = {
    "dimer_train": TRAIN_DATA / "dimer",
    "trimer_train": TRAIN_DATA / "trimer",
    "dimer_test": WORK / "processed" / "test" / "dimer",
    "trimer_test": WORK / "processed" / "test" / "trimer",
    "tetramer_test": WORK / "processed" / "test" / "tetramer",
    "hexamer_ring_test": WORK / "processed" / "test" / "hexamer_ring",
    "hexamer_prism_test": WORK / "processed" / "test" / "hexamer_prism",
}
SETS = tuple(PROCESSED)
TEST_RUN = 3
GRAPHS = WORK / "graphs"
TRAIN_DIR = WORK / "train"
EVAL_DIR = WORK / "eval"
METHOD, BASIS = "pbe", "def2-svp"
SEED = 7


def water_cluster(n, spacing=2.85, seed=SEED, shape="ring"):
    """``n`` water molecules on a ring, O-O about ``spacing`` Å apart, each turned
    at random; ``shape="prism"`` stacks two rings of n/2 instead (each water then
    has three neighbors). A starting point for MD, not an equilibrium structure."""
    from ase import Atoms
    from ase.build import molecule
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    layers = 2 if shape == "prism" else 1
    per_ring = n // layers
    radius = spacing / (2 * np.sin(np.pi / per_ring)) if per_ring > 1 else 0.0
    cluster = Atoms()
    for k in range(n):
        water = molecule("H2O")
        water.positions -= water.positions[0]
        water.positions = water.positions @ Rotation.random(random_state=rng).as_matrix().T
        angle = 2 * np.pi * (k % per_ring) / per_ring
        water.positions += [radius * np.cos(angle), radius * np.sin(angle),
                            spacing * (k // per_ring)]
        cluster += water
    return cluster


def hbond_partners(atoms, cutoff=3.3):
    """Per water (oxygen), how many other oxygens lie within ``cutoff`` Å."""
    oxygens = atoms.positions[atoms.numbers == 8]
    distances = np.linalg.norm(oxygens[:, None] - oxygens[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return (distances < cutoff).sum(axis=1)


def psi4_calculator(overlap_only=False):
    from samson_mlip_visualizer.hamiltonian import Psi4HamiltonianCalculator
    from samson_mlip_visualizer.psi4_backend import find_psi4

    return Psi4HamiltonianCalculator(find_psi4(), method=METHOD, basis=BASIS,
                                     overlap_only=overlap_only, threads=8, memory_mb=3000)


def write_ini(path, sections):
    """A MACE-H config file; keys left out take MACE-H's defaults."""
    from configparser import ConfigParser

    config = ConfigParser()
    config.optionxform = str  # MACE-H's keys are case-sensitive (DFT_data_dir)
    for section, values in sections.items():
        config[section] = {key: str(value) for key, value in values.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        config.write(handle)
    return path


def run_maceh(script, config, log):
    """Run one of MACE-H's ``deephe3-*.py`` scripts in its own Python."""
    import subprocess

    with open(log, "w", encoding="utf-8") as handle:
        subprocess.run([str(MACEH_PYTHON), str(MACEH_DIR / script), str(config), "-n", "8"],
                       cwd=WORK, stdout=handle, stderr=subprocess.STDOUT, check=True)


def latest_model():
    runs = sorted(p for p in TRAIN_DIR.glob("*") if (p / "best_model.pkl").is_file())
    if not runs:
        raise FileNotFoundError(f"No trained MACE-H model under {TRAIN_DIR}")
    return runs[-1]
