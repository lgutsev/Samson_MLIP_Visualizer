"""A small charge/spin MACE on the Fe₂XY frames, from scratch, on the laptop GPU.

    <defects env>/python train_fe2_spin.py [--epochs 60] [--seeds 1]

A preview of ClusterMLIP's Fe16 v1 model at molecule size: does a total-spin
embedding learn the UBPW91 spin ladders that UMA and the spin-blind foundation
models miss? Data: the Fe₂XY frames of ``dataset_spin_v0`` in its own grouped
split (whole source structures per split, so a test chain was never seen),
without the frames whose UBPW91 force RMS exceeds 5 eV/Å (stuck SCF roots,
e.g. the 118–134 eV/Å Fe₂OH/Fe₂NO frames). Note the caveat in ClusterMLIP's
INSTRUCTIONS.md: this collection predates the re-collect of step 3.

Architecture: MACE 64x0e+64x1o, r_max 5 Å, two interactions, with the
categorical ``total_spin`` (multiplicity) and ``total_charge`` graph embeddings
ClusterMLIP uses (``--embedding_specs``, ``--use_embedding_readout``). E0s are a
least-squares fit of the training energies to the composition (what
``--E0s=average`` does). float64, as ClusterMLIP locks it.

Output: ``WORK/fe2_spin/`` (data, runs, ``fe2_spin_seed1.model``). Then
``run_mace.py fe2-spin-mace WORK/fe2_spin/seed1/fe2_spin_seed1.model`` and
``analyze.py`` (which scores this model on test chains only).
"""

import argparse
import json
from collections import Counter

import numpy as np
from ase.io import read, write

from common import DATA, WORK, family

from samson_mlip_visualizer.training import TrainingSpec, train_local

OUT = WORK / "fe2_spin"
OUTLIER_RMS = 5.0
# Each graph embedding reads atoms.info[spec["key"]], and mace-torch 0.3.16 lets
# that key (default: the embedding's own name) override --total_spin_key and
# --total_charge_key. Without "key" the first run read the absent
# info["total_spin"] and trained spin-blind without any error ("total_spin: 0"
# in the data counts of the log).
EMBEDDING = {
    "total_spin": {"type": "categorical", "per": "graph", "in_dim": 1, "emb_dim": 64,
                   "num_classes": 101, "offset": 0, "key": "spin"},
    "total_charge": {"type": "categorical", "per": "graph", "in_dim": 1, "emb_dim": 64,
                     "num_classes": 201, "offset": 100, "key": "charge"},
}


def split(name):
    """Fe₂XY frames of one split, outliers dropped, the provenance blob removed."""
    frames = []
    for atoms in read(DATA.with_name(f"{name}.extxyz"), ":"):
        if family(atoms.info["formula"]) != "Fe2":
            continue
        rms = np.sqrt((atoms.arrays["REF_forces"] ** 2).sum(1).mean())
        if rms > OUTLIER_RMS:
            continue
        atoms.info = {k: atoms.info[k] for k in
                      ("record_id", "parent_record_id", "formula", "charge", "spin",
                       "multiplicity", "REF_energy")}
        frames.append(atoms)
    return frames


def fit_e0s(frames):
    elements = sorted({int(z) for a in frames for z in a.numbers})
    counts = np.array([[np.sum(a.numbers == z) for z in elements] for a in frames], float)
    energy = np.array([a.info["REF_energy"] for a in frames])
    e0, *_ = np.linalg.lstsq(counts, energy, rcond=None)
    residual = energy - counts @ e0
    return dict(zip(elements, e0.tolist())), float(np.abs(residual).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    sets = {}
    for name in ("train", "valid", "test"):
        path = data / f"{name}.extxyz"
        if not path.is_file():
            write(path, split(name))
        sets[name] = read(path, ":")
    e0s, spread = fit_e0s(sets["train"])
    parents = {n: {a.info["parent_record_id"] for a in f} for n, f in sets.items()}
    overlap = (parents["train"] & parents["test"]) | (parents["train"] & parents["valid"])
    if overlap:
        raise SystemExit(f"{len(overlap)} source structures in both train and valid/test")
    summary = {
        "frames": {n: len(f) for n, f in sets.items()},
        "source_structures": {n: len(p) for n, p in parents.items()},
        "multiplicities": dict(sorted(Counter(int(a.info["multiplicity"])
                                              for a in sets["train"]).items())),
        "e0s_eV": {str(z): e for z, e in e0s.items()},
        "e0s_fit_mae_eV": spread,
    }
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    spec = TrainingSpec(
        name="fe2_spin", foundation="", mode="scratch",
        train_file=str(data / "train.extxyz"), seeds=tuple(args.seeds),
        epochs=args.epochs, lr=0.01, batch_size=32,
        energy_weight=10.0, forces_weight=100.0, ema_decay=0.99,
        e0s=e0s,
        architecture={"hidden_irreps": "64x0e+64x1o", "r_max": 5.0,
                      "num_interactions": 2, "correlation": 3, "max_ell": 3},
        extra=(f"--valid_file={data / 'valid.extxyz'}",
               f"--test_file={data / 'test.extxyz'}",
               "--total_charge_key=charge", "--total_spin_key=spin",
               f"--embedding_specs={json.dumps(EMBEDDING, separators=(',', ':'))}",
               "--use_embedding_readout=True", "--scaling=rms_forces_scaling"),
        card={"reference": "UBPW91 (Gaussian 09, ClusterMLIP Warehouse 2)",
              "data": "dataset_spin_v0, Fe2XY frames, grouped split, force RMS <= 5 eV/A",
              "inputs": "atoms.info['charge'], atoms.info['spin'] = multiplicity"},
    )
    for run in train_local(spec, OUT, parallel=False):
        print(f"seed {run.seed}: {'ok' if run.ok else 'FAILED'} in {run.seconds / 60:.0f} min,"
              f" log {run.log}")


if __name__ == "__main__":
    main()
