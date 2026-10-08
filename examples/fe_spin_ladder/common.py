"""Shared pieces of the Fe spin-ladder example: paths and the spin chains.

The data are ClusterMLIP's Warehouse 2 labels (``dataset_spin_v0``, UBPW91 Gaussian
frames of Fe₂XY molecules and Fe16/Fe16N₂). Each *chain* starts from one warehouse
structure at high spin M, relaxes it (step s00), then hands the relaxed geometry
to M − 2 (s01) and relaxes again, down the ladder. The last frame of one step and
the first frame of the next have the same geometry (checked for all 1,061
hand-offs), so each hand-off is a vertical spin gap at fixed geometry.

Everything big goes to ``WORK`` (``FE_SPIN_WORK``, default
``D:\\MLIP_Work_Folder\\fe_spin_ladder``).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get(
    "FE_SPIN_DATA",
    r"D:\MLIP_Work_Folder\Cluster_MLIP\FenOm_Warehouse2\dataset_spin_v0\all.extxyz"))
WORK = Path(os.environ.get("FE_SPIN_WORK", r"D:\MLIP_Work_Folder\fe_spin_ladder"))
HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"

KEY_FRAMES = WORK / "key_frames.extxyz"  # first and last frame of every step
MEV = 1000.0

# chain-sNN, then optional restart tags (-r01, -rf01, -rf01-rerun-r01), then the frame
_RECORD = re.compile(r"^(?P<chain>.*-ladder-m\d+-m\d+)-s(?P<step>\d+)(?:-r[a-z]*\d*)*"
                     r"__frame(?P<frame>\d+)$")


def parse_record(record_id: str) -> tuple[str, int, int]:
    """(chain id, step, frame index in the chain) of a dataset record."""
    match = _RECORD.match(record_id)
    if match is None:
        raise ValueError(f"Not a spin-ladder record: {record_id}")
    return match["chain"], int(match["step"]), int(match["frame"])


def family(formula: str) -> str:
    """``Fe2`` for the small molecules, ``Fe16`` for Fe16 and Fe16N₂."""
    return "Fe16" if formula.startswith("Fe16") else "Fe2"


def load_chains(path: Path = DATA) -> dict[str, list]:
    """All frames grouped by chain, sorted by frame index: {chain: [Atoms]}."""
    from ase.io import iread

    chains = defaultdict(list)
    for atoms in iread(path, index=":"):
        chain, step, frame = parse_record(atoms.info["record_id"])
        atoms.info.update(chain=chain, step=step, frame=frame)
        chains[chain].append(atoms)
    for frames in chains.values():
        frames.sort(key=lambda a: a.info["frame"])
    return dict(chains)


def key_frames(chains: dict[str, list]) -> list:
    """The first and last frame of every step, tagged ``role`` = start/end.

    A one-frame step (a hand-off that converged at once) gives one frame with
    role ``start_end``.
    """
    out = []
    for frames in chains.values():
        steps = defaultdict(list)
        for atoms in frames:
            steps[atoms.info["step"]].append(atoms)
        for step in sorted(steps):
            first, last = steps[step][0], steps[step][-1]
            if first is last:
                first.info["role"] = "start_end"
                out.append(first)
            else:
                first.info["role"] = "start"
                last.info["role"] = "end"
                out += [first, last]
    return out


def read_key_frames() -> list:
    from ase.io import read

    if not KEY_FRAMES.is_file():
        raise SystemExit(f"{KEY_FRAMES} is missing: run make_frames.py first")
    return read(KEY_FRAMES, ":")


def predictions(name: str) -> dict:
    """A model's energies/forces on the key frames: {"energy": (n,), "forces": [..]}."""
    path = WORK / "predictions" / f"{name}.npz"
    data = np.load(path, allow_pickle=True)
    return {"energy": data["energy"], "forces": list(data["forces"])}


def save_predictions(name: str, energy, forces, meta: dict) -> Path:
    folder = WORK / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npz"
    np.savez(path, energy=np.asarray(energy, float),
             forces=np.array([np.asarray(f, float) for f in forces], dtype=object))
    (folder / f"{name}.json").write_text(json.dumps(meta, indent=1))
    return path
