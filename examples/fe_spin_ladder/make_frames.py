"""Pick the key frames of every spin chain and write them to ``WORK``.

    python make_frames.py

Writes ``key_frames.extxyz`` (the first and last frame of each step, with the
UBPW91 energy and forces, charge, multiplicity, chain, step and role) and
``chains.json`` (per chain: formula, charge, multiplicities, frame counts).
"""

import json
from collections import Counter

from ase.io import write
from common import DATA, KEY_FRAMES, WORK, family, key_frames, load_chains


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    chains = load_chains(DATA)
    frames = key_frames(chains)
    for atoms in frames:
        # keep the header small: drop the long provenance blob
        atoms.info.pop("metadata", None)
        atoms.info["family"] = family(atoms.info["formula"])
    write(KEY_FRAMES, frames)

    summary = {}
    for chain, members in chains.items():
        first = members[0].info
        summary[chain] = {
            "formula": first["formula"], "charge": int(first["charge"]),
            "family": family(first["formula"]),
            "multiplicities": sorted({int(a.info["multiplicity"]) for a in members},
                                     reverse=True),
            "frames": len(members),
        }
    (WORK / "chains.json").write_text(json.dumps(summary, indent=1))
    steps = Counter(len(s["multiplicities"]) for s in summary.values())
    print(f"{len(chains)} chains, {len(frames)} key frames -> {KEY_FRAMES}")
    print("steps per chain:", dict(sorted(steps.items())))
    print("families:", Counter(a.info["family"] for a in frames))


if __name__ == "__main__":
    main()
