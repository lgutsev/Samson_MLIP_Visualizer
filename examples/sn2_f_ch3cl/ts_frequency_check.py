"""ωB97X-D at the fine-tuned model's transition state: residual force and frequencies.

    python ts_frequency_check.py [work folder]

Run with SAMSON's Python after ``finetune_sn2.py``. Reads the TS frame of
``finetune/finetuned_irc.extxyz`` and computes, with ωB97X-D/def2-TZVPD
(finite-difference Hessian from analytic gradients, 37 gradients), the largest
force on it and its harmonic frequencies. Writes ``finetune/ts_frequency_check.json``.
"""

import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from samson_mlip_visualizer.psi4_backend import Psi4Calculator, find_psi4
from samson_mlip_visualizer.vibrations import harmonic_frequencies

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")


def main():
    frames = read(WORK / "finetune" / "finetuned_irc.extxyz", index=":")
    frame = frames[int(np.argmin([abs(f.info["irc_arc"]) for f in frames]))]
    ts = Atoms(frame.numbers, frame.positions)
    ts.calc = Psi4Calculator(find_psi4(), method="wb97x-d", basis="def2-tzvpd", charge=-1,
                             threads=8, memory_mb=1900)
    fmax = float(np.linalg.norm(ts.get_forces(), axis=1).max())
    frequencies = harmonic_frequencies(ts)
    result = {
        "reference": "wB97X-D/def2-TZVPD at the fine-tuned MACE TS",
        "max_force_ev_per_angstrom": fmax,
        "wavenumbers_cm": [round(float(w), 1) for w in frequencies.wavenumbers_cm],
        "n_imaginary": int(frequencies.n_imaginary),
    }
    (WORK / "finetune" / "ts_frequency_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
