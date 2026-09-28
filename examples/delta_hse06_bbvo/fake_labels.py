"""Dry run: synthetic labels in the collector's format, to test steps 3–4 before
LONI time is spent.  -> dry_run/labeled.extxyz

Not physics: "HSE06" is MACE-MP-0 plus a smooth pair term (Morse on every pair
within 5 Å) and a constant per atom; "PBE+U" is MACE-MP-0 plus a smaller pair
term. A correction trained on these should recover the pair term, which says
the pipeline (Δ labels with stress, training from scratch, the baseline found
from the model card, evaluation) works. It says nothing about the real HSE06
residual.
"""

from ase.calculators.morse import MorsePotential
from ase.io import read, write
from common import WORK, mace_mp0

base = mace_mp0()
terms = {"HSE06": (MorsePotential(epsilon=0.02, r0=2.2, rho0=4.0, rcut1=4.0, rcut2=5.0), -1.8),
         "PBEU": (MorsePotential(epsilon=0.005, r0=2.2, rho0=4.0, rcut1=4.0, rcut2=5.0), 0.0)}


def fake(frames):
    out = []
    for index, frame in enumerate(frames):
        atoms = frame.copy()
        atoms.calc = base
        e, f, s = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        labeled = frame.copy()
        labeled.info.update(frame=index, source="dry run (synthetic)", tag="dry-run")
        for prefix, (pair, per_atom) in terms.items():
            atoms.calc = pair
            extra = atoms.get_potential_energy() + per_atom * len(atoms)
            labeled.info[f"{prefix}_energy"] = e + extra
            labeled.info[f"{prefix}_stress"] = s + atoms.get_stress()
            labeled.arrays[f"{prefix}_forces"] = f + atoms.get_forces()
        out.append(labeled)
    return out


(WORK / "dry_run").mkdir(parents=True, exist_ok=True)
for source, target in (("frames.extxyz", "labeled.extxyz"),
                       ("doped_frames.extxyz", "labeled_doped.extxyz"),
                       ("dilute_frames.extxyz", "labeled_dilute.extxyz")):
    if (WORK / source).exists():
        out = fake(read(WORK / source, ":"))
        write(WORK / "dry_run" / target, out)
        print(f"{len(out)} synthetic frames -> {WORK / 'dry_run' / target}")
