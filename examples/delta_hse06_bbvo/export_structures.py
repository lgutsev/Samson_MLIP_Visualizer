"""Write every Ba₂BiVO₆ structure of this example for coauthors.
-> poscars/<name>/POSCAR, poscars/README.md, frames/*.extxyz

One folder per structure with its POSCAR (VASP 5 format, direct coordinates,
species grouped; it opens in VASP, VESTA, and ASE), for:

- the PBE+U structure (primitive and 40-atom cubic), the starting point of all
  the work;
- the V-site substitution series relaxed with MACE-MP-0 (``doping.py``);
- the MACE-MP-0 distortions of the cubic cell (``stability_check.py``);
- one Nb or Ta in the 80-atom cell, relaxed with MACE-MP-0 (``dilute.py``);

and, in ``frames/``, the exact frame sets the LONI packages were written from
(``make_frames.py``, ``doping.py``, ``dilute.py``), since MD on a GPU does not
repeat bit for bit.
"""

import shutil

import numpy as np
from ase.io import read, write
from common import HERE, WORK, conventional, primitive

OUT = HERE / "poscars"
FRAMES = HERE / "frames"
OUT.mkdir(exist_ok=True)
FRAMES.mkdir(exist_ok=True)
rows = []


def poscar(name, atoms, comment, source):
    atoms = atoms.copy()
    atoms.calc = None
    path = OUT / name / "POSCAR"
    path.parent.mkdir(exist_ok=True)
    write(path, atoms, format="vasp", direct=True, sort=False)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([comment] + lines[1:]) + "\n")
    lengths = " × ".join(f"{v:.3f}" for v in atoms.cell.lengths())
    rows.append((f"{name}/POSCAR", len(atoms), atoms.get_chemical_formula(mode="metal"), lengths,
                 source))


poscar("Ba2BiVO6_primitive_PBEU", primitive(), "Ba2BiVO6 Fm-3m, VASP PBE+U (U_V 3.25 eV) relaxed",
       "VASP PBE+U, 520 eV, relaxed in Fm-3m (the earlier HSE06 runs used this cell)")
poscar("Ba2BiVO6_cubic40_PBEU", conventional(), "Ba2BiVO6 Fm-3m, 40-atom cubic cell, PBE+U",
       "the same, as the 40-atom cubic cell")

doped = read(WORK / "doped_frames.extxyz", ":")
for frame in doped:
    group = frame.info["group"]
    if group == "pristine_min":
        poscar("Ba2BiVO6_cubic40_MACE-MP-0", frame, "Ba2BiVO6 relaxed with MACE-MP-0 (cubic)",
               "MACE-MP-0 small, cell and ions relaxed from the cubic cell (stays cubic)")
    elif group.endswith("_min"):
        dopant, x = frame.info["dopant"], frame.info["x"]
        label = "BiVO6" if x == 0 else f"Bi(V{1 - x:g}{dopant}{x:g})O6" if x < 1 else \
            f"Bi{dopant}O6"
        poscar(f"Ba2{label.replace('(', '_').replace(')', '_')}_x{x:g}_MACE-MP-0".replace("__",
                                                                                           "_"),
               frame, f"Ba2{label} relaxed with MACE-MP-0 (from the cubic cell)",
               f"MACE-MP-0 small, V → {dopant} at x = {x:g}, cell and ions relaxed from cubic")

stability = WORK / "_stability_frames" / "frames.extxyz"
if stability.exists():
    for frame in read(stability, ":"):
        group = frame.info["group"]
        if group == "distorted_fixed_cell":
            poscar("Ba2BiVO6_distorted_fixedcell_MACE-MP-0", frame,
                   "Ba2BiVO6 cubic cell, ions relaxed by MACE-MP-0 from a rattle",
                   "MACE-MP-0 small, ions relaxed from a 0.02 Å rattle at the PBE+U cell "
                   "(−144 meV/f.u. vs cubic); to be checked with DFT")
        elif group == "distorted_relaxed_cell":
            poscar("Ba2BiVO6_distorted_relaxedcell_MACE-MP-0", frame,
                   "Ba2BiVO6 cell and ions relaxed by MACE-MP-0 from a rattle",
                   "MACE-MP-0 small, cell and ions relaxed from a rattle (−608 meV/f.u., "
                   "+23 % volume, broken octahedra); to be checked with DFT")

dilute = WORK / "dilute_frames.extxyz"
if dilute.exists():
    seen = set()
    for frame in read(dilute, ":"):
        dopant = frame.info["dopant"]
        if dopant not in seen:  # the first of each dopant is the relaxed cell
            seen.add(dopant)
            poscar(f"Ba2BiVO6_80atom_1{dopant}_MACE-MP-0", frame,
                   f"Ba2BiVO6 2x2x2 primitive cell with one V -> {dopant}, MACE-MP-0",
                   f"MACE-MP-0 small, one {dopant} in the 80-atom cell (x = 12.5 %), ions "
                   "relaxed at the host lattice")

for name in ("frames.extxyz", "doped_frames.extxyz", "dilute_frames.extxyz"):
    if (WORK / name).exists():
        shutil.copyfile(WORK / name, FRAMES / name)

table = "\n".join(f"| `{name}` | {n} | {formula} | {lengths} | {source} |"
                  for name, n, formula, lengths, source in rows)
counts = {name: len(read(FRAMES / name, ":"))
          for name in ("frames.extxyz", "doped_frames.extxyz", "dilute_frames.extxyz")
          if (FRAMES / name).exists()}
(OUT / "README.md").write_text(f"""# Ba₂BiVO₆ structures

Every structure of the example, written by `export_structures.py`: one folder
per structure with its `POSCAR` (VASP 5, direct coordinates, species grouped in
the order Ba, V, Nb/Ta, Bi, O). Copy a folder into a VASP run, or open the file
in VESTA or ASE (`ase.io.read(path)`). Cell lengths are in Å.

| File | Atoms | Formula | Cell lengths (Å) | Where it comes from |
|---|---|---|---|---|
{table}

**Caution.** MACE-MP-0 finds the cubic cell unstable (`stability_check.py`). The
substitution series and the 80-atom cells were relaxed from the cubic cell and
stayed close to it, so they are cubic-saddle structures at MACE-MP-0 level, not
necessarily ground states. The two distorted structures are MACE-MP-0
predictions that the LONI stability check (`hpc_smoke_tests/08_vasp_bbvo_stability`)
is meant to confirm or rule out.

## `../frames/`

The exact frame sets the LONI packages were written from, as extended XYZ
(`ase.io.read(path, ":")`; `info["group"]` says where each frame comes from):

""" + "".join(f"- `{name}`: {count} frames\n" for name, count in counts.items()) + """
MD on a GPU does not repeat bit for bit, so rerunning `make_frames.py`,
`doping.py`, or `dilute.py` gives statistically equivalent but not identical
frames. Keep these with any labels computed from them: the collectors check
every returned geometry against its frame.
""", encoding="utf-8")
print(f"{len(rows)} POSCARs -> {OUT}; {len(counts)} frame sets -> {FRAMES}")
print(np.array([r[0] for r in rows]))
