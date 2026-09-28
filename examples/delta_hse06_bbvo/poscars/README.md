# Ba₂BiVO₆ structures

Every structure of the example, written by `export_structures.py`: one folder
per structure with its `POSCAR` (VASP 5, direct coordinates, species grouped in
the order Ba, V, Nb/Ta, Bi, O). Copy a folder into a VASP run, or open the file
in VESTA or ASE (`ase.io.read(path)`). Cell lengths are in Å.

| File | Atoms | Formula | Cell lengths (Å) | Where it comes from |
|---|---|---|---|---|
| `Ba2BiVO6_primitive_PBEU/POSCAR` | 10 | Ba2BiVO6 | 6.001 × 6.001 × 6.001 | VASP PBE+U, 520 eV, relaxed in Fm-3m (the earlier HSE06 runs used this cell) |
| `Ba2BiVO6_cubic40_PBEU/POSCAR` | 40 | Ba8Bi4V4O24 | 8.487 × 8.487 × 8.487 | the same, as the 40-atom cubic cell |
| `Ba2BiVO6_cubic40_MACE-MP-0/POSCAR` | 40 | Ba8Bi4V4O24 | 8.502 × 8.502 × 8.502 | MACE-MP-0 small, cell and ions relaxed from the cubic cell (stays cubic) |
| `Ba2Bi_V0.75Nb0.25_O6_x0.25_MACE-MP-0/POSCAR` | 40 | Ba8Bi4NbV3O24 | 8.555 × 8.555 × 8.555 | MACE-MP-0 small, V → Nb at x = 0.25, cell and ions relaxed from cubic |
| `Ba2Bi_V0.5Nb0.5_O6_x0.5_MACE-MP-0/POSCAR` | 40 | Ba8Bi4Nb2V2O24 | 8.609 × 8.609 × 8.599 | MACE-MP-0 small, V → Nb at x = 0.5, cell and ions relaxed from cubic |
| `Ba2Bi_V0.25Nb0.75_O6_x0.75_MACE-MP-0/POSCAR` | 40 | Ba8Bi4Nb3VO24 | 8.655 × 8.655 × 8.655 | MACE-MP-0 small, V → Nb at x = 0.75, cell and ions relaxed from cubic |
| `Ba2BiNbO6_x1_MACE-MP-0/POSCAR` | 40 | Ba8Bi4Nb4O24 | 8.703 × 8.703 × 8.703 | MACE-MP-0 small, V → Nb at x = 1, cell and ions relaxed from cubic |
| `Ba2Bi_V0.75Ta0.25_O6_x0.25_MACE-MP-0/POSCAR` | 40 | Ba8Bi4TaV3O24 | 8.552 × 8.552 × 8.552 | MACE-MP-0 small, V → Ta at x = 0.25, cell and ions relaxed from cubic |
| `Ba2Bi_V0.5Ta0.5_O6_x0.5_MACE-MP-0/POSCAR` | 40 | Ba8Bi4Ta2V2O24 | 8.604 × 8.604 × 8.594 | MACE-MP-0 small, V → Ta at x = 0.5, cell and ions relaxed from cubic |
| `Ba2Bi_V0.25Ta0.75_O6_x0.75_MACE-MP-0/POSCAR` | 40 | Ba8Bi4Ta3VO24 | 8.647 × 8.647 × 8.647 | MACE-MP-0 small, V → Ta at x = 0.75, cell and ions relaxed from cubic |
| `Ba2BiTaO6_x1_MACE-MP-0/POSCAR` | 40 | Ba8Bi4Ta4O24 | 8.691 × 8.691 × 8.691 | MACE-MP-0 small, V → Ta at x = 1, cell and ions relaxed from cubic |
| `Ba2BiVO6_distorted_fixedcell_MACE-MP-0/POSCAR` | 40 | Ba8Bi4V4O24 | 8.487 × 8.487 × 8.487 | MACE-MP-0 small, ions relaxed from a 0.02 Å rattle at the PBE+U cell (−144 meV/f.u. vs cubic); to be checked with DFT |
| `Ba2BiVO6_distorted_relaxedcell_MACE-MP-0/POSCAR` | 40 | Ba8Bi4V4O24 | 10.017 × 9.666 × 7.834 | MACE-MP-0 small, cell and ions relaxed from a rattle (−608 meV/f.u., +23 % volume, broken octahedra); to be checked with DFT |
| `Ba2BiVO6_80atom_1Nb_MACE-MP-0/POSCAR` | 80 | Ba16Bi8NbV7O48 | 12.024 × 12.024 × 12.024 | MACE-MP-0 small, one Nb in the 80-atom cell (x = 12.5 %), ions relaxed at the host lattice |
| `Ba2BiVO6_80atom_1Ta_MACE-MP-0/POSCAR` | 80 | Ba16Bi8TaV7O48 | 12.024 × 12.024 × 12.024 | MACE-MP-0 small, one Ta in the 80-atom cell (x = 12.5 %), ions relaxed at the host lattice |

**Caution.** MACE-MP-0 finds the cubic cell unstable (`stability_check.py`). The
substitution series and the 80-atom cells were relaxed from the cubic cell and
stayed close to it, so they are cubic-saddle structures at MACE-MP-0 level, not
necessarily ground states. The two distorted structures are MACE-MP-0
predictions that the LONI stability check (`hpc_smoke_tests/08_vasp_bbvo_stability`)
is meant to confirm or rule out.

## `../frames/`

The exact frame sets the LONI packages were written from, as extended XYZ
(`ase.io.read(path, ":")`; `info["group"]` says where each frame comes from):

- `frames.extxyz`: 58 frames
- `doped_frames.extxyz`: 73 frames
- `dilute_frames.extxyz`: 4 frames

MD on a GPU does not repeat bit for bit, so rerunning `make_frames.py`,
`doping.py`, or `dilute.py` gives statistically equivalent but not identical
frames. Keep these with any labels computed from them: the collectors check
every returned geometry against its frame.
