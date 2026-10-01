"""PBE+U phonons of cubic Ba₂BiVO₆ for LONI. Nothing is submitted.
-> phonons/ (also loni_smoke_tests/batch04_2026-09-30/18_vasp_bbvo_phonons, dispatcher
``run_vasp.slurm`` layout)

Smoke test 08: a symmetry-free PBE+U relaxation from a 0.05 Å rattle lowered the
cubic 40-atom cell by 48.6 meV/f.u. at fixed volume (coherent V and Bi
off-centering, R3 at 0.1 Å tolerance). That is strong evidence motivating a
stability test, not a proof of negative curvature at the cubic point: a finite
rattle can also cross a small barrier. Harmonic phonons at the cubic structure
settle the curvature; a frozen-mode energy scan along the soft eigenvector (and
along the cubic -> relaxed path) settles double well vs barrier.

Finite displacements (phonopy, ±0.01 Å) at the campaign's settings (MP POTCARs,
U(V) = 3.25 eV, 520 eV, PREC = Accurate, LREAL = .FALSE., ISYM = 0), EDIFF = 1e-8:

- ``sc40``: 40-atom conventional cell (Γ and X): polar off-centering and every
  octahedral tilt of a rock-salt double perovskite (a⁻a⁻a⁻, a⁰a⁰c⁻ at Γ; a⁰a⁰c⁺ at X);
- ``sc40k4``: the same on a 4×4×4 mesh (k convergence of the soft frequencies);
- ``sc80``: 2×2×2 primitive cells (Γ, X, L);
- ``p0.99``, ``p0.98``, ``t1.01``: ``sc40`` at the lattice ×0.99, ×0.98, ×1.01;
- ``path08``: 9 single points on the straight line from the cubic cell (λ = 0) to the 08
  fixed-cell minimum (λ = 1) and beyond (1.2): does the energy fall from the cubic point
  (instability along this path) or rise first (a barrier: a metastable cubic phase)?

The cubic lattice is the earlier PBE+U 8.487 Å (P = +2.3 kbar at these settings,
≈0.05 % in a). Run in the ``defects`` environment (phonopy):

    PYTHONPATH=../../src micromamba run -n defects python phonon_package.py [--out DIR] [--copy DIR]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from common import KSPACING, WORK, grouped, primitive
from loni_chain import write_frame, write_manifest, write_script
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from samson_mlip_visualizer.vasp_labeling import species_order

SUPERCELLS = {"sc40": [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
              "sc80": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]}
SETS = {"sc40": ("sc40", 1.00, None), "sc40k4": ("sc40", 1.00, (4, 4, 4)),
        "sc80": ("sc80", 1.00, None),
        "p0.99": ("sc40", 0.99, None), "p0.98": ("sc40", 0.98, None), "t1.01": ("sc40", 1.01, None)}
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
TAGS = {"EDIFF": "1E-8", "ADDGRID": ".FALSE.", "NCORE": 4, "KPAR": 4, "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE."}

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=WORK / "phonons")
p.add_argument("--copy", type=Path, default=Path(
    r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests\batch04_2026-09-30"
    r"\18_vasp_bbvo_phonons"))
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")


def to_ph(a):
    return PhonopyAtoms(symbols=a.get_chemical_symbols(), cell=a.cell[:],
                        scaled_positions=a.get_scaled_positions())


def to_ase(c):
    return Atoms(c.symbols, cell=c.cell, scaled_positions=c.scaled_positions, pbc=True)


frames, sets = [], {}
(args.out / "phonopy").mkdir(parents=True)
for name, (sc, scale, mesh) in SETS.items():
    ph = Phonopy(to_ph(primitive(scale)), supercell_matrix=SUPERCELLS[sc],
                 primitive_matrix=np.eye(3))
    ph.generate_displacements(distance=0.01, is_plusminus=True)
    ph.save(args.out / "phonopy" / f"{name}.yaml")
    perfect = to_ase(ph.supercell)
    assert species_order(perfect) == species_order(grouped(perfect)), (
        "phonopy supercell not grouped by species")
    first = len(frames)
    for i, cell in enumerate(ph.supercells_with_displacements, 1):
        used = write_frame(args.out, len(frames), to_ase(cell), {"pbe_u": TAGS}, KSPACING, mesh,
                           f"{name} disp {i}")
        frames.append({"frame": len(frames), "set": name, "displacement": i})
    sets[name] = dict(supercell=sc, lattice_scale=scale, atoms=len(perfect),
                      frames=[first, len(frames) - 1], kmesh=list(used),
                      phonopy_yaml=f"phonopy/{name}.yaml")
    print(f"{name}: {len(perfect)} atoms, {len(frames) - first} displacements, mesh {used}")
# linear path cubic (λ = 0) -> 08 fixed-cell PBE+U minimum (λ = 1), same cell: a double well has
# E falling from λ = 0 with negative curvature; a barrier shows E rising first (metastable cubic)
from ase.io import read  # noqa: E402
from common import conventional  # noqa: E402

start = conventional()
end = read(Path(__file__).parent / "polymorphs" / "pbeu08_fixedcell" / "POSCAR")
assert (start.get_chemical_symbols() == end.get_chemical_symbols()
        and np.allclose(start.cell, end.cell, atol=1e-6))
delta = end.get_scaled_positions() - start.get_scaled_positions()
delta -= np.round(delta)
first = len(frames)
for lam in LAMBDAS:
    img = start.copy()
    img.set_scaled_positions(start.get_scaled_positions() + lam * delta)
    used = write_frame(args.out, len(frames), img, {"pbe_u": {**TAGS, "EDIFF": "1E-7"}}, KSPACING,
                       None, f"path lambda {lam:.2f}")
    frames.append({"frame": len(frames), "set": "path08", "lambda": lam})
sets["path08"] = dict(supercell="sc40", lattice_scale=1.0, atoms=len(start),
                      frames=[first, len(frames) - 1], kmesh=list(used),
                      endpoints=["cubic 8.487 A", "08 CONTCAR.1_ions (-48.6 meV/f.u.)"],
                      lambdas=LAMBDAS)
print(f"path08: {len(LAMBDAS)} images")
write_script(args.out, len(frames), {"pbe_u": "poscar"}, name="bbvo-phonons", time="04:00:00",
             throttle=8)
write_manifest(args.out, frames, {"package": "18_vasp_bbvo_phonons", "sets": sets,
                                  "incar_extra": TAGS,
                                  "generator": "examples/delta_hse06_bbvo/phonon_package.py"})
rows = "\n".join(f"| `{k}` | {v['atoms']} atoms, a × {v['lattice_scale']} | frames "
                 f"{v['frames'][0]}–{v['frames'][1]} | "
                 f"{'×'.join(map(str, v['kmesh']))} |" for k, v in sets.items())
(args.out / "README.md").write_text(f"""# 18_vasp_bbvo_phonons: PBE+U phonons of cubic Ba2BiVO6

Written by `examples/delta_hse06_bbvo/phonon_package.py` (Samson_MLIP_Visualizer). \
Nothing was submitted.
Layout: the dispatcher's `run_vasp.slurm` (one level, `pbe_u`; a frame is done when
`outputs/frame_NNNN/pbe_u/vasprun.xml` is complete). Frame -> displacement map: `package.json`.

| Set | Cell | Frames | k-mesh |
|---|---|---|---|
{rows}

Cost: {len(frames)} single points; 40-atom ones take about 2–4 min on one 64-core QB4 node
(89–119 s at EDIFF 1e-6 in smoke test 08, more at 1e-8), 80-atom ones an estimated 15–30 min
(not yet measured). About 3–6 node-hours in all; elapsed time is set by the queue.

Check (route proposal for `dispatch/routes.json`, for the desk to add):
- agent: Samson_MLIP_Visualizer; source: examples/delta_hse06_bbvo/phonon_package.py
- check: `PYTHONPATH=../../src micromamba run -n defects python phonon_analyze.py <this folder>`:
  every frame converged (NELM not reached), forces summed to ~0, frequencies at Γ/X(/L)
- next: if imaginary modes, write the frozen-mode scan (`phonon_analyze.py --scan`) and
  compare its minimum with the 08 relaxation; if none, run the cubic -> relaxed path scan
  (a barrier would mean a metastable cubic phase, not an artefact)
""", encoding="utf-8")
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + f": {len(frames)} frames")
