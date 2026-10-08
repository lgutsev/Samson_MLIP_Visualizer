"""Package 31: three levers for CBM dispersion and the structural instability together, PBE+U.
For LONI; nothing is submitted.  -> lever_run/
(also loni_smoke_tests/batch09_2026-10-07/31_vasp_bbvo_levers)

Package 29 showed that partial Nb/Ta on the V site makes the V-3d conduction band heavier
(m*_e 2.8 -> 4.1-9.0 m_e at x = 0.25-0.75): the V site was the wrong lever. Package 18 showed the
polar Gamma instability (T1u) is Bi + O (V weight < 5 %) and the rotation (T1g) is a size effect.
Agreed with the PI on 2026-10-07 as one bounded screen (~25-40 node-hours), CBM dispersion first:

A. strain (does compression lighten the band and remove the off-centering?)
   - cubic 40-atom cell (19's x0_cubic40 static): hydrostatic a x 0.98, 0.99, 1.01 (ISIF 2);
     biaxial a = b x 0.98 and 1.02 with c relaxed (ISIF 3 + IOPTCELL, checked afterwards);
   - R3 (19's R3_polar static, 10 atoms): hydrostatic x 0.98 and 1.02 (ISIF 2).
B. lone-pair-free trivalent cation on the Bi site (removes the Bi-driven polar mode, raises the
   tolerance factor; In 5s / Sc 3d may add an empty dispersive band near V 3d):
   Ba2Bi(1-y)M(y)VO6, M = In, Sc, y = 0.25, 0.5, 1 (40-atom cell; full relax, 19's chain).
C. V/Nb(Ta) ordering: (111)-layered orderings on the B' sublattice, against 29's rock-salt-
   sublattice orderings (Pm-3m at x = 0.25, (001)-layered P4/mmm at x = 0.5):
   x = 0.5 in the doubled primitive cell (20 atoms, L-point order) and x = 0.25 in the
   quadrupled primitive cell (40 atoms, period-4 stacking along [111]); M = Nb, Ta.

Every frame: 1_relax -> 2_relax -> 3_static -> 4_rattle (0.05 A, ISIF 2, ISYM 0), then 29's band
levels at the rattle-relaxed geometry: 5_edges, 6_mass, 7_char (cbm_analyze.py reads them).
References at strain 1.00 and at the rock-salt orderings are package 29's frames.

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python lever_package.py \
        [--out DIR] [--copy DIR|'']
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import band_kpoints as bk
import numpy as np
from ase.build import make_supercell
from ase.io import read
from band_kpoints import SOURCE19, nbands
from common import WORK, grouped
from loni_chain import write_frame, write_manifest, write_script
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from samson_mlip_visualizer.vasp_labeling import incar, kpoint_mesh

SPACING = 0.25
CHAIN = {"1_relax": "poscar", "2_relax": "contcar", "3_static": "contcar",
         "4_rattle": ("rattle", 0.05, 1), "5_edges": "last", "6_mass": "last", "7_char": "last"}
BAND = {"NSW": 0, "IBRION": -1, "ISIF": 2, "ISYM": 0, "EDIFF": "1E-7", "NCORE": 4, "KPAR": 4,
        "LWAVE": ".FALSE.", "LCHARG": ".FALSE."}

p = argparse.ArgumentParser()
p.add_argument("--source19", type=Path, default=SOURCE19)
p.add_argument("--out", type=Path, default=WORK / "lever_run")
p.add_argument("--copy", type=lambda s: Path(s) if s else None,
               default=Path(r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests"
                            r"\batch09_2026-10-07\31_vasp_bbvo_levers"))
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg19 = json.loads((args.source19 / "package.json").read_text(encoding="utf-8"))
by_name = {f["name"]: f for f in pkg19["frames"]}
RELAX, STATIC, RATTLE = pkg19["relax"], pkg19["static"], pkg19["rattle"]


def from19(name, level="3_static"):
    out = args.source19 / "outputs" / f"frame_{by_name[name]['frame']:04d}" / level
    if "</modeling>" not in (out / "vasprun.xml").read_text(errors="ignore")[-200:]:
        sys.exit(f"19 {name} {level} did not finish")
    return grouped(read(out / "CONTCAR", format="vasp")), out / "CONTCAR"


def spg(atoms, tol=0.01):
    structure = AseAtomsAdaptor.get_structure(atoms)
    return SpacegroupAnalyzer(structure, symprec=tol).get_space_group_symbol()


def substitute(atoms, old, new, count):
    """Replace the first ``count`` ``old`` sites (index order); returns a grouped copy."""
    a = atoms.copy()
    idx = [i for i, s in enumerate(a.get_chemical_symbols()) if s == old][:count]
    syms = a.get_chemical_symbols()
    for i in idx:
        syms[i] = new
    a.set_chemical_symbols(syms)
    return grouped(a)


def relax_levels(mode):
    """1-4 for a frame: 'fixed' cell (ISIF 2), 'biaxial' (ISIF 3, only c free), or 'full'
    (ISIF 3)."""
    if mode == "fixed":
        r = {**RELAX, "ISIF": 2, "ISYM": 2}
    elif mode == "biaxial":
        r = {**RELAX, "ISIF": 3, "ISYM": 2, "IOPTCELL": "0 0 0 0 0 0 0 0 1"}
    else:
        r = {**RELAX, "ISYM": 2}
    return {"1_relax": r, "2_relax": r, "3_static": {**STATIC, "ISYM": 2}, "4_rattle": RATTLE}


plan = []
cub40, cub40_src = from19("x0_cubic40")
r3, r3_src = from19("R3_polar")
prim, prim_src = from19("cubic_Fm-3m")
for s in (0.98, 0.99, 1.01):  # A: hydrostatic, cubic
    a = cub40.copy()
    a.set_cell(a.cell[:] * s, scale_atoms=True)
    plan.append(("A", f"cubic40_hydro{s:.2f}", a, "fixed", cub40_src,
                 f"x0_cubic40 static, lattice x {s}"))
for s in (0.98, 1.02):  # A: biaxial, cubic
    a = cub40.copy()
    c = a.cell[:].copy()
    c[0] *= s
    c[1] *= s
    a.set_cell(c, scale_atoms=True)
    plan.append(("A", f"cubic40_biax{s:.2f}", a, "biaxial", cub40_src,
                 f"x0_cubic40 static, a = b x {s}, c relaxed"))
for s in (0.98, 1.02):  # A: hydrostatic, R3
    a = r3.copy()
    a.set_cell(a.cell[:] * s, scale_atoms=True)
    plan.append(("A", f"R3_hydro{s:.2f}", a, "fixed", r3_src, f"R3_polar static, lattice x {s}"))
for m in ("In", "Sc"):  # B: Bi-site substitution in the 40-atom cell (4 Bi sites)
    for y, n in ((0.25, 1), (0.5, 2), (1.0, 4)):
        a = substitute(cub40, "Bi", m, n)
        plan.append(("B", f"Bi{m}{y:g}", a, "full", cub40_src,
                     f"x0_cubic40 static, {n} of 4 Bi -> {m}"))
for m in ("Nb", "Ta"):  # C: (111)-layered orderings from the 10-atom primitive cell
    a2 = grouped(make_supercell(prim, np.diag([2, 1, 1])))
    plan.append(("C", f"{m}0.5_111", substitute(a2, "V", m, 1), "full", prim_src,
                 "cubic_Fm-3m static x [2,1,1]: V/M alternate along a1 (L-point, (111) layers)"))
    a4 = grouped(make_supercell(prim, np.diag([4, 1, 1])))
    plan.append(("C", f"{m}0.25_111", substitute(a4, "V", m, 1), "full", prim_src,
                 "cubic_Fm-3m static x [4,1,1]: V V V M along a1 ((111) stacking, period 4)"))

frames = []
for lever, name, atoms, mode, src, how in plan:
    index = len(frames)
    mesh = kpoint_mesh(atoms, SPACING)
    folder = args.out / "inputs" / f"frame_{index:04d}"
    write_frame(args.out, index, atoms, relax_levels(mode), SPACING, mesh=mesh, title=name)
    n_uniform = 8 if len(atoms) <= 10 else (6 if len(atoms) <= 20 else 4)
    edges = [(k, f"uniform {n_uniform}") for k in bk.uniform(n_uniform)] + bk.path()
    mass = [pt for label, k in bk.trims().items() for pt in bk.stencil(atoms.cell, k, label)]
    char = [(k, label) for label, k in bk.trims().items()]
    nb = nbands(atoms)
    for level, extra, lorbit in (("5_edges", edges, 0), ("6_mass", mass, 0), ("7_char", char, 10)):
        labels = bk.write_explicit(folder / f"KPOINTS.{level}", mesh, extra, f"{name} {level}")
        bk.write_labels(folder, level, labels)
        text = incar("pbe_u", atoms, extra={**BAND, "NBANDS": nb, "LORBIT": lorbit})
        text = text.replace("single point", f"{name} {level}")
        (folder / f"INCAR.{level}").write_text(text, newline="\n")
    frames.append({"frame": index, "lever": lever, "name": name, "kind": "relaxed",
                   "relax_mode": mode, "natoms": len(atoms),
                   "formula": atoms.get_chemical_formula(), "start_spacegroup": spg(atoms),
                   "cell_start_A": np.round(atoms.cell.lengths(), 4).tolist(),
                   "kmesh": list(mesh), "nbands": nb, "geometry": how, "source_file": str(src),
                   "source_sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest()})
    print(f"frame {index:2d} [{lever}] {name:18s} {atoms.get_chemical_formula():20s} "
          f"{len(atoms):3d} at  {spg(atoms):8s} mesh {mesh} NBANDS {nb} {mode}")

write_script(args.out, len(frames), CHAIN, name="bbvo-levers", time="12:00:00", throttle=8,
             keep=("EIGENVAL",), level_kpoints=True)
write_manifest(args.out, frames, {
    "package": "31_vasp_bbvo_levers", "kspacing": SPACING,
    "chain": {k: str(v) for k, v in CHAIN.items()},
    "relax": RELAX, "static": STATIC, "rattle": RATTLE, "band": BAND,
    "stencil_steps_A-1": list(bk.STEPS),
    "references": "package 29 (strain 1.00, rock-salt orderings, x0 control); "
                  "package 19 rattle drops",
    "budget_nh": "25-40 (agreed with the PI 2026-10-07)",
    "generator": "examples/delta_hse06_bbvo/lever_package.py",
    "analyzer": "examples/delta_hse06_bbvo/cbm_analyze.py",
    "decision_rule": "a lever advances only if m*_e (conductivity, relaxed) < 2.8 m_e "
                     "(29's x = 0), PBE+U gap <= ~1.7 eV, "
                     "and the rattle drop is no larger than the reference's"})
rows = "\n".join(f"| {f['frame']} | {f['lever']} | {f['name']} | {f['natoms']} | "
                 f"{f['start_spacegroup']} | {f['relax_mode']} | "
                 f"{'×'.join(map(str, f['kmesh']))} | {f['geometry']} |" for f in frames)
(args.out / "README.md").write_text(f"""# 31_vasp_bbvo_levers: strain, Bi-site In/Sc, and B'-ordering screen (PBE+U)

Written by `examples/delta_hse06_bbvo/lever_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
Owner: Samson_MLIP_Visualizer. Agreed with the PI on 2026-10-07 as one bounded screen (budget 25-40 node-hours).
Primary criterion: CBM dispersion (AGENTS.md); the instability is the secondary, optional objective and does
not gate anything.

| Frame | Lever | Name | Atoms | Start SG | Relax | SCF mesh | Geometry |
|---|---|---|---|---|---|---|---|
{rows}

Levels (dispatcher `run_vasp.slurm`, `vasp6/6.6.1-cpu`, one 64-core node per frame): `1_relax` -> `2_relax` ->
`3_static` -> `4_rattle` (0.05 A, ISIF 2, ISYM 0) -> `5_edges` / `6_mass` / `7_char` at the rattle-relaxed
geometry (package 29's band levels, explicit `KPOINTS.<level>` with zero-weight points).
- Relax modes: `fixed` = ISIF 2 at the strained cell; `biaxial` = ISIF 3 with `IOPTCELL = 0 0 0 0 0 0 0 0 1`
  (only c relaxes; **first use of IOPTCELL here**: check that a and b in `2_relax/CONTCAR` equal the start);
  `full` = 19's ISIF 3 relaxation.
- POTCARs: MP set, new here In_d and Sc_sv (`POTCAR.names`); U on V only.

Cost (package 29 measured: 40-atom chain + bands 1.8 h, band levels alone 0.7 h; 10-atom bands 0.13 h):
about 15 frames x 40 atoms x 1-2 h + 2 x 20 atoms + 2 x 10 atoms, roughly 25-35 node-hours; 12 h limit, 8 at once.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/cbm_analyze.py <this folder>`,
then the rattle drops (E(4_rattle) - E(3_static)) and, for the biaxial frames, the in-plane lattice check.
Compare with package 29 (x0 relaxed m*_e 2.76 m_e, R3 2.79 m_e; rock-salt Nb/Ta x = 0.25 4.1-4.4, x = 0.5 5.4).

Decision rule (agreed): a lever advances only if the relaxed electron conductivity mass falls below 2.8 m_e,
the PBE+U gap stays at or below ~1.7 eV, and the rattle drop is no larger than its reference's. If none
qualifies, the stabilization/dispersion search stops and the paper goes ahead as a structure-property study.
""", encoding="utf-8")  # noqa: E501
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + f": {len(frames)} frames")
