"""Packages 46-47: polar columnar Ba₂ScVO₆, its conduction band (PBE+U) and an r2SCAN tie-break.
For LONI; nothing is submitted.  -> sc_polar_run/46_..., 47_...
(also loni_smoke_tests/batch11_2026-10-11/)

Package 45 put the polar columnar B-site ordering (P4mm, VO₅ pyramids) 131 meV/f.u. below rock
salt at PBE+U; package 41 put Cmc2₁ at −11. PI-approved on 2026-10-11, ≈ 3-6 nh in all
(bbvo_analysis_review handoffs/to_Samson_MLIP_Visualizer_2026-10-11_columnar_bands_r2scan.md).

Common cell: 45 frame 2's 4_rattle CONTCAR (P4mm, 40 atoms) reduced to its 10-atom primitive cell
with spglib (``standardize_cell(to_primitive=True)``, symprec 0.01), then rotated 45° about c so
the in-plane V-O bonds lie along x and y: VASP's d_xy / d_yz / d_xz (LORBIT 11) and the stencil
directions are then those of the VO₅ pyramid, as in the rock-salt cells. Atom order Ba, Sc, V, O.
Checked: P4mm, the 40-atom structure reproduced, V five-coordinate.

- 46_vasp_bbvo_sc_polar_bands (PBE+U, 31's settings): 1_relax -> 2_relax (19's relax tags,
  ISIF 3, ISYM 0, NSW 300) -> 3_static -> 31's 5_edges, 6_mass (stencil at all eight TRIMs)
  and 7_char, the last two with LORBIT 11. cbm_analyze.py reads CB1-3 per orbital and direction.
- 47_vasp_bbvo_sc_r2scan (r2SCAN, no U, Materials Project convention): rock salt (42 frame 0's
  start cell), polar columnar (the common cell) and Cmc2₁ (41's final CONTCAR); 1_relax -> 2_relax
  (ISIF 3, ISYM 0, NSW 300) -> 3_static -> 4_rattle (frames 1 and 2 only).

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python sc_polar_package.py \\
        [--out DIR] [--copy DIR|'']
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import band_kpoints as bk
import numpy as np
import spglib
from ase import Atoms
from ase.io import read
from band_kpoints import nbands
from common import WORK
from loni_chain import write_frame, write_manifest, write_script
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from samson_mlip_visualizer.vasp_labeling import incar, kpoint_mesh

SPACING = 0.25
H = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests")
SRC45 = H / "batch10_2026-10-09" / "45_vasp_bbvo_sc_columnar"
SRC41 = H / "batch10_2026-10-09" / "41_vasp_bbvo_sc_cmc21_isif3"
SRC42 = H / "batch10_2026-10-09" / "42_vasp_bbvo_scb_electronic"
SRC19 = H / "batch04_2026-09-30" / "19_vasp_bbvo_polymorphs"
SRC31 = H / "batch09_2026-10-07" / "31_vasp_bbvo_levers"
BATCH = "batch11_2026-10-11"
NAMES = {46: "46_vasp_bbvo_sc_polar_bands", 47: "47_vasp_bbvo_sc_r2scan"}
ORDER = ["Ba", "Sc", "V", "O"]

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=WORK / "sc_polar_run")
p.add_argument("--copy", type=lambda s: Path(s) if s else None,
               default=Path(r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests") / BATCH)
args = p.parse_args()
for d in [args.out / n for n in NAMES.values()] + (
        [args.copy / n for n in NAMES.values()] if args.copy else []):
    if d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg19 = json.loads((SRC19 / "package.json").read_text(encoding="utf-8"))
pkg31 = json.loads((SRC31 / "package.json").read_text(encoding="utf-8"))
pkg45 = json.loads((SRC45 / "package.json").read_text(encoding="utf-8"))
pkg41 = json.loads((SRC41 / "package.json").read_text(encoding="utf-8"))


def finished(out, what):
    if "</modeling>" not in (out / "vasprun.xml").read_text(errors="ignore")[-200:]:
        sys.exit(f"{what} did not finish")
    return out / "CONTCAR"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered(atoms, order=ORDER):
    idx = sorted(range(len(atoms)), key=lambda i: order.index(atoms[i].symbol))
    return atoms[idx]


def v_o(atoms):
    """Sorted V-O distances (Å) of each V within 3 Å."""
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", atoms, 3.0)
    s = atoms.get_chemical_symbols()
    return [sorted(round(float(x), 3) for a, b, x in zip(i, j, d, strict=True)
                   if a == v and s[b] == "O") for v in range(len(atoms)) if s[v] == "V"]


# ---- the common polar columnar 10-atom cell ----
f45 = pkg45["frames"][2]
if f45["name"] != "columnar_polar_pz":
    sys.exit(f"45 frame 2 is {f45['name']}, expected columnar_polar_pz")
src45 = finished(SRC45 / "outputs" / "frame_0002" / "4_rattle", "45 frame 2 4_rattle")
big = read(src45, format="vasp")
lat, pos, num = spglib.standardize_cell(
    (big.cell[:], big.get_scaled_positions(), big.get_atomic_numbers()), to_primitive=True,
    symprec=0.01)
prim = ordered(Atoms(numbers=num, cell=lat, scaled_positions=pos, pbc=True))
if len(prim) != 10:
    sys.exit(f"spglib primitive has {len(prim)} atoms, expected 10")
# rotate about z so an in-plane V-O bond lies along x (tetragonal cell: c stays along z)
if not np.allclose(prim.cell[:2, 2], 0, atol=1e-6) or not np.allclose(prim.cell[2, :2], 0,
                                                                         atol=1e-6):
    sys.exit("spglib cell is not c along z")
vi = [i for i, s in enumerate(prim.get_chemical_symbols()) if s == "V"][0]
best = None
for oi in [i for i, s in enumerate(prim.get_chemical_symbols()) if s == "O"]:
    dv = prim.get_distance(vi, oi, mic=True, vector=True)
    if abs(dv[2]) < 0.3 and np.linalg.norm(dv) < 2.3 and (best is None or dv[0] > best[0]):
        best = dv
phi = math.degrees(math.atan2(best[1], best[0]))
prim.rotate(-phi, "z", rotate_cell=True)
for oi in [i for i, s in enumerate(prim.get_chemical_symbols()) if s == "O"]:
    dv = prim.get_distance(vi, oi, mic=True, vector=True)
    if abs(dv[2]) < 0.3 and np.linalg.norm(dv) < 2.3:
        ang = math.degrees(math.atan2(dv[1], dv[0])) % 90
        if min(ang, 90 - ang) > 0.5:
            sys.exit(f"in-plane V-O bond {ang:.2f} deg off the x/y axes after the rotation")
sga = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(prim), symprec=0.01)
spg_prim = sga.get_space_group_symbol()
matcher = StructureMatcher(ltol=0.05, stol=0.05, angle_tol=1, primitive_cell=True, scale=False)
same = matcher.fit(AseAtomsAdaptor.get_structure(big), AseAtomsAdaptor.get_structure(prim))
rms = matcher.get_rms_dist(AseAtomsAdaptor.get_structure(big), AseAtomsAdaptor.get_structure(prim))
vo = v_o(prim)
cn = [sum(1 for x in d if x < 2.4) for d in vo]
if spg_prim != "P4mm" or not same or cn != [5]:
    sys.exit(f"polar cell check failed: {spg_prim}, matches 45: {same}, V CN {cn}")
lengths = prim.cell.lengths()
polar_info = {"geometry": "45 frame 2 (columnar_polar_pz) 4_rattle CONTCAR, spglib primitive "
                          "(symprec 0.01), rotated so in-plane V-O bonds lie along x and y",
              "source_file": str(src45), "source_sha256": sha(src45), "spacegroup": spg_prim,
              "a_A": round(float(lengths[0]), 4), "c_A": round(float(lengths[2]), 4),
              "matches_45_rms": [round(float(x), 5) for x in rms],
              "V_O_A": vo[0], "rotation_deg": round(-phi, 3)}
print(f"polar cell: {spg_prim}, a {lengths[0]:.4f} c {lengths[2]:.4f} A, matches 45 (rms "
      f"{rms[0]:.1e}), V-O {vo[0][:6]}, rotated {-phi:.2f} deg")

RELAX = {**pkg19["relax"], "ISIF": 3, "ISYM": 0, "NSW": 300}
STATIC = {**pkg19["static"], "ISYM": 0}

# ---- 46: polar columnar conduction band (PBE+U) ----
root = args.out / NAMES[46]
mesh = kpoint_mesh(prim, SPACING)
write_frame(root, 0, prim, {"1_relax": RELAX, "2_relax": RELAX, "3_static": STATIC}, SPACING,
            mesh=mesh, title="polar_columnar")
folder = root / "inputs" / "frame_0000"
edges = [(k, "uniform 8") for k in bk.uniform(8)] + bk.path()
mass = [pt for label, k in bk.trims().items() for pt in bk.stencil(prim.cell, k, label)]
char = [(k, label) for label, k in bk.trims().items()]
nb = nbands(prim)
for level, extra, lorbit in (("5_edges", edges, 0), ("6_mass", mass, 11), ("7_char", char, 11)):
    bk.write_labels(folder, level, bk.write_explicit(folder / f"KPOINTS.{level}", mesh, extra,
                                                     f"polar_columnar {level}"))
    text = incar("pbe_u", prim, extra={**pkg31["band"], "NBANDS": nb, "LORBIT": lorbit})
    (folder / f"INCAR.{level}").write_text(
        text.replace("single point", f"polar_columnar {level}"), newline="\n")
CHAIN46 = {"1_relax": "poscar", "2_relax": "contcar", "3_static": "contcar", "5_edges": "last",
           "6_mass": "last", "7_char": "last"}
write_script(root, 1, CHAIN46, name="bbvo-sc-polar-bands", time="12:00:00",
             keep=("EIGENVAL",), level_kpoints=True, resume=("1_relax", "2_relax"))
write_manifest(root, [{"frame": 0, "name": "polar_columnar", "kind": "relaxed", "natoms": 10,
                       "formula": prim.get_chemical_formula(), "kmesh": list(mesh),
                       "nbands": nb, **polar_info}],
               {"package": NAMES[46], "kspacing": SPACING, "relax": RELAX, "static": STATIC,
                "band": pkg31["band"], "chain": CHAIN46,
                "energy_check": "1_relax first ionic step per f.u. vs 45 frame 2 4_rattle final "
                                "(spec: within ~2 meV/f.u.; meshes differ)",
                "budget_nh": "46 + 47 about 3-6 (PI, 2026-10-11)",
                "generator": "examples/delta_hse06_bbvo/sc_polar_package.py",
                "analyzer": "examples/delta_hse06_bbvo/cbm_analyze.py"})
(root / "README.md").write_text(f"""# {NAMES[46]}: conduction band of polar columnar Ba2ScVO6 (P4mm), PBE+U

Written by `examples/delta_hse06_bbvo/sc_polar_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
PI-approved on 2026-10-11 (bbvo_analysis_review, package A of the columnar-bands / r2SCAN handoff).

One frame: 45 frame 2's rattle-relaxed polar columnar cell reduced to 10 atoms (spglib, symprec 0.01; P4mm,
a = {lengths[0]:.4f}, c = {lengths[2]:.4f} A; V-O {vo[0][0]:.3f} / {vo[0][-2] if len(vo[0]) > 5 else ''} A), rotated so the
in-plane V-O bonds lie along x and y: d_xy, d_yz, d_xz and the stencil directions are the pyramid's. Atom order Ba, Sc, V, O.

Levels: `1_relax` -> `2_relax` (19's relax tags, ISIF 3, ISYM 0, NSW 300) -> `3_static` -> 31's `5_edges` (uniform 8^3 +
Γ-TRIM lines), `6_mass` (±0.04 / ±0.08 A^-1 stencil at all eight TRIMs) and `7_char` (TRIMs), the last two with LORBIT 11.
Mesh {'×'.join(map(str, mesh))} (0.25 A^-1), NBANDS {nb}. Cost: 29's 10-atom band levels took 0.13 h; about 0.3-0.5 h with the relax.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/cbm_analyze.py <this folder>`: gap, CBM/VBM,
mass tensors at every TRIM, and (LORBIT 11) CB1-3 at the CBM TRIM per orbital and direction, with their splitting. Energy check:
`1_relax` first ionic step (OSZICAR line 1, F) per f.u. against 45 frame 2's `4_rattle` final energy per f.u. (spec: within
~2 meV/f.u.; the 10- and 40-atom meshes differ).
""", encoding="utf-8", newline="\n")  # noqa: E501
print(f"{NAMES[46]}: mesh {mesh}, NBANDS {nb}, {len(mass)} stencil points")

# ---- 47: r2SCAN tie-break ----


def r2scan(atoms, tags, title):
    """r2SCAN, no U (Materials Project convention: V5+ is d0), from the PBE+U INCAR."""
    text = incar("pbe_u", atoms, extra={**tags, "LWAVE": ".FALSE."})
    keep = [line for line in text.splitlines()
            if not line.split("=")[0].strip().startswith(("LDAU", "LMAXMIX", "GGA", "ALGO"))]
    keep[0] = f"SYSTEM = {atoms.get_chemical_formula()} r2SCAN (no U) {title}"
    keep += ["METAGGA = R2SCAN", "LMIXTAU = .TRUE.", "ALGO = All", "LDAU = .FALSE."]
    return "\n".join(keep) + "\n"


R2_RELAX = {"NSW": 300, "IBRION": 2, "ISIF": 3, "ISYM": 0, "EDIFF": "1E-6", "EDIFFG": -0.01,
            "NCORE": 4, "KPAR": 4, "LCHARG": ".FALSE."}
R2_STATIC = {"NSW": 0, "IBRION": -1, "ISIF": 2, "ISYM": 0, "EDIFF": "1E-7", "NCORE": 4,
             "KPAR": 4, "LCHARG": ".FALSE."}
R2_RATTLE = {**R2_RELAX, "ISIF": 2}
src42 = SRC42 / "inputs" / "frame_0000" / "POSCAR"
f41 = pkg41["frames"][0]
src41 = finished(SRC41 / "outputs" / "frame_0000" / "4_rattle", "41 4_rattle")
plan = [("rocksalt", read(src42, format="vasp"), src42, "42 frame 0 start POSCAR (Fm-3m)", False),
        ("polar_columnar", prim, src45, polar_info["geometry"], True),
        ("Cmc21", read(src41, format="vasp"), src41, f"41 frame 0 ({f41['name']}) 4_rattle CONTCAR",
         True)]
root = args.out / NAMES[47]
frames = []
for i, (name, atoms, src, how, rattle) in enumerate(plan):
    m = kpoint_mesh(atoms, SPACING)
    write_frame(root, i, atoms, {}, None, mesh=m, title=name)
    folder = root / "inputs" / f"frame_{i:04d}"
    levels = {"1_relax": R2_RELAX, "2_relax": R2_RELAX, "3_static": R2_STATIC}
    if rattle:
        levels["4_rattle"] = R2_RATTLE
    for level, tags in levels.items():
        (folder / f"INCAR.{level}").write_text(r2scan(atoms, tags, f"{name} {level}"),
                                               newline="\n")
    frames.append({"frame": i, "name": name, "group": "Ba2ScVO6", "natoms": len(atoms),
                   "formula": atoms.get_chemical_formula(), "kmesh": list(m), "isym": 0,
                   "geometry": how, "source_file": str(src), "source_sha256": sha(src),
                   "levels": list(levels)})
    print(f"{NAMES[47]} frame {i}: {name:15s} {len(atoms):2d} at mesh {m} levels {list(levels)}")
CHAIN47 = {"1_relax": "poscar", "2_relax": "contcar", "3_static": "contcar",
           "4_rattle": ("rattle", 0.05, 1)}
write_script(root, len(frames), CHAIN47, name="bbvo-sc-r2scan", time="12:00:00",
             resume=("1_relax", "2_relax", "4_rattle"))
write_manifest(root, frames, {
    "package": NAMES[47], "kspacing": SPACING, "functional": "r2SCAN, no U (MP convention)",
    "relax": R2_RELAX, "static": R2_STATIC, "rattle": R2_RATTLE,
    "chain": {k: str(v) for k, v in CHAIN47.items()}, "reference": "rocksalt",
    "budget_nh": "46 + 47 about 3-6 (PI, 2026-10-11)",
    "generator": "examples/delta_hse06_bbvo/sc_polar_package.py",
    "analyzer": "examples/delta_hse06_bbvo/polymorph_analyze.py (skips the rattle where a frame "
                "has none); V offset: bbvo_analysis_review scripts/columnar_analyze.py",
    "decision": "if polar columnar stays lowest in r2SCAN (any margin), it is the reference "
                "structure for the retargeted property set; otherwise the PI decides"})
table = "\n".join(f"| {f['frame']} | {f['name']} | {f['natoms']} | {f['geometry']} | "
                  f"{' -> '.join(f['levels'])} |" for f in frames)
(root / "README.md").write_text(f"""# {NAMES[47]}: r2SCAN tie-break of rock salt, polar columnar and Cmc2_1 Ba2ScVO6

Written by `examples/delta_hse06_bbvo/sc_polar_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
PI-approved on 2026-10-11 (bbvo_analysis_review, package B).

| Frame | Structure | Atoms | Start | Levels |
|---|---|---|---|---|
{table}

r2SCAN (`METAGGA = R2SCAN`, `LMIXTAU = .TRUE.`, `ALGO = All`, LASPH), no U (`LDAU = .FALSE.`; V5+ is d0, the Materials
Project's r2SCAN convention), MP PBE PAW POTCARs, 520 eV, PREC Accurate, LREAL .FALSE., ISMEAR 0 / SIGMA 0.05, 0.25 A^-1.
`1_relax` -> `2_relax`: ISIF 3, ISYM 0, NSW 300, EDIFFG -0.01, EDIFF 1e-6; `3_static`: EDIFF 1e-7, ISYM 0; `4_rattle` (0.05 A,
ISIF 2, ISYM 0) on frames 1 and 2 only. Cost: r2SCAN about 2-4x PBE; about 2-3 node-hours. 12 h limit.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/polymorph_analyze.py <this folder>`
(frame 0 has no 4_rattle by design; the analyzer skips it): E(3_static) per f.u. against frame 0 (rocksalt), space group,
V coordination, cell and c/a, gap on the mesh (r2SCAN, for information). V offset along c: bbvo_analysis_review
`scripts/columnar_analyze.py`.
""", encoding="utf-8", newline="\n")  # noqa: E501

if args.copy:
    for n in NAMES.values():
        shutil.copytree(args.out / n, args.copy / n)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else ""))
