"""Package 37: does Ba₂ScVO₆ have a VO₄ polymorph far below its perovskite? PBE+U.
For LONI; nothing is submitted.  -> sc_polymorph_run/
(also loni_smoke_tests/batch10_2026-10-09/37_vasp_bbvo_sc_polymorphs)

Ba₂ScVO₆ (31 frame 12, BiSc1) is the lead CBM candidate. OQMD has no VO₄-type Ba₂ScVO₆, while
for Ba₂BiVO₆ the VO₄ structure Cmc2₁ lies 1.31 eV/f.u. below the perovskite (package 19). Approved
by the PI on 2026-10-09, budget <= 5 node-hours (bbvo_analysis_review
handoffs/to_Samson_MLIP_Visualizer_2026-10-09_polymorph_optics.md, item 1).

Frames, all Ba₂ScVO₆, 19's chain and tags (MP POTCARs, Sc_sv as in 31):
- ``OQMD_Cmc21_Sc``: 19's OQMD_Cmc21_1344250 3_static CONTCAR, every Bi -> Sc (20 atoms);
- ``MACE_VO4_Sc``: 19's MACE_VO4 3_static CONTCAR, every Bi -> Sc (40 atoms, P1 as in 19);
- ``ref_BiSc1``: 31 frame 12 final geometry (4_rattle CONTCAR), the perovskite on the same footing.

Chain: 1_relax -> 2_relax (both ISIF 3) -> 3_static -> 4_rattle (0.05 Å, ISIF 2, ISYM 0), then
31's 5_edges (PBE+U band edges at the rattle-relaxed geometry; cheap, context only). 31 frame 12
ran the same INCAR tags but at 3×3×3 (0.25 Å⁻¹ on its larger Ba₂BiVO₆ start cell); the ref frame
gets 4×4×4 on its relaxed cell, so 31's energy is a cross-check, not a substitute.

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python sc_polymorph_package.py \\
        [--out DIR] [--copy DIR|'']
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import band_kpoints as bk
from ase.io import read
from band_kpoints import SOURCE19, nbands
from common import WORK, grouped
from loni_chain import write_frame, write_manifest, write_script
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from samson_mlip_visualizer.vasp_labeling import incar, kpoint_mesh

SPACING = 0.25
SOURCE31 = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests\batch09_2026-10-07\31_vasp_bbvo_levers")
CHAIN = {"1_relax": "poscar", "2_relax": "contcar", "3_static": "contcar",
         "4_rattle": ("rattle", 0.05, 1), "5_edges": "last"}
BAND = {"NSW": 0, "IBRION": -1, "ISIF": 2, "ISYM": 0, "EDIFF": "1E-7", "NCORE": 4, "KPAR": 4,
        "LWAVE": ".FALSE.", "LCHARG": ".FALSE.", "LORBIT": 0}

p = argparse.ArgumentParser()
p.add_argument("--source19", type=Path, default=SOURCE19)
p.add_argument("--source31", type=Path, default=SOURCE31)
p.add_argument("--out", type=Path, default=WORK / "sc_polymorph_run")
p.add_argument("--copy", type=lambda s: Path(s) if s else None,
               default=Path(r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests"
                            r"\batch10_2026-10-09\37_vasp_bbvo_sc_polymorphs"))
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg19 = json.loads((args.source19 / "package.json").read_text(encoding="utf-8"))
pkg31 = json.loads((args.source31 / "package.json").read_text(encoding="utf-8"))
RELAX, STATIC, RATTLE = pkg19["relax"], pkg19["static"], pkg19["rattle"]


def finished(out, what):
    if "</modeling>" not in (out / "vasprun.xml").read_text(errors="ignore")[-200:]:
        sys.exit(f"{what} did not finish")
    return grouped(read(out / "CONTCAR", format="vasp")), out / "CONTCAR"


def spg(atoms, tol=0.01):
    s = AseAtomsAdaptor.get_structure(atoms)
    return SpacegroupAnalyzer(s, symprec=tol).get_space_group_symbol()


def bi_to_sc(atoms):
    a = atoms.copy()
    a.set_chemical_symbols(["Sc" if s == "Bi" else s for s in a.get_chemical_symbols()])
    return grouped(a)


plan = []
for name in ("OQMD_Cmc21_1344250", "MACE_VO4"):
    f19 = next(f for f in pkg19["frames"] if f["name"] == name)
    atoms, src = finished(args.source19 / "outputs" / f"frame_{f19['frame']:04d}" / "3_static",
                          f"19 {name} 3_static")
    plan.append((f"{name.split('_1344250')[0]}_Sc", bi_to_sc(atoms), f19["isym"], src,
                 f"19 frame {f19['frame']} {name} 3_static CONTCAR, all Bi -> Sc"))
f31 = pkg31["frames"][12]
if f31["name"] != "BiSc1":
    sys.exit(f"31 frame 12 is {f31['name']}, expected BiSc1")
atoms, src = finished(args.source31 / "outputs" / "frame_0012" / "4_rattle", "31 BiSc1 4_rattle")
plan.append(("ref_BiSc1", atoms, 2, src, "31 frame 12 (BiSc1) 4_rattle CONTCAR (final geometry)"))

frames = []
for name, atoms, isym, src, how in plan:
    index = len(frames)
    mesh = kpoint_mesh(atoms, SPACING)
    levels = {"1_relax": {**RELAX, "ISYM": isym}, "2_relax": {**RELAX, "ISYM": isym},
              "3_static": {**STATIC, "ISYM": isym}, "4_rattle": RATTLE}
    write_frame(args.out, index, atoms, levels, SPACING, mesh=mesh, title=name)
    folder = args.out / "inputs" / f"frame_{index:04d}"
    n_uniform = 8 if len(atoms) <= 10 else (6 if len(atoms) <= 20 else 4)
    edges = [(k, f"uniform {n_uniform}") for k in bk.uniform(n_uniform)] + bk.path()
    bk.write_labels(folder, "5_edges",
                    bk.write_explicit(folder / "KPOINTS.5_edges", mesh, edges, f"{name} 5_edges"))
    nb = nbands(atoms)
    text = incar("pbe_u", atoms, extra={**BAND, "NBANDS": nb}).replace("single point",
                                                                       f"{name} 5_edges")
    (folder / "INCAR.5_edges").write_text(text, newline="\n")
    frames.append({"frame": index, "name": name, "group": "Ba2ScVO6",
                   "natoms": len(atoms), "formula": atoms.get_chemical_formula(),
                   "start_spacegroup": spg(atoms), "isym": isym, "kmesh": list(mesh),
                   "nbands_edges": nb, "geometry": how, "source_file": str(src),
                   "source_sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest()})
    print(f"frame {index}: {name:16s} {atoms.get_chemical_formula():16s} {len(atoms):3d} at "
          f"{spg(atoms):8s} ISYM {isym} mesh {mesh}")

write_script(args.out, len(frames), CHAIN, name="bbvo-sc-polymorphs", time="12:00:00",
             throttle=3, keep=("EIGENVAL",), level_kpoints=True,
             resume=("1_relax", "2_relax", "4_rattle"))
write_manifest(args.out, frames, {
    "package": "37_vasp_bbvo_sc_polymorphs", "kspacing": SPACING,
    "chain": {k: str(v) for k, v in CHAIN.items()},
    "relax": RELAX, "static": STATIC, "rattle": RATTLE, "band": BAND,
    "reference": "ref_BiSc1",
    "reference_crosscheck": "31 frame 12 (BiSc1) 3_static: same tags, but 3x3x3 there vs "
                            "4x4x4 here (0.25 A-1 on each start cell), so not a substitute",
    "budget_nh": "<= 5 (PI, 2026-10-09)",
    "generator": "examples/delta_hse06_bbvo/sc_polymorph_package.py",
    "analyzer": "examples/delta_hse06_bbvo/polymorph_analyze.py",
    "energy_for_comparison": "3_static e_fr_energy per formula unit (10 atoms)",
    "decision_rule": "if either VO4 frame lies more than ~50 meV/f.u. below ref_BiSc1, "
                     "Ba2ScVO6 is flagged as thermodynamically disfavoured like BBVO and its "
                     "hybrid gap (B1) is not run; otherwise B1 is proposed to the PI"})
rows = "\n".join(f"| {f['frame']} | {f['name']} | {f['natoms']} | {f['start_spacegroup']} | "
                 f"ISYM {f['isym']} | {'×'.join(map(str, f['kmesh']))} | {f['geometry']} |"
                 for f in frames)
(args.out / "README.md").write_text(f"""# 37_vasp_bbvo_sc_polymorphs: Ba2ScVO6 in the two VO4 frameworks vs its perovskite (PBE+U)

Written by `examples/delta_hse06_bbvo/sc_polymorph_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
Approved by the PI on 2026-10-09, budget <= 5 node-hours (bbvo_analysis_review, item A2 / P1).
AGENTS.md ordering: CBM dispersion first; rattle drops and imaginary modes gate nothing.

| Frame | Name | Atoms | Start SG | Symmetry in relax | k-mesh | Geometry |
|---|---|---|---|---|---|---|
{rows}

Levels (`vasp6/6.6.1-cpu`, one 64-core node per frame): package 19's `1_relax` -> `2_relax` (ISIF 3) ->
`3_static` -> `4_rattle` (0.05 A, ISIF 2, ISYM 0), then package 31's `5_edges` (band edges at the
rattle-relaxed geometry, explicit `KPOINTS.5_edges`). MP POTCARs, `Sc_sv` as in 31; U on V only.
A timed-out relax or rattle relax resumes from its own CONTCAR.

Cost (measured on 19 and 31, same chain): Cmc2_1 20 atoms 0.6 h, MACE VO4 40 atoms 1.5 h, BiSc1 40 atoms
0.3 h, 5_edges up to 0.3 h each: about 2-3 node-hours. 12 h limit, all three at once.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/polymorph_analyze.py <this folder>`:
every relax reached EDIFFG, `3_static` EDIFF; E(3_static) per f.u. relative to `ref_BiSc1`, space group,
V coordination, rattle drop, gap on the mesh. Cross-check `ref_BiSc1` against 31 frame 12's `3_static`
(same tags; 31 ran 3x3x3 on its larger start cell, here 4x4x4, so expect a few meV/f.u. difference).

Decision (PI-approved): if either VO4 frame lies more than ~50 meV/f.u. below `ref_BiSc1`, Ba2ScVO6 is
flagged as thermodynamically disfavoured in the same way as BBVO and its hybrid gap (B1) is not run;
otherwise B1 is proposed to the PI. Not hull energies.
""", encoding="utf-8", newline="\n")  # noqa: E501  (LF: read on LONI)
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + f": {len(frames)} frames")
