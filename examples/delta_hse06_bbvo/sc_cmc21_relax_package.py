"""Package 41: the rattled Cmc2₁ Ba₂ScVO₆ of package 37, re-relaxed with the cell free (PBE+U).
For LONI; nothing is submitted.  -> sc_cmc21_relax_run/
(also loni_smoke_tests/batch10_2026-10-09/41_vasp_bbvo_sc_cmc21_isif3)

Package 37: OQMD's Cmc2₁ VO₄ framework with Bi -> Sc is +661.5 meV/f.u. above the Ba₂ScVO₆
perovskite (ref_BiSc1) as relaxed, but its rattle drops it by 620 meV/f.u. to +41.4, close to the
PI's 50 meV/f.u. line. That rattle relaxed the atoms only, in a cell 35 % larger per f.u. than the
perovskite's. Here the rattled structure (37 frame 0's 4_rattle CONTCAR) relaxes cell and ions
(ISIF 3), so its energy compares directly with ref_BiSc1's E(3_static). 37's MACE_VO4_Sc relaxed
back to the perovskite (+0.6 meV/f.u., V CN 6), so it has no VO₄ structure to re-relax.
PI-approved on 2026-10-09 within 37's 5 node-hours (bbvo_analysis_review 10_CALCULATION_PLAN.md).

Chain: 1_relax -> 2_relax (19's relax tags, ISIF 3, ISYM 0: the rattled cell has no symmetry to
keep; NSW 300 instead of 200, since 37's fixed-cell rattle took 215 steps) -> 3_static (19's static
tags, as ref_BiSc1) -> 4_rattle (0.05 Å, ISIF 2, ISYM 0). Mesh: 0.25 Å⁻¹ on the rattled cell, as 37.

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python sc_cmc21_relax_package.py \\
        [--out DIR] [--copy DIR|'']
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from ase.io import read
from band_kpoints import SOURCE19
from common import WORK, grouped
from loni_chain import write_frame, write_manifest, write_script

from samson_mlip_visualizer.vasp_labeling import kpoint_mesh

SPACING = 0.25
SOURCE37 = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests\batch10_2026-10-09"
                r"\37_vasp_bbvo_sc_polymorphs")
CHAIN = {"1_relax": "poscar", "2_relax": "contcar", "3_static": "contcar",
         "4_rattle": ("rattle", 0.05, 1)}

p = argparse.ArgumentParser()
p.add_argument("--source19", type=Path, default=SOURCE19)
p.add_argument("--source37", type=Path, default=SOURCE37)
p.add_argument("--out", type=Path, default=WORK / "sc_cmc21_relax_run")
p.add_argument("--copy", type=lambda s: Path(s) if s else None,
               default=Path(r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests"
                            r"\batch10_2026-10-09\41_vasp_bbvo_sc_cmc21_isif3"))
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg19 = json.loads((args.source19 / "package.json").read_text(encoding="utf-8"))
pkg37 = json.loads((args.source37 / "package.json").read_text(encoding="utf-8"))
RELAX = {**pkg19["relax"], "ISYM": 0, "NSW": 300}
STATIC, RATTLE = pkg19["static"], pkg19["rattle"]
if pkg37["static"] != STATIC or pkg37["rattle"] != RATTLE:
    sys.exit("37's static/rattle tags differ from 19's; the comparison would not be direct")

f37 = pkg37["frames"][0]
if f37["name"] != "OQMD_Cmc21_Sc":
    sys.exit(f"37 frame 0 is {f37['name']}, expected OQMD_Cmc21_Sc")
out = args.source37 / "outputs" / "frame_0000" / "4_rattle"
if "</modeling>" not in (out / "vasprun.xml").read_text(errors="ignore")[-200:]:
    sys.exit("37 OQMD_Cmc21_Sc 4_rattle did not finish")
src = out / "CONTCAR"
atoms = grouped(read(src, format="vasp"))
name = "Cmc21_Sc_isif3"
mesh = kpoint_mesh(atoms, SPACING)
levels = {"1_relax": RELAX, "2_relax": RELAX, "3_static": {**STATIC, "ISYM": 0},
          "4_rattle": RATTLE}
write_frame(args.out, 0, atoms, levels, SPACING, mesh=mesh, title=name)
frames = [{"frame": 0, "name": name, "group": "Ba2ScVO6", "natoms": len(atoms),
           "formula": atoms.get_chemical_formula(), "isym": 0, "kmesh": list(mesh),
           "geometry": "37 frame 0 (OQMD_Cmc21_Sc) 4_rattle CONTCAR",
           "source_file": str(src),
           "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}]
print(f"frame 0: {name} {atoms.get_chemical_formula()} {len(atoms)} at mesh {mesh} "
      f"cell {atoms.cell.lengths().round(3)}")

write_script(args.out, 1, CHAIN, name="bbvo-sc-cmc21-isif3", time="12:00:00",
             resume=("1_relax", "2_relax", "4_rattle"))
write_manifest(args.out, frames, {
    "package": "41_vasp_bbvo_sc_cmc21_isif3", "kspacing": SPACING,
    "chain": {k: str(v) for k, v in CHAIN.items()},
    "relax": RELAX, "static": STATIC, "rattle": RATTLE,
    "reference": "ref_BiSc1",
    "reference_package": "batch10_2026-10-09/37_vasp_bbvo_sc_polymorphs",
    "budget_nh": "within 37's <= 5 (PI, 2026-10-09)",
    "generator": "examples/delta_hse06_bbvo/sc_cmc21_relax_package.py",
    "analyzer": "examples/delta_hse06_bbvo/polymorph_analyze.py",
    "energy_for_comparison": "3_static e_fr_energy per formula unit (10 atoms), "
                             "against 37's ref_BiSc1 3_static",
    "decision_rule": "if Cmc21_Sc_isif3 lies more than ~50 meV/f.u. below ref_BiSc1, Ba2ScVO6 "
                     "is flagged as thermodynamically disfavoured like BBVO and B1 is not run; "
                     "otherwise B1 is proposed to the PI"})
(args.out / "README.md").write_text(f"""# 41_vasp_bbvo_sc_cmc21_isif3: package 37's rattled Cmc2_1 Ba2ScVO6, re-relaxed with the cell free (PBE+U)

Written by `examples/delta_hse06_bbvo/sc_cmc21_relax_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
PI-approved on 2026-10-09 as the follow-up to 37, within 37's 5 node-hours (bbvo_analysis_review, item A2).

| Frame | Name | Atoms | k-mesh | Geometry |
|---|---|---|---|---|
| 0 | {name} | {len(atoms)} | {'×'.join(map(str, mesh))} | 37 frame 0 (OQMD_Cmc21_Sc) `4_rattle` CONTCAR |

Why: in 37, Cmc2_1-Sc was +661.5 meV/f.u. above the perovskite (`ref_BiSc1`) as relaxed, but +41.4 after a
fixed-cell rattle (drop 620 meV/f.u.), in a cell 35 % larger per f.u. than the perovskite's. Here cell and ions relax.
37's MACE_VO4_Sc relaxed back to the perovskite, so there is nothing to re-relax there.

Levels (`vasp6/6.6.1-cpu`, one 64-core node): `1_relax` -> `2_relax` (19's relax tags, ISIF 3, ISYM 0, NSW 300) ->
`3_static` (19's static tags, as `ref_BiSc1`) -> `4_rattle` (0.05 A, ISIF 2, ISYM 0). MP POTCARs (Sc_sv), U on V
only. A timed-out level resumes from its own CONTCAR. Cost: 37 frame 0 took about 1 h for its chain; expect 1-2 h.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/polymorph_analyze.py <this folder>`
(it reads `ref_BiSc1` from `reference_package`, next to this one in the results clone): every relax reached EDIFFG,
E(3_static) per f.u. relative to `ref_BiSc1`, space group, V coordination, volume per f.u., rattle drop.

Decision (PI-approved rule): if `{name}` lies more than ~50 meV/f.u. below `ref_BiSc1`, Ba2ScVO6 is flagged as
thermodynamically disfavoured like BBVO and its hybrid gap (B1) is not run; otherwise B1 is proposed to the PI.
""", encoding="utf-8", newline="\n")  # noqa: E501  (LF: read on LONI)
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + ": 1 frame")
