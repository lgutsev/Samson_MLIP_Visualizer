"""Package 45 ("Sc-col"): rock-salt vs columnar B-site ordering in Ba₂ScVO₆, PBE+U.
For LONI; nothing is submitted, the PI launches it.  -> sc_columnar_run/
(also loni_smoke_tests/batch10_2026-10-09/45_vasp_bbvo_sc_columnar)

Berger & Neaton (PRB 86, 165211, 2012; LDA) put the ferroelectric columnar ordering of Ba₂ScVO₆
about 80 meV/f.u. below rock salt; every Ba₂ScVO₆ cell so far (31, 37, 41-44) is rock salt.
Spec: bbvo_analysis_review handoffs/to_Samson_MLIP_Visualizer_2026-10-10_columnar.md.

Start structures are bbvo_analysis_review's (outputs/columnar_package/), used as given: atom order
Ba, Sc, V, O, each file's sha256 checked against atom_map.json. Frame 0 is 37's ref_BiSc1
4_rattle CONTCAR (reordered); frames 1-5 the columnar cell with columns along c: centrosymmetric,
polar seeds +c, -c, +a, and a 0.05 Å rattle.

Settings are 37's Ba₂ScVO₆ reference without imposed symmetry: 1_relax -> 2_relax (19's relax
tags, ISIF 3, ISYM 0, NSW 300 as in 41) -> 3_static (19's static tags, ISYM 0) -> 4_rattle
(0.05 Å, ISIF 2, ISYM 0) -> 31's 5_edges; Γ-centred mesh at 0.25 Å⁻¹ (4×4×4). The analyser is
bbvo_analysis_review's scripts/columnar_analyze.py.

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python sc_columnar_package.py \\
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
from common import WORK
from loni_chain import write_frame, write_manifest, write_script

from samson_mlip_visualizer.vasp_labeling import incar, kpoint_mesh, species_order

SPACING = 0.25
SOURCE = Path(r"C:\Users\lguts\OneDrive\Desktop\Test_Code\bbvo_analysis_review\outputs"
              r"\columnar_package")
SOURCE31 = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests\batch09_2026-10-07\31_vasp_bbvo_levers")
CHAIN = {"1_relax": "poscar", "2_relax": "contcar", "3_static": "contcar",
         "4_rattle": ("rattle", 0.05, 1), "5_edges": "last"}

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, default=SOURCE)
p.add_argument("--source19", type=Path, default=SOURCE19)
p.add_argument("--source31", type=Path, default=SOURCE31)
p.add_argument("--out", type=Path, default=WORK / "sc_columnar_run")
p.add_argument("--copy", type=lambda s: Path(s) if s else None,
               default=Path(r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests"
                            r"\batch10_2026-10-09\45_vasp_bbvo_sc_columnar"))
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg19 = json.loads((args.source19 / "package.json").read_text(encoding="utf-8"))
pkg31 = json.loads((args.source31 / "package.json").read_text(encoding="utf-8"))
amap = json.loads((args.source / "atom_map.json").read_text(encoding="utf-8"))
RELAX = {**pkg19["relax"], "ISIF": 3, "ISYM": 0, "NSW": 300}
STATIC = {**pkg19["static"], "ISYM": 0}
RATTLE = pkg19["rattle"]
BAND = {**pkg31["band"], "LORBIT": 0}

frames = []
for f in amap["frames"]:
    src = args.source / f["file"]
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if sha != f["sha256"]:
        sys.exit(f"{src.name}: sha256 {sha[:12]} differs from atom_map.json {f['sha256'][:12]}")
    atoms = read(src, format="vasp")  # as given: no regrouping, POTCAR order follows the file
    if atoms.get_chemical_symbols() != f["atom_order"]:
        sys.exit(f"{src.name}: atom order differs from atom_map.json")
    if len(set(species_order(atoms))) != len(species_order(atoms)):
        sys.exit(f"{src.name}: species are not contiguous")
    index, name = f["frame"], f["name"]
    if index != len(frames):
        sys.exit("atom_map.json frames are not 0, 1, 2, ...")
    mesh = kpoint_mesh(atoms, SPACING)
    levels = {"1_relax": RELAX, "2_relax": RELAX, "3_static": STATIC, "4_rattle": RATTLE}
    write_frame(args.out, index, atoms, levels, SPACING, mesh=mesh, title=name)
    folder = args.out / "inputs" / f"frame_{index:04d}"
    edges = [(k, "uniform 4") for k in bk.uniform(4)] + bk.path()
    bk.write_labels(folder, "5_edges",
                    bk.write_explicit(folder / "KPOINTS.5_edges", mesh, edges, f"{name} 5_edges"))
    nb = nbands(atoms)
    text = incar("pbe_u", atoms, extra={**BAND, "NBANDS": nb}).replace("single point",
                                                                       f"{name} 5_edges")
    (folder / "INCAR.5_edges").write_text(text, newline="\n")
    frames.append({"frame": index, "name": name, "how": f["how"], "natoms": len(atoms),
                   "formula": atoms.get_chemical_formula(), "kmesh": list(mesh),
                   "start_spacegroup": f"{f['spg_1e-3']} / {f['spg_0.1']}",
                   "nbands_edges": nb, "source_file": str(src), "source_sha256": sha})
    print(f"frame {index}: {name:20s} {f['spg_1e-3']:12s} mesh {mesh} NBANDS {nb}")

write_script(args.out, len(frames), CHAIN, name="bbvo-sc-columnar", time="12:00:00",
             throttle=6, keep=("EIGENVAL",), level_kpoints=True,
             resume=("1_relax", "2_relax", "4_rattle"))
write_manifest(args.out, frames, {
    "package": "45_vasp_bbvo_sc_columnar", "kspacing": SPACING,
    "chain": {k: str(v) for k, v in CHAIN.items()},
    "relax": RELAX, "static": STATIC, "rattle": RATTLE, "band": BAND,
    "reference": "rocksalt_ref",
    "start_structures": "bbvo_analysis_review outputs/columnar_package (commit 5fa23e4), "
                        "atom_map.json sha256 checked",
    "generator": "examples/delta_hse06_bbvo/sc_columnar_package.py",
    "analyzer": "bbvo_analysis_review scripts/columnar_analyze.py",
    "cost_nh": "8-15 (bbvo_analysis_review estimate); trim frame 4 first, then frame 3"})
rows = "\n".join(f"| {f['frame']} | {f['name']} | {f['start_spacegroup']} | "
                 f"{'×'.join(map(str, f['kmesh']))} | {f['how']} |" for f in frames)
(args.out / "README.md").write_text(f"""# 45_vasp_bbvo_sc_columnar: Ba2ScVO6 rock-salt vs columnar B-site ordering (PBE+U)

Written by `examples/delta_hse06_bbvo/sc_columnar_package.py` (Samson_MLIP_Visualizer). Nothing was submitted; the PI launches it.
Spec: bbvo_analysis_review `handoffs/to_Samson_MLIP_Visualizer_2026-10-10_columnar.md`. Berger & Neaton (PRB 86, 165211, 2012;
LDA) put the ferroelectric columnar ordering about 80 meV/f.u. below rock salt; every Ba2ScVO6 cell so far is rock salt.

| Frame | Name | Start SG (1e-3 / 0.1 A) | k-mesh | Start |
|---|---|---|---|---|
{rows}

Start structures: bbvo_analysis_review `outputs/columnar_package/` (commit 5fa23e4), used as given (atom order Ba, Sc, V, O;
sha256 checked against `atom_map.json`). Checked here as well: frame 0 is 37's ref_BiSc1 (species-matched, 1e-15 A); frames
1-5 have 2 like and 4 unlike B-site neighbours (columns along c), frame 0 has 0 / 6 (rock salt).

Levels (`vasp6/6.6.1-cpu`, one 64-core node per frame): `1_relax` -> `2_relax` (19's relax tags, ISIF 3, ISYM 0, NSW 300) ->
`3_static` (19's static tags, ISYM 0) -> `4_rattle` (0.05 A, ISIF 2, ISYM 0) -> 31's `5_edges`. MP POTCARs (Ba_sv, Sc_sv, V_pv,
O), U on V only. A timed-out relax or rattle relax resumes from its own CONTCAR.

Cost (bbvo_analysis_review's estimate): 8-15 node-hours; 37's 40-atom chain + 5_edges took 1.15 h, the columnar frames start
far from a minimum without symmetry (1.5-3 h each). To trim: drop frame 4 first, then frame 3. 12 h limit, 6 at once.

Check: `python scripts/columnar_analyze.py <this folder>` in bbvo_analysis_review (reference: frame 0).
""", encoding="utf-8", newline="\n")  # noqa: E501
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + f": {len(frames)} frames")
