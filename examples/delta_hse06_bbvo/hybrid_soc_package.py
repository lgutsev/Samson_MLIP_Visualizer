"""Package 30: hybrid gaps of the low Ba₂BiVO₆ polymorphs, and SOC on the whole shortlist.
For LONI; nothing is submitted.  -> hybrid_soc_run/ (also loni_smoke_tests/30_vasp_bbvo_hybrid_soc)

Package 19 ranked the polymorphs at PBE+U: OQMD's non-perovskite Cmc2₁ lowest, then the MACE-MP-0
VO₄ structure, ..., R3 the lowest perovskite (the target phase), cubic highest. Two questions:

1. **Gaps at the hybrid level** (Stage 4 of the audit's calculation plan): HSE06 on Cmc2₁, R3, the
   MACE-MP-0 VO₄ structure and, as the reference at the same settings, the 10-atom cubic cell.
   Per frame: ``a_edges`` PBE+U with zero-weight points (dense uniform grid, Γ-TRIM lines, the
   TRIMs: where the edges are and the converged PBE+U gap); ``b_pbeu_mesh`` PBE+U on 19's SCF mesh
   and symmetry, keeping its WAVECAR; ``c_hse06`` HSE06 from that WAVECAR on the same mesh.
   HSE06 gap ≈ PBE+U converged gap + (HSE06 − PBE+U) on the mesh; the mesh gaps are reported too.
2. **Does SOC change the ranking?** PBE+U and PBE+U+SOC (``vasp_ncl``) single points on all 14 of
   19's static geometries, both at ISYM 0 on 19's mesh (``d_pbeu``, ``e_pbeu_soc``): ΔE_SOC per
   f.u. per structure and the SOC-corrected order within each composition.
3. **The cost of HSE06+SOC**, measured once (the plan's request): R3, PBE+U+SOC on 19's mesh and
   symmetry keeping its WAVECAR, then HSE06+SOC from it (``f_hse06_soc``).

Settings are 19's (MP POTCARs, U(V) = 3.25 eV only, 520 eV, PREC Accurate, 0.25 Å⁻¹ mesh); HSE06
as in the labeling packages (AEXX 0.25, HFSCREEN 0.2, PRECFOCK Normal, ALGO Damped). Geometries:
19's ``3_static`` CONTCARs, the structures its energies belong to.

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python hybrid_soc_package.py \\
        [--out DIR] [--copy DIR, '' for none]
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

from samson_mlip_visualizer.vasp_labeling import incar

CHAIN = {
    "a_edges": "poscar",
    "b_pbeu_mesh": "poscar",
    "c_hse06": "poscar",
    "d_pbeu": "poscar",
    "e_pbeu_soc": "poscar",
    "f_hse06_soc": "poscar",
}
EXE = {"e_pbeu_soc": "vasp_ncl", "f_hse06_soc": "vasp_ncl"}
WAVE = ("c_hse06", "f_hse06_soc")
HYBRID = ("cubic_Fm-3m", "R3_polar", "OQMD_Cmc21_1344250", "MACE_VO4")
BASE = {
    "NSW": 0,
    "IBRION": -1,
    "ISIF": 2,
    "EDIFF": "1E-6",
    "NCORE": 4,
    "KPAR": 4,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
    "LORBIT": 0,
}

p = argparse.ArgumentParser()
p.add_argument("--source19", type=Path, default=SOURCE19)
p.add_argument("--out", type=Path, default=WORK / "hybrid_soc_run")
p.add_argument(
    "--copy",
    type=lambda s: Path(s) if s else None,
    default=Path(
        r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests\30_vasp_bbvo_hybrid_soc"
    ),
)
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg19 = json.loads((args.source19 / "package.json").read_text(encoding="utf-8"))


def soc_tags(atoms):
    return {
        "LSORBIT": ".TRUE.",
        "GGA_COMPAT": ".FALSE.",
        "MAGMOM": f"{3 * len(atoms)}*0",
        "NBANDS": nbands(atoms, soc=True),
    }


def pbeu(atoms, tags, title, level):
    text = incar("pbe_u", atoms, extra=tags).replace("single point", f"{title} {level}")
    return text.replace("ISPIN = 1\n", "") if "LSORBIT" in tags else text


def hse(atoms, tags, title, level):
    text = incar("hse06", atoms, extra=tags).replace("single point", f"{title} {level}")
    return text.replace("ISPIN = 1\n", "") if "LSORBIT" in tags else text


plan = [(f["name"], "hybrid") for f in pkg19["frames"] if f["name"] in HYBRID]
plan += [(f["name"], "soc") for f in pkg19["frames"]]
plan += [("R3_polar", "hybrid_soc_cost")]
frames = []
for name, kind in plan:
    f19 = next(f for f in pkg19["frames"] if f["name"] == name)
    out19 = args.source19 / "outputs" / f"frame_{f19['frame']:04d}" / "3_static"
    if "</modeling>" not in (out19 / "vasprun.xml").read_text(errors="ignore")[-200:]:
        sys.exit(f"19 {name} 3_static did not finish")
    atoms = grouped(read(out19 / "CONTCAR", format="vasp"))
    mesh, isym, nb = tuple(f19["kmesh"]), f19["isym"], nbands(atoms)
    index, title = len(frames), f"{name}_{kind}"
    write_frame(args.out, index, atoms, {}, None, mesh=mesh, title=title)
    folder = args.out / "inputs" / f"frame_{index:04d}"
    inc = {}
    if kind == "hybrid":
        n_uniform = 8 if len(atoms) <= 10 else (6 if len(atoms) <= 20 else 4)
        extra = (
            [(k, f"uniform {n_uniform}") for k in bk.uniform(n_uniform)]
            + bk.path()
            + [(k, lab) for lab, k in bk.trims().items()]
        )
        bk.write_labels(
            folder,
            "a_edges",
            bk.write_explicit(folder / "KPOINTS.a_edges", mesh, extra, f"{title} a_edges"),
        )
        inc["a_edges"] = pbeu(
            atoms, {**BASE, "ISYM": 0, "EDIFF": "1E-7", "NBANDS": nb}, title, "a_edges"
        )
        inc["b_pbeu_mesh"] = pbeu(
            atoms, {**BASE, "ISYM": isym, "NBANDS": nb, "LWAVE": ".TRUE."}, title, "b_pbeu_mesh"
        )
        inc["c_hse06"] = hse(atoms, {**BASE, "ISYM": isym, "NBANDS": nb}, title, "c_hse06")
    elif kind == "soc":
        inc["d_pbeu"] = pbeu(
            atoms, {**BASE, "ISYM": 0, "EDIFF": "1E-7", "NBANDS": nb}, title, "d_pbeu"
        )
        inc["e_pbeu_soc"] = pbeu(
            atoms, {**BASE, "ISYM": 0, "EDIFF": "1E-7", **soc_tags(atoms)}, title, "e_pbeu_soc"
        )
    else:  # one HSE06+SOC run, from a PBE+U+SOC WAVECAR on the same mesh and symmetry
        inc["e_pbeu_soc"] = pbeu(
            atoms, {**BASE, "ISYM": isym, "LWAVE": ".TRUE.", **soc_tags(atoms)}, title, "e_pbeu_soc"
        )
        inc["f_hse06_soc"] = hse(
            atoms, {**BASE, "ISYM": isym, **soc_tags(atoms)}, title, "f_hse06_soc"
        )
    for level, text in inc.items():
        (folder / f"INCAR.{level}").write_text(text, newline="\n")
    frames.append(
        {
            "frame": index,
            "name": name,
            "kind": kind,
            "natoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "kmesh": list(mesh),
            "isym_hse": isym,
            "nbands": nb,
            "levels": list(inc),
            "geometry": f"19 frame {f19['frame']} {name} 3_static CONTCAR",
            "source_sha256": hashlib.sha256((out19 / "CONTCAR").read_bytes()).hexdigest(),
        }
    )
    print(f"frame {index}: {title} {atoms.get_chemical_formula()} mesh {mesh} levels {list(inc)}")

write_script(
    args.out,
    len(frames),
    CHAIN,
    name="bbvo-hybrid-soc",
    time="48:00:00",
    throttle=8,
    exe=EXE,
    wavecar=WAVE,
    keep=("EIGENVAL",),
    level_kpoints=True,
)
write_manifest(
    args.out,
    frames,
    {
        "package": "30_vasp_bbvo_hybrid_soc",
        "kspacing": 0.25,
        "chain": CHAIN,
        "exe": EXE,
        "wavecar_from_previous": list(WAVE),
        "source19": str(args.source19),
        "generator": "examples/delta_hse06_bbvo/hybrid_soc_package.py",
        "analyzer": "examples/delta_hse06_bbvo/hybrid_soc_analyze.py",
    },
)
table = "\n".join(
    f"| {f['frame']} | {f['name']} | {f['kind']} | {f['natoms']} | "
    f"{'×'.join(map(str, f['kmesh']))} | {' → '.join(f['levels'])} |"
    for f in frames
)
(args.out / "README.md").write_text(
    f"""# 30_vasp_bbvo_hybrid_soc: HSE06 gaps of the low Ba2BiVO6 polymorphs, SOC on the shortlist

Written by `examples/delta_hse06_bbvo/hybrid_soc_package.py` (Samson_MLIP_Visualizer). \
Nothing was submitted.
Owner: Samson_MLIP_Visualizer. Geometries: package 19's `3_static` CONTCARs; settings are 19's.

| Frame | Structure | Part | Atoms | SCF mesh | Levels |
|---|---|---|---|---|---|
{table}

- **hybrid** (frames 0-3): `a_edges` PBE+U with zero-weight band points (edges, converged gap),
  `b_pbeu_mesh` PBE+U on 19's mesh/symmetry (WAVECAR kept), `c_hse06` HSE06 from that WAVECAR.
- **soc** (frames 4-17): `d_pbeu` and `e_pbeu_soc` (`vasp_ncl`, LSORBIT) at ISYM 0 on 19's mesh.
- **hybrid_soc_cost** (frame 18): R3 `e_pbeu_soc` (WAVECAR kept) then `f_hse06_soc`: one HSE06+SOC
  run, to measure what HSE06+SOC costs before any more are planned.

**Before submitting: `vasp_ncl`.** No package has used it yet. On a login node:
`module load vasp6/6.6.1-cpu && command -v vasp_ncl`. The SOC levels stop with a clear message if
it is missing; the non-SOC levels of every frame still run.

Cost: PBE+U levels minutes to ~1 h; HSE06 10-atom ≈ 3-6 h, Cmc2₁ (20 atoms) ≈ 5-15 h, MACE VO₄
(40 atoms, P1, 3×3×4) ≈ 15-30 h (the doped 40-atom cells took up to 28 h); SOC single points
≈ 4-8× their PBE+U; HSE06+SOC on R3 not measured (that is the point of frame 18). 48 h per task,
8 at once. Roughly 60-110 node-hours, most of it frames 2, 3 and 18.

Check: `PYTHONPATH=../../src micromamba run -n defects python \
examples/delta_hse06_bbvo/hybrid_soc_analyze.py <this folder>`.
""",
    encoding="utf-8",
)
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + f": {len(frames)} frames")
