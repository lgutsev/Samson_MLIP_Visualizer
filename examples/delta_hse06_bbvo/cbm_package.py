"""Package 29: CBM dispersion of Ba₂Bi(V,Nb,Ta)O₆ at relaxed geometries, PBE+U. For LONI;
nothing is submitted.  -> cbm_run/ (also loni_smoke_tests/29_vasp_bbvo_cbm_screen)

The primary dopant objective is the conduction-band dispersion (AGENTS.md). Package 19 showed the
cubic cells are saddles: every doped 40-atom cell relaxes 90-165 meV/f.u. lower after a rattle,
and the pristine perovskite falls to R3. So the screening geometry is the symmetry-free relaxed
cell, with the symmetric (cubic-derived) cell kept as a labelled reference:

- relaxed: R3 (pristine, 10 atoms) and 19's rattle-relaxed CONTCARs (``4_rattle``) of the
  40-atom x = 0 control and Nb/Ta x = 0.25, 0.5, 1;
- cubic reference: 19's symmetric relaxed geometries (``3_static`` CONTCAR) of the same cells;
- x = 0.75 (not in 19): relaxed here with 19's chain and settings (relax, relax, static,
  0.05 Å rattle-relax), then the same band levels.

Band levels (PBE+U, 19's settings: MP POTCARs, U(V) = 3.25 eV only, 520 eV, PREC Accurate,
0.25 Å⁻¹ SCF mesh; ISYM 0, EDIFF 1e-7, extra empty bands). Each is one SCF run with the extra
k-points at weight 0 (``band_kpoints.py``):

- ``5_edges``: dense uniform grid (4×4×4 on 40 atoms, 8×8×8 on 10) and the lines Γ → the
  other seven TRIMs: band edges, gap, where the CBM is, dispersion;
- ``6_mass``: Cartesian stencils (±0.04 Å⁻¹ on 3 axes and 6 diagonals, ±0.08 on the axes) at all
  eight TRIMs: the Hessian of each band there, hence electron and hole mass tensors;
- ``7_char``: the TRIMs with LORBIT 10: element and s/p/d character of the band edges.

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python cbm_package.py \\
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

from samson_mlip_visualizer.vasp_labeling import incar, kpoint_mesh

HERE = Path(__file__).parent
SPACING = 0.25
RELAX_LEVELS = ("1_relax", "2_relax", "3_static", "4_rattle")
BAND_LEVELS = ("5_edges", "6_mass", "7_char")
CHAIN = {
    "1_relax": "poscar",
    "2_relax": "contcar",
    "3_static": "contcar",
    "4_rattle": ("rattle", 0.05, 1),
    "5_edges": "last",
    "6_mass": "last",
    "7_char": "last",
}

p = argparse.ArgumentParser()
p.add_argument("--source19", type=Path, default=SOURCE19)
p.add_argument("--out", type=Path, default=WORK / "cbm_run")
p.add_argument(
    "--copy",
    type=lambda s: Path(s) if s else None,
    default=Path(
        r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests\29_vasp_bbvo_cbm_screen"
    ),
)
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg19 = json.loads((args.source19 / "package.json").read_text(encoding="utf-8"))
by_name = {f["name"]: f for f in pkg19["frames"]}


def from19(name, level):
    """A 19 geometry (CONTCAR of ``level``), refusing an unfinished run."""
    f = by_name[name]
    out = args.source19 / "outputs" / f"frame_{f['frame']:04d}" / level
    xml = (out / "vasprun.xml").read_text(errors="ignore")
    if "</modeling>" not in xml[-200:]:
        sys.exit(f"19 {name} {level} did not finish")
    contcar = out / "CONTCAR"
    return grouped(read(contcar, format="vasp")), contcar


BAND = {
    "NSW": 0,
    "IBRION": -1,
    "ISIF": 2,
    "ISYM": 0,
    "EDIFF": "1E-7",
    "NCORE": 4,
    "KPAR": 4,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}


def band_inputs(folder, atoms, mesh, name):
    """KPOINTS.<level> and their labels for the three band levels; returns the INCAR tags."""
    n_uniform = 8 if len(atoms) <= 10 else 4
    edges = [(k, f"uniform {n_uniform}") for k in bk.uniform(n_uniform)] + bk.path()
    mass = [pt for label, k in bk.trims().items() for pt in bk.stencil(atoms.cell, k, label)]
    char = [(k, label) for label, k in bk.trims().items()]
    for level, extra in zip(BAND_LEVELS, (edges, mass, char), strict=True):
        labels = bk.write_explicit(folder / f"KPOINTS.{level}", mesh, extra, f"{name} {level}")
        bk.write_labels(folder, level, labels)
    nb = nbands(atoms)
    return {
        "5_edges": {**BAND, "NBANDS": nb, "LORBIT": 0},
        "6_mass": {**BAND, "NBANDS": nb, "LORBIT": 0},
        "7_char": {**BAND, "NBANDS": nb, "LORBIT": 10},
    }, {"5_edges": len(edges), "6_mass": len(mass), "7_char": len(char)}


plan = [("R3_polar", "relaxed", "4_rattle")]
for name in ("x0_cubic40", "Nb0.25", "Nb0.5", "Nb1", "Ta0.25", "Ta0.5", "Ta1"):
    plan.append((name, "relaxed", "4_rattle"))
for name in ("x0_cubic40", "Nb0.25", "Nb0.5", "Nb1", "Ta0.25", "Ta0.5", "Ta1"):
    plan.append((name, "cubic_ref", "3_static"))
for metal in ("Nb", "Ta"):
    plan.append(
        (
            f"{metal}0.75",
            "relaxed",
            HERE / "poscars" / f"Ba2Bi_V0.25{metal}0.75_O6_x0.75_MACE-MP-0" / "POSCAR",
        )
    )

frames = []
for name, kind, src in plan:
    relax_here = isinstance(src, Path)
    if relax_here:
        atoms, path_ = grouped(read(src, format="vasp")), src
    else:
        atoms, path_ = from19(name, src)
    mesh = kpoint_mesh(atoms, SPACING)
    title = f"{name}_{kind}"
    index = len(frames)
    folder = args.out / "inputs" / f"frame_{index:04d}"
    levels = {}
    if relax_here:  # 19's chain and settings, symmetry kept as for 19's doped seeds
        levels = {
            "1_relax": {**pkg19["relax"], "ISYM": 2},
            "2_relax": {**pkg19["relax"], "ISYM": 2},
            "3_static": {**pkg19["static"], "ISYM": 2},
            "4_rattle": pkg19["rattle"],
        }
    write_frame(args.out, index, atoms, levels, SPACING, mesh=mesh, title=title)
    tags, counts = band_inputs(folder, atoms, mesh, title)
    for level, extra in tags.items():
        text = incar("pbe_u", atoms, extra=extra).replace("single point", f"{title} {level}")
        (folder / f"INCAR.{level}").write_text(text, newline="\n")
    frames.append(
        {
            "frame": index,
            "name": name,
            "kind": kind,
            "natoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "kmesh": list(mesh),
            "nbands": tags["5_edges"]["NBANDS"],
            "zero_weight_points": counts,
            "geometry": (
                "relaxed here: 19's chain from the MACE-MP-0 seed"
                if relax_here
                else f"19 {name} {src} CONTCAR"
            ),
            "source_file": str(path_),
            "source_sha256": hashlib.sha256(Path(path_).read_bytes()).hexdigest(),
        }
    )
    print(
        f"frame {index}: {title} {atoms.get_chemical_formula()} mesh {mesh} "
        f"NBANDS {tags['5_edges']['NBANDS']} + {counts} zero-weight k"
    )

write_script(
    args.out,
    len(frames),
    CHAIN,
    name="bbvo-cbm",
    time="12:00:00",
    throttle=6,
    keep=("EIGENVAL",),
    level_kpoints=True,
)
write_manifest(
    args.out,
    frames,
    {
        "package": "29_vasp_bbvo_cbm_screen",
        "kspacing": SPACING,
        "chain": {k: str(v) for k, v in CHAIN.items()},
        "relax": pkg19["relax"],
        "static": pkg19["static"],
        "rattle": pkg19["rattle"],
        "band": BAND,
        "stencil_steps_A-1": list(bk.STEPS),
        "source19": str(args.source19),
        "generator": "examples/delta_hse06_bbvo/cbm_package.py",
        "analyzer": "examples/delta_hse06_bbvo/cbm_analyze.py",
    },
)

table = "\n".join(
    f"| {f['frame']} | {f['name']} | {f['kind']} | {f['natoms']} | "
    f"{'×'.join(map(str, f['kmesh']))} | {f['nbands']} | {f['geometry']} |"
    for f in frames
)
(args.out / "README.md").write_text(
    f"""# 29_vasp_bbvo_cbm_screen: CBM dispersion of Ba2Bi(V,Nb,Ta)O6 at relaxed geometries (PBE+U)

Written by `examples/delta_hse06_bbvo/cbm_package.py` (Samson_MLIP_Visualizer). \
Nothing was submitted.
Owner: Samson_MLIP_Visualizer. Purpose: the primary dopant criterion, CBM dispersion (band edges,
gap, electron and hole mass tensors, edge character), at the symmetry-free relaxed cells, with the
symmetric cubic-derived cells as the labelled reference. Not gated on phonons.

| Frame | Structure | Geometry | Atoms | SCF mesh | NBANDS | Source |
|---|---|---|---|---|---|---|
{table}

Levels (dispatcher `run_vasp.slurm` layout, `vasp6/6.6.1-cpu`, one 64-core node per frame):
- frames 15-16 only: `1_relax` -> `2_relax` -> `3_static` -> `4_rattle` (package 19's chain);
- every frame: `5_edges`, `6_mass`, `7_char`, each one SCF from the frame's geometry (or the
  `4_rattle` CONTCAR) with an explicit `KPOINTS.<level>`: the SCF mesh at weight 1 plus
  zero-weight band points. The labels of every k-point are in
  `inputs/frame_NNNN/KPOINTS.<level>.labels.json`.

A frame is done when every level it has wrote a complete `vasprun.xml`; `EIGENVAL` is copied
back too.

Cost (19's timings: a 40-atom PBE+U SCF ~7 s per k-point): the band levels of a 40-atom frame
are ~160 + ~230 + ~35 k-points, about 1 h per frame; the two x = 0.75 relaxations ~4 h each.
About 20-25 node-hours in all; 12 h limit per task, 6 tasks at once.

Check: `PYTHONPATH=../../src micromamba run -n defects python \
examples/delta_hse06_bbvo/cbm_analyze.py <this folder>`
(every level finished, EDIFF reached; gap and CBM/VBM location; whether the edges sit at a TRIM;
electron/hole mass tensors there with a parabolicity check; edge character; relaxed vs cubic
per composition).

Caveats: U is on V only, so the x = 1 end members are plain PBE; gaps are PBE+U and too small
(HSE06 is package 30). SOC is not included here (the manuscript reports SOC changing the cubic CB
mass by ~2×): HSE06+SOC or PBE+U+SOC masses on the best candidate are the follow-up.
""",
    encoding="utf-8",
)
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + f": {len(frames)} frames")
