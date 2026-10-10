"""Packages 42-44: the Ba₂ScVO₆ "Sc-B" set (PBE+U, HSE06), for LONI; nothing is submitted.
-> scb_run/42_..., 43_..., 44_... (also loni_smoke_tests/batch10_2026-10-09/)

PI-approved on 2026-10-10, ≤ 5 node-hours in all, to run in one batch with package 41
(bbvo_analysis_review handoffs/to_Samson_MLIP_Visualizer_2026-10-10_cmc21_isif3.md, part 2).

Common geometry: 37's ref_BiSc1 final structure (Imm2 at 0.01 Å, Fm-3m at 0.1 Å) symmetrized to
Fm-3m and written as the 10-atom primitive cell in package 18's basis (a1 = (0, h, h), ...), so
q-points and the conventional supercell matrix are 18's. Every frame that needs the relaxed cell
relaxes it itself first (``1_relax``, ``2_relax``: 19's relax tags, ISIF 3, ISYM 2, 0.25 Å⁻¹):
identical inputs in every frame, about a minute each, and no frame waits for another.

- 42_vasp_bbvo_scb_electronic:
  - frame 0 (common + B1): the relax, ``3_static`` (19's static tags; E per f.u. against 37's
    ref_BiSc1), then package 30's hybrid protocol without SOC: ``a_edges`` (PBE+U, zero-weight
    band points), ``b_pbeu_mesh`` (PBE+U on the SCF mesh, WAVECAR kept), ``c_hse06``;
  - frame 1 (B3): 31 frame 12's rattle-relaxed BiSc1 (the cell its masses came from; 37's
    ref_BiSc1 differs by ≤ 0.0075 Å) on 31's 3×3×3 mesh: ``6_mass``, 31's stencil at Γ only,
    and ``7_char`` at Γ, both LORBIT 11 (m-resolved d weights).
- 43_vasp_bbvo_scb_optics (B2): the relax, then 38's LOPTICS level on Γ 12×12×12; the scissor
  is B1's HSE06 − PBE+U on the mesh, read from 42 by optics_analyze.py.
- 44_vasp_bbvo_scb_phonons (A2): package 18's sc40 protocol (±0.01 Å, phonopy, EDIFF 1e-8,
  ISYM 0, Γ 3×3×3). Each frame relaxes the 10-atom cell, then ``supercell.awk`` builds its
  displaced 40-atom supercell from that CONTCAR and a per-frame template (primitive index,
  translation, Cartesian displacement): the displacement pattern is fixed by Fm-3m symmetry, so
  only the positions follow the relaxed cell. The 80-atom set (L) is deferred: with it the batch
  estimate (≈ 4.8-5.2 nh) reaches the 5 nh budget.

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python scb_package.py [--out DIR] [--copy DIR|'']
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
from ase import Atoms
from ase.io import read
from band_kpoints import SOURCE19, ZVAL, nbands
from common import WORK, grouped
from loni_chain import write_frame, write_manifest, write_script
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from samson_mlip_visualizer.vasp_labeling import incar, kpoint_mesh

SPACING = 0.25
BATCH = "batch10_2026-10-09"
SOURCE37 = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests") / BATCH / "37_vasp_bbvo_sc_polymorphs"
SOURCE31 = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests\batch09_2026-10-07\31_vasp_bbvo_levers")
NAMES = {42: "42_vasp_bbvo_scb_electronic", 43: "43_vasp_bbvo_scb_optics",
         44: "44_vasp_bbvo_scb_phonons"}
M_CONV = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]  # 18's sc40: primitive -> 40-atom cubic cell
# package 30's hybrid levels (hybrid_soc_package.py BASE), no SOC
HYB = {"NSW": 0, "IBRION": -1, "ISIF": 2, "EDIFF": "1E-6", "NCORE": 4, "KPAR": 4,
       "LWAVE": ".FALSE.", "LCHARG": ".FALSE.", "LORBIT": 0}
# package 38's optics level (optics_package.py OPTICS)
OPTICS = {"NSW": 0, "IBRION": -1, "ISIF": 2, "EDIFF": "1E-7", "LOPTICS": ".TRUE.",
          "CSHIFT": 0.05, "NEDOS": 4000, "ISMEAR": 0, "SIGMA": 0.01, "NCORE": 1, "KPAR": 8,
          "LWAVE": ".FALSE.", "LCHARG": ".FALSE.", "LORBIT": 0}
# package 18's phonon single points (phonon_package.py TAGS)
PHON = {"EDIFF": "1E-8", "ADDGRID": ".FALSE.", "NCORE": 4, "KPAR": 4, "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE.", "ISYM": 0}

SUPERCELL_AWK = r'''
# supercell.awk CONTCAR TEMPLATE: a displaced phonon supercell built from a relaxed primitive cell.
# CONTCAR: VASP5, Direct. TEMPLATE: "M" rows (integer supercell matrix, rows: S = M P), "MINV" rows,
# "SPECIES", "COUNTS", then per supercell atom "ATOM p t1 t2 t3 dx dy dz": primitive atom p
# (1-based), lattice translation t, Cartesian displacement in A. Fractional: (f_p + t) M^-1 + d S^-1.
FNR == NR {
  if (FNR == 2) s = $1
  if (FNR >= 3 && FNR <= 5) for (j = 1; j <= 3; j++) P[FNR - 2, j] = $j * s
  if (FNR == 7) { n = 0; for (j = 1; j <= NF; j++) n += $j }
  if (FNR == 8 && $1 !~ /^[Dd]/) { print "supercell.awk: CONTCAR must be Direct" > "/dev/stderr"; bad = 1; exit 2 }
  if (FNR > 8 && FNR <= 8 + n) for (j = 1; j <= 3; j++) F[FNR - 8, j] = $j
  next
}
$1 == "M"       { r++; for (j = 1; j <= 3; j++) M[r, j] = $(j + 1); next }
$1 == "MINV"    { q++; for (j = 1; j <= 3; j++) Mi[q, j] = $(j + 1); next }
$1 == "SPECIES" { sp = $0; sub(/^SPECIES +/, "", sp); next }
$1 == "COUNTS"  { ct = $0; sub(/^COUNTS +/, "", ct); next }
$1 == "ATOM"    { k++; p[k] = $2; for (j = 1; j <= 3; j++) { t[k, j] = $(j + 2); d[k, j] = $(j + 5) }; next }
END {
  if (bad) exit 2
  if (n == 0 || k == 0 || r != 3 || q != 3) { print "supercell.awk: incomplete CONTCAR or template" > "/dev/stderr"; exit 2 }
  for (i = 1; i <= 3; i++) for (j = 1; j <= 3; j++) { S[i, j] = 0; for (l = 1; l <= 3; l++) S[i, j] += M[i, l] * P[l, j] }
  det = S[1,1]*(S[2,2]*S[3,3]-S[2,3]*S[3,2]) - S[1,2]*(S[2,1]*S[3,3]-S[2,3]*S[3,1]) + S[1,3]*(S[2,1]*S[3,2]-S[2,2]*S[3,1])
  B[1,1] =  (S[2,2]*S[3,3]-S[2,3]*S[3,2])/det; B[1,2] = -(S[1,2]*S[3,3]-S[1,3]*S[3,2])/det; B[1,3] =  (S[1,2]*S[2,3]-S[1,3]*S[2,2])/det
  B[2,1] = -(S[2,1]*S[3,3]-S[2,3]*S[3,1])/det; B[2,2] =  (S[1,1]*S[3,3]-S[1,3]*S[3,1])/det; B[2,3] = -(S[1,1]*S[2,3]-S[1,3]*S[2,1])/det
  B[3,1] =  (S[2,1]*S[3,2]-S[2,2]*S[3,1])/det; B[3,2] = -(S[1,1]*S[3,2]-S[1,2]*S[3,1])/det; B[3,3] =  (S[1,1]*S[2,2]-S[1,2]*S[2,1])/det
  print "phonon supercell from the relaxed primitive cell (supercell.awk)"
  print "1.0"
  for (i = 1; i <= 3; i++) printf "%22.14f %22.14f %22.14f\n", S[i, 1], S[i, 2], S[i, 3]
  print sp; print ct; print "Direct"
  for (a = 1; a <= k; a++) {
    for (j = 1; j <= 3; j++) {
      g = 0
      for (l = 1; l <= 3; l++) g += (F[p[a], l] + t[a, l]) * Mi[l, j] + d[a, l] * B[l, j]
      g -= int(g); if (g < 0) g += 1
      x[j] = g
    }
    printf "%18.14f %18.14f %18.14f\n", x[1], x[2], x[3]
  }
}
'''  # noqa: E501

p = argparse.ArgumentParser()
p.add_argument("--source19", type=Path, default=SOURCE19)
p.add_argument("--source31", type=Path, default=SOURCE31)
p.add_argument("--source37", type=Path, default=SOURCE37)
p.add_argument("--out", type=Path, default=WORK / "scb_run")
p.add_argument("--copy", type=lambda s: Path(s) if s else None,
               default=Path(r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests") / BATCH)
args = p.parse_args()
for d in [args.out / n for n in NAMES.values()] + (
        [args.copy / n for n in NAMES.values()] if args.copy else []):
    if d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg19 = json.loads((args.source19 / "package.json").read_text(encoding="utf-8"))
pkg31 = json.loads((args.source31 / "package.json").read_text(encoding="utf-8"))
RELAX = {**pkg19["relax"], "ISYM": 2}
STATIC = {**pkg19["static"], "ISYM": 2}


def finished(out, what):
    if "</modeling>" not in (out / "vasprun.xml").read_text(errors="ignore")[-200:]:
        sys.exit(f"{what} did not finish")
    return out / "CONTCAR"


def fm3m_primitive(src):
    """37's structure symmetrized to Fm-3m (0.1 Å), as a 10-atom cell in package 18's basis."""
    s = AseAtomsAdaptor.get_structure(read(src, format="vasp"))
    sga = SpacegroupAnalyzer(s, symprec=0.1)
    if sga.get_space_group_symbol() != "Fm-3m":
        sys.exit(f"{src}: {sga.get_space_group_symbol()} at 0.1 A, expected Fm-3m")
    conv = sga.get_refined_structure()  # conventional cubic cell, symmetrized
    a = conv.lattice.a
    h = a / 2
    prim = np.array([[0, h, h], [h, 0, h], [h, h, 0]])
    inv = np.linalg.inv(prim)
    sites = {}
    for site in conv:
        f = site.coords @ inv
        f -= np.floor(f + 1e-8)
        key = tuple(np.round(f, 6) % 1.0)
        sites.setdefault(key, (site.specie.symbol, f))
    if len(sites) != 10:
        sys.exit(f"Fm-3m primitive has {len(sites)} sites, expected 10")
    sym = [v[0] for v in sites.values()]
    return grouped(Atoms(sym, scaled_positions=[v[1] for v in sites.values()], cell=prim,
                         pbc=True)), a


def gamma_kpoints(path_, mesh, title):
    path_.write_text(f"Gamma-centered {title}\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n",
                     newline="\n")


def readme(root, text):
    (root / "README.md").write_text(text, encoding="utf-8", newline="\n")


src37 = finished(args.source37 / "outputs" / "frame_0002" / "4_rattle", "37 ref_BiSc1 4_rattle")
prim, a_conv = fm3m_primitive(src37)
mesh = kpoint_mesh(prim, SPACING)
start = {"geometry": "37 ref_BiSc1 4_rattle CONTCAR symmetrized to Fm-3m (0.1 A), 10-atom "
                     "primitive in package 18's basis", "a_conv_start_A": round(a_conv, 5),
         "source_file": str(src37),
         "source_sha256": hashlib.sha256(src37.read_bytes()).hexdigest()}
RELAX_LEVELS = {"1_relax": RELAX, "2_relax": RELAX}
print(f"Fm-3m start: a_conv {a_conv:.4f} A, {prim.get_chemical_formula()}, relax mesh {mesh}")
common_meta = {"relax": RELAX, "kspacing": SPACING, "start": start,
               "budget_nh": "Sc-B set <= 5 in all (PI, 2026-10-10)",
               "generator": "examples/delta_hse06_bbvo/scb_package.py"}

# ---- 42: common geometry + B1 (frame 0), B3 (frame 1) ----
root = args.out / NAMES[42]
nb = nbands(prim)
write_frame(root, 0, prim, {**RELAX_LEVELS, "3_static": STATIC}, SPACING, mesh=mesh,
            title="Fm3m_B1")
folder = root / "inputs" / "frame_0000"
n_uniform = 8
extra = ([(k, f"uniform {n_uniform}") for k in bk.uniform(n_uniform)] + bk.path()
         + [(k, lab) for lab, k in bk.trims().items()])
bk.write_labels(folder, "a_edges",
                bk.write_explicit(folder / "KPOINTS.a_edges", mesh, extra, "Fm3m_B1 a_edges"))
inc = {"a_edges": incar("pbe_u", prim, extra={**HYB, "ISYM": 0, "EDIFF": "1E-7", "NBANDS": nb}),
       "b_pbeu_mesh": incar("pbe_u", prim, extra={**HYB, "ISYM": 2, "NBANDS": nb,
                                                  "LWAVE": ".TRUE."}),
       "c_hse06": incar("hse06", prim, extra={**HYB, "ISYM": 2, "NBANDS": nb})}
for level, text in inc.items():
    (folder / f"INCAR.{level}").write_text(text.replace("single point", f"Fm3m_B1 {level}"),
                                           newline="\n")
f31 = pkg31["frames"][12]
if f31["name"] != "BiSc1":
    sys.exit(f"31 frame 12 is {f31['name']}, expected BiSc1")
src31 = finished(args.source31 / "outputs" / "frame_0012" / "4_rattle", "31 BiSc1 4_rattle")
b3 = grouped(read(src31, format="vasp"))
mesh31 = tuple(f31["kmesh"])
write_frame(root, 1, b3, {}, None, mesh=mesh31, title="BiSc1_B3")
folder = root / "inputs" / "frame_0001"
gpt = bk.trims()["G"]
for level, pts in (("6_mass", bk.stencil(b3.cell, gpt, "G")), ("7_char", [(gpt, "G")])):
    bk.write_labels(folder, level, bk.write_explicit(folder / f"KPOINTS.{level}", mesh31, pts,
                                                     f"BiSc1_B3 {level}"))
    text = incar("pbe_u", b3, extra={**pkg31["band"], "NBANDS": f31["nbands"], "LORBIT": 11})
    (folder / f"INCAR.{level}").write_text(text.replace("single point", f"BiSc1_B3 {level}"),
                                           newline="\n")
CHAIN42 = {"1_relax": "poscar", "2_relax": "contcar", "3_static": "contcar", "a_edges": "last",
           "b_pbeu_mesh": "last", "c_hse06": "last", "6_mass": "last", "7_char": "last"}
write_script(root, 2, CHAIN42, name="bbvo-scb-elec", time="12:00:00", wavecar=("c_hse06",),
             keep=("EIGENVAL",), level_kpoints=True, resume=("1_relax", "2_relax"))
write_manifest(root, [
    {"frame": 0, "name": "Fm3m_B1", "items": ["common geometry", "B1"], "natoms": 10,
     "formula": prim.get_chemical_formula(), "kmesh": list(mesh), "nbands": nb,
     "levels": list(CHAIN42)[:6], **start},
    {"frame": 1, "name": "BiSc1_B3", "items": ["B3"], "natoms": len(b3),
     "formula": b3.get_chemical_formula(), "kmesh": list(mesh31), "nbands": f31["nbands"],
     "levels": ["6_mass", "7_char"],
     "geometry": "31 frame 12 (BiSc1) 4_rattle CONTCAR: the cell 31's masses came from; 37's "
                 "ref_BiSc1 differs by <= 0.0075 A", "source_file": str(src31),
     "source_sha256": hashlib.sha256(src31.read_bytes()).hexdigest()}],
    {"package": NAMES[42], **common_meta, "static": STATIC, "hybrid": HYB, "band": pkg31["band"],
     "chain": CHAIN42, "reference": "ref_BiSc1",
     "reference_package": f"{BATCH}/37_vasp_bbvo_sc_polymorphs",
     "analyzer": "examples/delta_hse06_bbvo/scb_analyze.py"})
readme(root, f"""# {NAMES[42]}: Ba2ScVO6 Fm-3m cell, HSE06 gap (B1) and orbital-resolved CBM (B3)

Written by `examples/delta_hse06_bbvo/scb_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
PI-approved on 2026-10-10 (bbvo_analysis_review, Sc-B set; <= 5 nh for 42-44 together).

| Frame | Name | Atoms | Mesh | Levels |
|---|---|---|---|---|
| 0 | Fm3m_B1 | 10 | {'×'.join(map(str, mesh))} | `1_relax` -> `2_relax` (ISIF 3, ISYM 2, 19's relax tags) -> `3_static` -> `a_edges` -> `b_pbeu_mesh` -> `c_hse06` |
| 1 | BiSc1_B3 | {len(b3)} | {'×'.join(map(str, mesh31))} | `6_mass` (31's stencil at Γ, LORBIT 11) -> `7_char` (Γ, LORBIT 11) |

- Frame 0 starts from 37's ref_BiSc1, symmetrized to Fm-3m (0.1 A, a = {a_conv:.4f} A conventional), in package 18's
  primitive basis. `3_static` gives E per f.u. against 37's ref_BiSc1 (expected within ~1 meV/f.u.). The hybrid levels
  are package 30's protocol without SOC: HSE06 gap estimate = converged PBE+U gap (`a_edges`) + (HSE06 - PBE+U) on the mesh.
- Frame 1 is 31 frame 12's rattle-relaxed BiSc1 on 31's mesh, so the stencil reproduces 31's Γ masses; LORBIT 11 gives
  the V d_xy / d_yz / d_xz weights of the lowest three conduction bands at each stencil point.

Cost: 30's cubic Ba2BiVO6 HSE06 took 1.9 h; frame 0 about 2-2.5 h. Frame 1 about 0.15 h. 12 h limit.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/scb_analyze.py <this folder>`.
""")  # noqa: E501
print(f"{NAMES[42]}: frame 0 Fm3m_B1 (NBANDS {nb}), frame 1 BiSc1_B3 ({len(b3)} at, mesh {mesh31})")

# ---- 43: B2 optics ----
root = args.out / NAMES[43]
nocc = sum(ZVAL[s] for s in prim.get_chemical_symbols()) / 2
nb_opt = 8 * math.ceil(3 * nocc / 8)
write_frame(root, 0, prim, {**RELAX_LEVELS, "3_optics": {**OPTICS, "ISYM": 2, "NBANDS": nb_opt}},
            SPACING, mesh=mesh, title="Fm3m_B2")
gamma_kpoints(root / "inputs" / "frame_0000" / "KPOINTS.3_optics", (12, 12, 12), "Fm3m_B2 optics")
CHAIN43 = {"1_relax": "poscar", "2_relax": "contcar", "3_optics": "last"}
write_script(root, 1, CHAIN43, name="bbvo-scb-optics", time="06:00:00", level_kpoints=True,
             resume=("1_relax", "2_relax"))
write_manifest(root, [{"frame": 0, "name": "Fm3m_B2", "phase": "Ba2ScVO6_Fm-3m", "natoms": 10,
                       "formula": prim.get_chemical_formula(), "kmesh": [12, 12, 12],
                       "isym": 2, "nbands": nb_opt, "scissor_eV": None, **start}],
               {"package": NAMES[43], **common_meta, "optics": OPTICS, "chain": CHAIN43,
                "optics_level": "3_optics",
                "scissor_from": {"package": f"{BATCH}/{NAMES[42]}", "frame": 0,
                                 "pbeu": "b_pbeu_mesh", "hse": "c_hse06"},
                "analyzer": "examples/delta_hse06_bbvo/optics_analyze.py"})
readme(root, f"""# {NAMES[43]}: PBE+U dielectric function of Fm-3m Ba2ScVO6 (B2, LOPTICS)

Written by `examples/delta_hse06_bbvo/scb_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
PI-approved on 2026-10-10 (bbvo_analysis_review, Sc-B set).

One frame: the 10-atom Fm-3m cell (same start and relax as 42 frame 0), then `3_optics`: package 38's tags (LOPTICS,
CSHIFT 0.05, NEDOS 4000, ISMEAR 0 / SIGMA 0.01, NCORE 1 / KPAR 8), NBANDS {nb_opt} (3x the {nocc:.0f} occupied), Γ 12x12x12.
The scissor (HSE06 - PBE+U on the mesh) comes from 42 frame 0 (B1); optics_analyze.py reads it from 42 next to this
package in the results clone, so run it after 42 is back. Cost: about 0.1 h.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/optics_analyze.py <this folder>`.
""")  # noqa: E501
print(f"{NAMES[43]}: Fm3m_B2, NBANDS {nb_opt}, 12x12x12")

# ---- 44: A2 phonons, sc40 on the relaxed cell ----
root = args.out / NAMES[44]
ph = Phonopy(PhonopyAtoms(symbols=prim.get_chemical_symbols(), cell=prim.cell[:],
                          scaled_positions=prim.get_scaled_positions()),
             supercell_matrix=M_CONV, primitive_matrix=np.eye(3))
ph.generate_displacements(distance=0.01, is_plusminus=True)
(root / "phonopy").mkdir(parents=True)
ph.save(root / "phonopy" / "sc40.yaml")
sup = ph.supercell
Mrow = np.round(sup.cell @ np.linalg.inv(prim.cell[:])).astype(int)
if not np.allclose(Mrow @ prim.cell[:], sup.cell, atol=1e-8) or not np.allclose(
        sup.cell, np.diag(np.diag(sup.cell)), atol=1e-6):
    sys.exit("phonopy supercell is not the 40-atom cubic cell M P")
Minv = np.linalg.inv(Mrow)
fp = prim.get_scaled_positions()
species, counts = [], []
for s in sup.symbols:
    if not species or species[-1] != s:
        species.append(s)
        counts.append(0)
    counts[-1] += 1
if len(species) != len(set(species)) or species != list(dict.fromkeys(prim.get_chemical_symbols())):
    sys.exit("phonopy supercell is not grouped in the primitive's species order")
mapping = []  # supercell atom -> (primitive index, translation)
for g in sup.scaled_positions:
    fr = g @ Mrow
    diff = fr - fp
    j = int(np.argmin(np.abs(diff - np.round(diff)).max(axis=1)))
    if np.abs(diff[j] - np.round(diff[j])).max() > 1e-6:
        sys.exit("supercell atom does not map onto the primitive cell")
    mapping.append((j, np.round(diff[j]).astype(int)))
frames = []
for i, disp in enumerate(ph.supercells_with_displacements):
    delta = (disp.scaled_positions - sup.scaled_positions)
    delta -= np.round(delta)
    delta = delta @ sup.cell
    write_frame(root, i, prim, {**RELAX_LEVELS, "pbe_u": PHON}, SPACING, mesh=mesh,
                title=f"sc40 disp {i + 1}")
    folder = root / "inputs" / f"frame_{i:04d}"
    gamma_kpoints(folder / "KPOINTS.pbe_u", (3, 3, 3), f"sc40 disp {i + 1}")
    lines = [f"M {r[0]} {r[1]} {r[2]}" for r in Mrow]
    lines += [f"MINV {r[0]:.16f} {r[1]:.16f} {r[2]:.16f}" for r in Minv]
    lines += ["SPECIES " + " ".join(species), "COUNTS " + " ".join(map(str, counts))]
    lines += [f"ATOM {j + 1} {t[0]} {t[1]} {t[2]} {d[0]:.12f} {d[1]:.12f} {d[2]:.12f}"
              for (j, t), d in zip(mapping, delta, strict=True)]
    (folder / "PHONON.template").write_text("\n".join(lines) + "\n", newline="\n")
    frames.append({"frame": i, "set": "sc40", "displacement": i + 1})
CHAIN44 = {"1_relax": "poscar", "2_relax": "contcar",
           "pbe_u": ("awk", "supercell.awk", "PHONON.template")}
write_script(root, len(frames), CHAIN44, name="bbvo-scb-phonons", time="04:00:00", throttle=8,
             level_kpoints=True, resume=("1_relax", "2_relax"),
             scripts={"supercell.awk": SUPERCELL_AWK})
write_manifest(root, frames, {
    "package": NAMES[44], **common_meta, "incar_extra": PHON,
    "chain": {k: str(v) for k, v in CHAIN44.items()},
    "title": "Fm-3m Ba$_2$ScVO$_6$",
    "sets": {"sc40": {"supercell": "sc40", "lattice_scale": 1.0, "atoms": len(sup.symbols),
                      "frames": [0, len(frames) - 1], "kmesh": [3, 3, 3],
                      "phonopy_yaml": "phonopy/sc40.yaml", "unitcell_from": "2_relax",
                      "displaced_poscar_from": "outputs/frame_NNNN/pbe_u/POSCAR"}},
    "deferred": "sc80 (L point): with it the batch estimate reaches the 5 nh budget",
    "analyzer": "examples/delta_hse06_bbvo/phonon_analyze.py"})
readme(root, f"""# {NAMES[44]}: PBE+U phonons of Fm-3m Ba2ScVO6, 40-atom cell (A2: Γ, X)

Written by `examples/delta_hse06_bbvo/scb_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
PI-approved on 2026-10-10 (bbvo_analysis_review, Sc-B set). Package 18's sc40 protocol: phonopy ±0.01 A,
EDIFF 1e-8, ADDGRID .FALSE., ISYM 0, Γ 3x3x3; {len(frames)} displacements.

Each frame first relaxes the 10-atom Fm-3m cell (`1_relax`, `2_relax`; identical in every frame and in 42/43), then
`supercell.awk` builds its displaced 40-atom supercell from that CONTCAR and `inputs/frame_NNNN/PHONON.template`
(primitive index, lattice translation and Cartesian displacement per supercell atom; the pattern is fixed by Fm-3m
symmetry, only the positions follow the relaxed cell). The POSCAR actually run is copied to `outputs/.../pbe_u/`.
The 80-atom cell (L) is deferred: with it the batch estimate (4.8-5.2 nh) reaches the 5 nh budget.

Cost: 18's sc40 took 0.24 h for 8 frames; with the relaxes about 0.5 h. 4 h limit.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/phonon_analyze.py <this folder>`:
the 2_relax cells agree across frames, the POSCARs run equal phonopy's displaced supercells of the relaxed cell,
frequencies at Γ and X with the character of any imaginary mode.
""")  # noqa: E501
print(f"{NAMES[44]}: {len(frames)} displacements, supercell {len(sup.symbols)} atoms, "
      f"species {species} {counts}")

if args.copy:
    for n in NAMES.values():
        shutil.copytree(args.out / n, args.copy / n)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else ""))
