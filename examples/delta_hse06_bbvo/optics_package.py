"""Package 38: independent-particle dielectric function of cubic and R3 Ba₂BiVO₆, PBE+U LOPTICS.
For LONI; nothing is submitted.  -> optics_run/
(also loni_smoke_tests/batch10_2026-10-09/38_vasp_bbvo_optics)

No dielectric function exists for any phase, and the detailed-balance bound assumes strong edge
absorption. The PBE+U gaps are nearly direct, which is necessary but not sufficient. This set asks
whether a sub-µm film absorbs strongly within ~0.2 eV of the gap. Approved by the PI on
2026-10-09, budget <= 6 node-hours (bbvo_analysis_review
handoffs/to_Samson_MLIP_Visualizer_2026-10-09_polymorph_optics.md, item 2).

Frames (package 30's geometries, i.e. 19's 3_static CONTCARs; static, no relaxation):
- ``cubic_Fm-3m`` (30 frame 0) and ``R3_polar`` (30 frame 1), 10 atoms, Γ-centred 12×12×12;
- ``R3_polar_k14``: R3 again at 14×14×14, the k-convergence spot check.
The Ba₂ScVO₆ frame waits for package 37 (only if 37 keeps it); it is not in this batch.

INCAR: PBE+U as in 29/30 (MP U on V), LOPTICS, CSHIFT 0.05, NEDOS 4000, ISMEAR 0, SIGMA 0.01,
NBANDS ≈ 3× the occupied bands (a multiple of 8), 30's symmetry. NCORE 1 (the optical matrix
elements are computed per k-point; plane-wave distribution is not needed for 10 atoms).

Usage (defects env):
    PYTHONPATH=../../src micromamba run -n defects python optics_package.py \\
        [--out DIR] [--copy DIR|'']
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

from ase.io import read
from band_kpoints import ZVAL
from common import WORK, grouped
from loni_chain import write_frame, write_manifest, write_script

SOURCE30 = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests\30_vasp_bbvo_hybrid_soc")
CHAIN = {"1_optics": "poscar"}
OPTICS = {"NSW": 0, "IBRION": -1, "ISIF": 2, "EDIFF": "1E-7", "LOPTICS": ".TRUE.",
          "CSHIFT": 0.05, "NEDOS": 4000, "ISMEAR": 0, "SIGMA": 0.01, "NCORE": 1, "KPAR": 8,
          "LWAVE": ".FALSE.", "LCHARG": ".FALSE.", "LORBIT": 0}
# HSE06 - PBE+U gap differences from package 30 (scissor estimate, labelled as such)
SCISSOR = {"cubic_Fm-3m": 0.598, "R3_polar": 0.705}
PLAN = [("cubic_Fm-3m", 0, (12, 12, 12)), ("R3_polar", 1, (12, 12, 12)),
        ("R3_polar", 1, (14, 14, 14))]

p = argparse.ArgumentParser()
p.add_argument("--source30", type=Path, default=SOURCE30)
p.add_argument("--out", type=Path, default=WORK / "optics_run")
p.add_argument("--copy", type=lambda s: Path(s) if s else None,
               default=Path(r"C:\Users\lguts\OneDrive\Desktop\Test_Code\loni_smoke_tests"
                            r"\batch10_2026-10-09\38_vasp_bbvo_optics"))
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
pkg30 = json.loads((args.source30 / "package.json").read_text(encoding="utf-8"))


def nbands_optics(atoms):
    """About 3x the occupied bands, rounded up to a multiple of 8."""
    nocc = sum(ZVAL[s] for s in atoms.get_chemical_symbols()) / 2
    return 8 * math.ceil(3 * nocc / 8)


frames = []
for name, frame30, mesh in PLAN:
    f30 = pkg30["frames"][frame30]
    if (f30["name"], f30["kind"]) != (name, "hybrid"):
        sys.exit(f"30 frame {frame30} is {f30['name']} {f30['kind']}, expected {name} hybrid")
    src = args.source30 / "inputs" / f"frame_{frame30:04d}" / "POSCAR"
    atoms = grouped(read(src, format="vasp"))
    index = len(frames)
    title = name if mesh == (12, 12, 12) else f"{name}_k{mesh[0]}"
    nb = nbands_optics(atoms)
    tags = {**OPTICS, "ISYM": f30["isym_hse"], "NBANDS": nb}
    write_frame(args.out, index, atoms, {"1_optics": tags}, None, mesh=mesh, title=title)
    frames.append({"frame": index, "name": title, "phase": name, "natoms": len(atoms),
                   "formula": atoms.get_chemical_formula(), "kmesh": list(mesh),
                   "isym": f30["isym_hse"], "nbands": nb, "scissor_eV": SCISSOR[name],
                   "geometry": f"30 frame {frame30} {name} POSCAR ({f30['geometry']})",
                   "source_file": str(src),
                   "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    print(f"frame {index}: {title:14s} {atoms.get_chemical_formula()} mesh {mesh} "
          f"ISYM {f30['isym_hse']} NBANDS {nb}")

write_script(args.out, len(frames), CHAIN, name="bbvo-optics", time="06:00:00")
write_manifest(args.out, frames, {
    "package": "38_vasp_bbvo_optics", "chain": CHAIN, "optics": OPTICS,
    "scissor_eV": SCISSOR,
    "scissor_note": "HSE06 - PBE+U gap from package 30 (cubic 1.734 - 1.136, R3 2.349 - 1.644); "
                    "a rigid shift, a scissor estimate only",
    "budget_nh": "<= 6 (PI, 2026-10-09)", "later": "Ba2ScVO6 frame, only if package 37 keeps it",
    "generator": "examples/delta_hse06_bbvo/optics_package.py",
    "analyzer": "examples/delta_hse06_bbvo/optics_analyze.py"})
rows = "\n".join(f"| {f['frame']} | {f['name']} | {f['natoms']} | "
                 f"{'×'.join(map(str, f['kmesh']))} | ISYM {f['isym']} | {f['nbands']} | "
                 f"+{f['scissor_eV']:.3f} |" for f in frames)
(args.out / "README.md").write_text(f"""# 38_vasp_bbvo_optics: PBE+U dielectric function of cubic and R3 Ba2BiVO6 (LOPTICS)

Written by `examples/delta_hse06_bbvo/optics_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
Approved by the PI on 2026-10-09, budget <= 6 node-hours (bbvo_analysis_review, item A3 / P2).

| Frame | Name | Atoms | k-mesh (Γ) | Symmetry | NBANDS | Scissor (eV) |
|---|---|---|---|---|---|---|
{rows}

One static level, `1_optics` (`vasp6/6.6.1-cpu`, one 64-core node per frame): PBE+U as in 29/30 (MP POTCARs,
U on V only), `LOPTICS = .TRUE.`, `CSHIFT = 0.05`, `NEDOS = 4000`, `ISMEAR = 0`, `SIGMA = 0.01`, NBANDS about
3x occupied, NCORE 1, KPAR 8. Geometries: package 30's frames 0 and 1. Frame 2 repeats R3 at 14x14x14 as the
k-convergence spot check. No hybrid optics. The Ba2ScVO6 frame waits for package 37.

Cost: 30's 10-atom PBE+U runs with 792 k-points and 64 bands took 5 min; here roughly 0.1-0.5 h per frame,
under 2 node-hours in all. 6 h limit.

Check: `PYTHONPATH=../../src micromamba run -n defects python examples/delta_hse06_bbvo/optics_analyze.py <this folder>`:
EDIFF reached, the dielectric function present in vasprun.xml; eps1, eps2 and alpha(E) 0-5 eV; PBE+U gap
(fundamental and direct on the mesh); alpha at gap + 0.1, 0.2, 0.5 eV and the single-pass absorptance of a
500 nm film, unshifted and with the scissor shift (labelled as an estimate); the 12 vs 14 mesh difference for R3.
""", encoding="utf-8", newline="\n")  # noqa: E501  (LF: read on LONI)
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + f": {len(frames)} frames")
