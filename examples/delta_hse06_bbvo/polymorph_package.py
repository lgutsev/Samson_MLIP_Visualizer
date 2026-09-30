"""Ba₂BiVO₆ polymorph shortlist and Nb/Ta stabilization tests at one consistent PBE+U level.
For LONI; nothing is submitted.  -> polymorphs_run/ (also hpc_smoke_tests/19_vasp_bbvo_polymorphs)

Distortions of the cubic perovskite need not exhaust the relevant structures: OQMD lists a 20-atom
Cmc2₁ Ba₂BiVO₆ (entry 1344250) about 16 meV/atom above its hull, against ~140 meV/atom for the
cubic double perovskite. Every candidate here is relaxed and then recomputed with identical
settings, so their **relative** energies (same composition) can be compared directly. Hull
energies cannot: those need the competing phases (Ba₂Bi₂O₅, Ba₃V₂O₈, Bi₂O₃, Ba₂BiV₃O11, ...)
computed the same way, plus the Materials Project mixing corrections.

Settings: MP POTCARs, U(V) = 3.25 eV (V only; none on Nb/Ta, as in the MP scheme), 520 eV,
PREC = Accurate, LREAL = .FALSE., Γ-centred mesh at 0.25 Å⁻¹, ISMEAR 0 / SIGMA 0.05.
Chain per frame (dispatcher run_vasp.slurm layout):

1. ``1_relax``: cell + ions (ISIF 3), symmetry kept for symmetric seeds (ISYM 2), off for P1;
2. ``2_relax``: again from the CONTCAR (a fresh basis for the new cell: Pulay stress);
3. ``3_static``: single point, EDIFF 1e-7, LORBIT 11 (the energy used for comparison is
   ``e_fr_energy`` of this step; gap from its eigenvalues);
4. ``4_rattle``: CONTCAR rattled by 0.05 Å (Gaussian, seed 1), ions relaxed with symmetry off at
   the fixed cell (ISIF 2). E(4) − E(3) < 0 means the symmetric relaxed structure is not a local
   minimum along some direction the rattle found; ≈ 0 is consistent with a minimum (not proof).

Usage (defects env, for pymatgen):
    PYTHONPATH=../../src micromamba run -n defects python polymorph_package.py \
        [--candidates polymorphs/candidates.json] [--out DIR] [--copy DIR]
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from ase.io import read
from common import WORK, grouped
from loni_chain import write_frame, write_manifest, write_script
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).parent
SPACING = 0.25
RELAX = {"NSW": 200, "IBRION": 2, "ISIF": 3, "EDIFF": "1E-6", "EDIFFG": -0.01,
         "NCORE": 4, "KPAR": 4, "LWAVE": ".FALSE.", "LCHARG": ".FALSE."}
STATIC = {"NSW": 0, "IBRION": -1, "ISIF": 2, "EDIFF": "1E-7", "LORBIT": 11,
          "NCORE": 4, "KPAR": 4, "LWAVE": ".FALSE.", "LCHARG": ".FALSE."}
RATTLE = {**RELAX, "ISIF": 2, "ISYM": 0, "NSW": 300}
CHAIN = {"1_relax": "poscar", "2_relax": "contcar", "3_static": "contcar", "4_rattle": ("rattle", 0.05, 1)}

p = argparse.ArgumentParser()
p.add_argument("--candidates", type=Path, default=HERE / "polymorphs" / "candidates.json")
p.add_argument("--out", type=Path, default=WORK / "polymorphs_run")
p.add_argument("--copy", type=Path, default=Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests\19_vasp_bbvo_polymorphs"))
args = p.parse_args()
for d in (args.out, args.copy):
    if d and d.exists():
        sys.exit(f"{d} exists; remove it to write the package again")
cands = json.loads(args.candidates.read_text(encoding="utf-8"))

frames, blocked = [], []
for c in cands:
    src = (args.candidates.parent / c["path"]) if c.get("path") else None
    if not src or not src.exists():
        blocked.append({**c, "reason": c.get("blocked", f"structure file missing: {src}")})
        print(f"BLOCKED {c['name']}: {blocked[-1]['reason']}")
        continue
    atoms = grouped(read(src))
    if c.get("symmetrize"):  # e.g. the 08 fixed-cell product -> its R3 primitive cell
        s = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(s, symprec=c["symmetrize"])
        s = sga.get_primitive_standard_structure()
        atoms = grouped(AseAtomsAdaptor.get_atoms(s))
        c["symmetrized_to"] = f"{sga.get_space_group_symbol()} ({sga.get_space_group_number()}), {len(atoms)} atoms"
    isym = c.get("isym", 2)
    levels = {"1_relax": {**RELAX, "ISYM": isym}, "2_relax": {**RELAX, "ISYM": isym},
              "3_static": {**STATIC, "ISYM": isym}, "4_rattle": RATTLE}
    mesh = write_frame(args.out, len(frames), atoms, levels, SPACING, title=c["name"])
    frames.append({"frame": len(frames), "name": c["name"], "group": c.get("group", "polymorph"),
                   "source": c.get("source"), "natoms": len(atoms), "formula": atoms.get_chemical_formula(),
                   "isym": isym, "kmesh": list(mesh), "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                   **({"symmetrized_to": c["symmetrized_to"]} if "symmetrized_to" in c else {})})
    print(f"frame {len(frames) - 1}: {c['name']} {atoms.get_chemical_formula()} ({len(atoms)} atoms) mesh {mesh} ISYM {isym}")

write_script(args.out, len(frames), CHAIN, name="bbvo-polymorphs", time="24:00:00")
write_manifest(args.out, frames, {"package": "19_vasp_bbvo_polymorphs", "kspacing": SPACING, "chain": {k: str(v) for k, v in CHAIN.items()},
                                  "relax": RELAX, "static": STATIC, "rattle": RATTLE, "blocked": blocked,
                                  "generator": "examples/delta_hse06_bbvo/polymorph_package.py",
                                  "energy_for_comparison": "3_static e_fr_energy per formula unit (10 atoms)"})
table = "\n".join(f"| {f['frame']} | {f['name']} | {f['group']} | {f['natoms']} | ISYM {f['isym']} | {'×'.join(map(str, f['kmesh']))} | {f['source']} |"
                  for f in frames)
btable = "\n".join(f"| {b['name']} | {b.get('source')} | {b['reason']} |" for b in blocked) or "| — | — | — |"
(args.out / "README.md").write_text(f"""# 19_vasp_bbvo_polymorphs: Ba2BiVO6 polymorph shortlist + Nb/Ta stabilization, consistent PBE+U

Written by `examples/delta_hse06_bbvo/polymorph_package.py` (Samson_MLIP_Visualizer). Nothing was submitted.
Chain per frame: `1_relax` (ISIF 3) -> `2_relax` (ISIF 3) -> `3_static` -> `4_rattle` (0.05 A, ISIF 2, ISYM 0).
Dispatcher layout: `run_vasp.slurm` (+ `rattle.awk`, plain bash/awk); a frame is done when all four
`outputs/frame_NNNN/<level>/vasprun.xml` are complete; a resubmitted task resumes after the last finished level.

| Frame | Candidate | Group | Atoms | Symmetry in relax | k-mesh | Source |
|---|---|---|---|---|---|---|
{table}

**Blocked (not in this package yet):**

| Candidate | Source | Reason |
|---|---|---|
{btable}

Energies to compare: `3_static` e_fr_energy per formula unit, **within one composition only**.
These are relative polymorph energies at PBE+U(MP U); they are not hull energies.

Cost (from smoke test 08 timings on one 64-core QB4 node: 40-atom PBE+U ISIF 2 relaxation
1.7 h for 185 steps, ISIF 3 1.05 h for 120 steps; single point 1.5–2 min): 40-atom frames
roughly 2.5–5 node-hours each, 10-atom frames under 1. The package total is in `10_CALCULATION_PLAN.md`
of the audit. Elapsed time depends on the queue; tasks run in parallel.

Route proposal (for the desk to add to `dispatch/routes.json`):
- agent: Samson_MLIP_Visualizer; source: examples/delta_hse06_bbvo/polymorph_package.py
- check: `python polymorph_analyze.py <this folder>`: every level converged (EDIFFG reached, NSW not hit),
  energies per f.u. by composition, rattle drop, space group, V coordination, gap on the mesh
- next: HSE06(+SOC) single points on the lowest two or three polymorphs; phonons of the lowest
""", encoding="utf-8")
if args.copy:
    shutil.copytree(args.out, args.copy)
print(f"-> {args.out}" + (f" and {args.copy}" if args.copy else "") + f": {len(frames)} frames, {len(blocked)} blocked")
