"""Analyse package 19 (Ba₂BiVO₆ polymorphs + Nb/Ta stabilization, consistent PBE+U).

    PYTHONPATH=../../src micromamba run -n defects python polymorph_analyze.py \
        PACKAGE [--out DIR] [--partial]

Per frame and level it checks, and refuses to use a level that fails (``--partial`` reports the
rest instead of stopping):
- relaxations (1_relax, 2_relax, 4_rattle): OUTCAR finished and "reached required accuracy"
  (IBRION 2), i.e. EDIFFG met rather than NSW exhausted; every ionic step's SCF reached EDIFF;
- the static (3_static): ``parse_vasp_run`` (finished, EDIFF reached); its e_fr_energy is the
  comparison energy; the gap is read from its eigenvalues for both spin channels, and any
  occupation between 0.01 and 0.99 is flagged (a gap from fractional occupations is not reported).

Reports energies per formula unit (10 atoms) relative to the lowest frame **of the same
composition**, the rattle drop E(4_rattle) − E(3_static), space group (0.01 / 0.1 Å) and V/B-site
coordination of the static geometry, volume per f.u., and the k-sampling cross-check between the
10- and 40-atom cubic cells. A package whose package.json names a ``reference`` frame (37:
Ba₂ScVO₆) also gets energies relative to it. Nothing here is a hull energy.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from samson_mlip_visualizer.vasp_labeling import OutputError, parse_vasp_run

p = argparse.ArgumentParser()
p.add_argument("package", type=Path)
p.add_argument("--out", type=Path)
p.add_argument("--partial", action="store_true")
a = p.parse_args()
meta = json.loads((a.package / "package.json").read_text(encoding="utf-8"))
out = a.out or a.package / "analysis"
out.mkdir(parents=True, exist_ok=True)
problems = []


def fail(msg):
    if a.partial:
        problems.append(msg)
        return None
    sys.exit(msg)


def relax_ok(folder):
    t = ((folder / "OUTCAR").read_text(encoding="utf-8", errors="ignore")
         if (folder / "OUTCAR").exists() else "")
    if "General timing and accounting" not in t:
        return "OUTCAR missing or unfinished"
    if "reached required accuracy" not in t:
        return "EDIFFG not reached (NSW exhausted?)"
    ediff = t.count("aborting loop because EDIFF is reached")
    ionic = len(re.findall(r"^\s*-+\s*Iteration\s+(\d+)\(\s*1\)", t, re.M))
    if ediff < ionic:
        return f"{ionic - ediff} ionic step(s) with SCF not converged"
    return None


def gap(vasprun):
    root = ET.parse(vasprun).getroot()
    calc = root.findall(".//calculation")[-1]
    vb, cb, frac = -np.inf, np.inf, 0
    for spin in calc.find("eigenvalues/array/set").findall("set"):
        for k in spin.findall("set"):
            eo = np.array([[float(x) for x in r.text.split()] for r in k.findall("r")])
            frac += int(((eo[:, 1] > 0.01) & (eo[:, 1] < 0.99)).sum())
            vb = max(vb, eo[eo[:, 1] >= 0.5, 0].max())
            cb = min(cb, eo[eo[:, 1] < 0.5, 0].min())
    return (None if frac else float(cb - vb)), frac


rows = []
for f in meta["frames"]:
    base = a.package / "outputs" / f"frame_{f['frame']:04d}"
    row = {k: f[k] for k in ("frame", "name", "group", "formula", "natoms")}
    bad = False
    for lvl in ("1_relax", "2_relax"):
        why = relax_ok(base / lvl)
        if why:
            fail(f"{f['name']} {lvl}: {why}")
            bad = True
    try:
        st = parse_vasp_run(base / "3_static")
    except OutputError as ex:
        fail(f"{f['name']} 3_static: {ex}")
        continue
    fu = f["natoms"] / 10
    row["E_static_eV_per_fu"] = st["energy"] / fu
    row["gap_mesh_eV"], row["fractional_occupations"] = gap(base / "3_static" / "vasprun.xml")
    s = Structure.from_file(str(base / "3_static" / "CONTCAR"))
    row["volume_per_fu_A3"] = s.volume / fu
    for tol in (0.01, 0.1):
        try:
            row[f"spg_{tol}"] = SpacegroupAnalyzer(s, symprec=tol).get_space_group_symbol()
        except Exception as ex:
            row[f"spg_{tol}"] = f"failed ({ex})"
    row["B_site_CN"] = {
        el: sorted({len([n for n in s.get_neighbors(site, 2.4) if n.specie.symbol == "O"])
                    for site in s if site.specie.symbol == el})
        for el in ("V", "Nb", "Ta") if el in s.composition}
    why = relax_ok(base / "4_rattle")
    if why:
        fail(f"{f['name']} 4_rattle: {why}")
    else:
        e4 = float(re.findall(r"free  energy\s+TOTEN\s+=\s+(-?\d+\.\d+)",
                              (base / "4_rattle" / "OUTCAR").read_text(errors="ignore"))[-1])
        row["rattle_drop_meV_per_fu"] = (e4 - st["energy"]) / fu * 1000
    row["incomplete_relax"] = bad
    rows.append(row)

groups = {}
for r in rows:
    groups.setdefault(Composition(r["formula"]).reduced_formula, []).append(r)
for _comp, rs in groups.items():
    e0 = min(r["E_static_eV_per_fu"] for r in rs)
    for r in rs:
        r["dE_meV_per_fu_vs_lowest_same_composition"] = (r["E_static_eV_per_fu"] - e0) * 1000
# a package may name its reference frame (37: the Ba2ScVO6 perovskite)
ref = next((r for r in rows if r["name"] == meta.get("reference")), None)
if meta.get("reference") and ref is None:
    fail(f"reference frame {meta['reference']} has no usable 3_static")
for r in rows if ref else ():
    if Composition(r["formula"]).reduced_formula == Composition(ref["formula"]).reduced_formula:
        r["dE_meV_per_fu_vs_reference"] = (r["E_static_eV_per_fu"]
                                           - ref["E_static_eV_per_fu"]) * 1000
kcheck = {r["name"]: r["E_static_eV_per_fu"] for r in rows
          if r["name"] in ("cubic_Fm-3m", "x0_cubic40")}
report = {"rows": rows, "problems": problems,
          "note": "relative polymorph energies at PBE+U (MP U, V only); not hull energies",
          "kpoint_crosscheck_meV_per_fu":
              (kcheck["x0_cubic40"] - kcheck["cubic_Fm-3m"]) * 1000 if len(kcheck) == 2 else None}
(out / "polymorph_report.json").write_text(json.dumps(report, indent=1, default=float),
                                           encoding="utf-8")
lines = ["| Frame | Name | Composition | ΔE (meV/f.u., same comp.) | Rattle drop (meV/f.u.) "
         "| SG 0.01/0.1 | B CN | V/f.u. | Gap (mesh) |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (Composition(r["formula"]).reduced_formula,
                                     r["dE_meV_per_fu_vs_lowest_same_composition"])):
    gap_s = "n/a (fractional occ.)" if r["gap_mesh_eV"] is None else f"{r['gap_mesh_eV']:.2f}"
    lines.append(f"| {r['frame']} | {r['name']} | {Composition(r['formula']).reduced_formula} | "
                 f"{r['dE_meV_per_fu_vs_lowest_same_composition']:.1f} | "
                 f"{r.get('rattle_drop_meV_per_fu', float('nan')):.1f} | "
                 f"{r['spg_0.01']} / {r['spg_0.1']} | {r['B_site_CN']} | "
                 f"{r['volume_per_fu_A3']:.1f} | {gap_s} |")
if ref:
    lines += ["", f"Relative to the reference {ref['name']} (meV/f.u.): "
              + ", ".join(f"{r['name']} {r['dE_meV_per_fu_vs_reference']:+.1f}"
                          for r in rows if "dE_meV_per_fu_vs_reference" in r)]
lines += ["", "k-sampling cross-check (40- vs 10-atom cubic, same structure): "
          f"{report['kpoint_crosscheck_meV_per_fu']} meV/f.u.",
          "Problems: " + ("; ".join(problems) if problems else "none")]
(out / "polymorph_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
