"""Analyse package 30 (HSE06 gaps of the low polymorphs; SOC on the 19 shortlist).

    PYTHONPATH=../../src micromamba run -n defects python hybrid_soc_analyze.py PACKAGE [--out DIR]

- hybrid frames: the converged PBE+U gap and edge locations (``a_edges``), the PBE+U and HSE06
  gaps on 19's mesh (``b_pbeu_mesh``, ``c_hse06``), and the HSE06 estimate
  gap(a) + [gap(c) − gap(b)];
- soc frames: ΔE_SOC = E(e_pbeu_soc) − E(d_pbeu) per formula unit, the order within each
  composition with and without SOC, and the mesh gaps with and without SOC;
- the HSE06+SOC cost frame: its gap on the mesh and the elapsed time of each level.

Every level must have finished with EDIFF reached. Writes hybrid_soc_report.json/.md to --out
(default: a folder next to the package).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymatgen.io.vasp import Vasprun

p = argparse.ArgumentParser()
p.add_argument("package", type=Path)
p.add_argument("--out", type=Path)
args = p.parse_args()
pkg = args.package
out = args.out or pkg.parent / f"{pkg.name}_analysis"
meta = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
problems = []


def level(frame, name):
    folder = pkg / "outputs" / f"frame_{frame['frame']:04d}" / name
    text = (folder / "OUTCAR").read_text(errors="replace") if (folder / "OUTCAR").exists() else ""
    if "General timing and accounting" not in text:
        problems.append(f"frame {frame['frame']} {frame['name']} {name}: not finished")
        return None
    if "aborting loop because EDIFF is reached" not in text:
        problems.append(f"frame {frame['frame']} {frame['name']} {name}: EDIFF not reached")
        return None
    vr = Vasprun(str(folder / "vasprun.xml"), parse_dos=False, parse_potcar_file=False)
    ((spin, eig),) = vr.eigenvalues.items()
    E = eig[:, :, 0]
    nelect = int(round(vr.parameters["NELECT"]))
    nocc = (
        nelect
        if vr.parameters.get("LNONCOLLINEAR") or vr.parameters.get("LSORBIT")
        else nelect // 2
    )
    weights = np.array(vr.actual_kpoints_weights)
    elapsed = re.findall(r"Elapsed time \(sec\):\s+([\d.]+)", text)
    return {
        "E": E,
        "k": np.array(vr.actual_kpoints),
        "w": weights,
        "nocc": nocc,
        "energy": float(vr.final_energy),
        "hours": float(elapsed[-1]) / 3600 if elapsed else None,
    }


def gap(run, mesh_only=False):
    sel = run["w"] > 0 if mesh_only else np.ones(len(run["w"]), bool)
    cb, vb = run["E"][sel, run["nocc"]], run["E"][sel, run["nocc"] - 1]
    return (
        float(cb.min() - vb.max()),
        int(np.flatnonzero(sel)[cb.argmin()]),
        int(np.flatnonzero(sel)[vb.argmax()]),
    )


hybrid, soc, cost = [], [], []
for f in meta["frames"]:
    fu = f["natoms"] / 10
    if f["kind"] == "hybrid":
        a, b, c = (level(f, n) for n in ("a_edges", "b_pbeu_mesh", "c_hse06"))
        if not (a and b and c):
            continue
        labels = json.loads(
            (pkg / "inputs" / f"frame_{f['frame']:04d}" / "KPOINTS.a_edges.labels.json").read_text(
                encoding="utf-8"
            )
        )
        g_a, icb, ivb = gap(a)
        g_b, g_c = gap(b)[0], gap(c)[0]
        hybrid.append(
            {
                "name": f["name"],
                "natoms": f["natoms"],
                "pbeu_gap_converged_eV": round(g_a, 3),
                "cbm_at": labels[icb],
                "vbm_at": labels[ivb],
                "pbeu_gap_mesh_eV": round(g_b, 3),
                "hse06_gap_mesh_eV": round(g_c, 3),
                "hse06_gap_estimate_eV": round(g_a + g_c - g_b, 3),
                "hse06_hours": c["hours"],
            }
        )
    elif f["kind"] == "soc":
        d, e = level(f, "d_pbeu"), level(f, "e_pbeu_soc")
        if not (d and e):
            continue
        soc.append(
            {
                "name": f["name"],
                "formula": f["formula"],
                "fu": fu,
                "E_pbeu_per_fu": d["energy"] / fu,
                "E_soc_per_fu": e["energy"] / fu,
                "dE_soc_meV_per_fu": round(1000 * (e["energy"] - d["energy"]) / fu, 1),
                "gap_mesh_eV": round(gap(d, True)[0], 3),
                "gap_mesh_soc_eV": round(gap(e, True)[0], 3),
                "soc_hours": e["hours"],
            }
        )
    else:
        e, fh = level(f, "e_pbeu_soc"), level(f, "f_hse06_soc")
        if e and fh:
            cost.append(
                {
                    "name": f["name"],
                    "pbeu_soc_gap_mesh_eV": round(gap(e, True)[0], 3),
                    "hse06_soc_gap_mesh_eV": round(gap(fh, True)[0], 3),
                    "pbeu_soc_hours": e["hours"],
                    "hse06_soc_hours": fh["hours"],
                }
            )

# order within each composition, with and without SOC (per f.u., relative to the lowest)
ranking = {}
for r in soc:  # element counts per formula unit (10 atoms)
    r["composition"] = tuple(
        sorted(
            (el, round(float(n or 1) / r["fu"], 3))
            for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", r["formula"])
        )
    )
by_comp = defaultdict(list)
for r in soc:
    by_comp[r["composition"]].append(r)
for comp, rows in by_comp.items():
    if len(rows) < 2:
        continue
    lo0 = min(r["E_pbeu_per_fu"] for r in rows)
    lo1 = min(r["E_soc_per_fu"] for r in rows)
    for r in rows:
        r["rel_pbeu_meV"] = round(1000 * (r["E_pbeu_per_fu"] - lo0), 1)
        r["rel_soc_meV"] = round(1000 * (r["E_soc_per_fu"] - lo1), 1)
    order0 = [r["name"] for r in sorted(rows, key=lambda r: r["E_pbeu_per_fu"])]
    order1 = [r["name"] for r in sorted(rows, key=lambda r: r["E_soc_per_fu"])]
    ranking["".join(f"{el}{n:g}" for el, n in comp)] = {
        "pbeu": order0,
        "pbeu_soc": order1,
        "changed": order0 != order1,
    }

out.mkdir(parents=True, exist_ok=True)
for r in soc:
    r.pop("composition", None)
report = {
    "package": str(pkg),
    "hybrid": hybrid,
    "soc": soc,
    "soc_ranking": ranking,
    "hse06_soc_cost": cost,
    "problems": problems,
}
(out / "hybrid_soc_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
md = [
    "## HSE06 gaps (eV)",
    "",
    "| Structure | PBE+U converged | CBM / VBM at | PBE+U mesh | "
    "HSE06 mesh | HSE06 estimate | HSE06 h |",
    "|---|---|---|---|---|---|---|",
]
md += [
    f"| {r['name']} | {r['pbeu_gap_converged_eV']} | {r['cbm_at']} / {r['vbm_at']} | "
    f"{r['pbeu_gap_mesh_eV']} | {r['hse06_gap_mesh_eV']} | **{r['hse06_gap_estimate_eV']}** | "
    f"{r['hse06_hours'] or 0:.1f} |"
    for r in hybrid
]
md += [
    "",
    "## SOC (PBE+U, meV/f.u. within each composition)",
    "",
    "| Structure | ΔE_SOC | rel. PBE+U | rel. PBE+U+SOC | gap mesh → SOC (eV) |",
    "|---|---|---|---|---|",
]
md += [
    f"| {r['name']} | {r['dE_soc_meV_per_fu']} | {r.get('rel_pbeu_meV', '—')} | "
    f"{r.get('rel_soc_meV', '—')} | {r['gap_mesh_eV']} → {r['gap_mesh_soc_eV']} |"
    for r in soc
]
md += [
    "",
    "Order changed by SOC: "
    + (", ".join(k for k, v in ranking.items() if v["changed"]) or "nowhere"),
    "",
    "## HSE06+SOC cost",
    "",
]
md += [
    f"- {r['name']}: gap on the mesh {r['hse06_soc_gap_mesh_eV']} eV (PBE+U+SOC "
    f"{r['pbeu_soc_gap_mesh_eV']}); {r['hse06_soc_hours'] or 0:.1f} h for HSE06+SOC"
    for r in cost
]
md += ["", "Problems: " + ("; ".join(problems) or "none")]
(out / "hybrid_soc_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print(f"-> {out}")
sys.exit(1 if problems else 0)
