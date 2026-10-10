"""Analyse package 42 (Ba₂ScVO₆ Sc-B set: Fm-3m cell, HSE06 gap B1, orbital-resolved CBM B3).

    PYTHONPATH=../../src micromamba run -n defects python scb_analyze.py PACKAGE [--out DIR]

- Every level finished: relaxations reached EDIFFG (IBRION 2) with every SCF converged; single
  points have a timing summary and reached EDIFF; explicit k-points match the package's lists.
- Common geometry (frame 0): E(3_static) per f.u. against 37's ref_BiSc1 (``reference_package``,
  next to this package in the results clone), conventional a, space group, V coordination.
- B1 (frame 0): the converged PBE+U gap from ``a_edges`` (uniform grid, Γ-TRIM lines, TRIMs) and
  where its edges are; PBE+U and HSE06 gaps on the SCF mesh (``b_pbeu_mesh``, ``c_hse06``); the
  HSE06 estimate = converged PBE+U + (HSE06 − PBE+U on the mesh). That difference is the scissor
  optics_analyze.py applies to package 43.
- B3 (frame 1): for the lowest three conduction bands at every point of the Γ stencil, the V d
  weights (d_xy, d_yz, d_xz, d_z², d_x²−y² as fractions of V d, Cartesian axes of the cubic
  cell) and the V d share of the band; the band's Γ mass tensor (least-squares fit as in
  cbm_analyze.py), to compare with 31 frame 12.
Writes scb_report.json and scb_report.md to --out (default: ``<package>_analysis`` next to it).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import band_kpoints as bk
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from samson_mlip_visualizer.vasp_labeling import OutputError, parse_vasp_run

p = argparse.ArgumentParser()
p.add_argument("package", type=Path)
p.add_argument("--out", type=Path)
args = p.parse_args()
pkg = args.package
out = args.out or pkg.parent / f"{pkg.name}_analysis"
meta = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
problems = []
D_ORB = {"dxy": 4, "dyz": 5, "dz2": 6, "dxz": 7, "dx2": 8}  # VASP LORBIT 11 order


def outcar(folder):
    f = folder / "OUTCAR"
    return f.read_text(errors="replace") if f.exists() else ""


def relax_ok(folder):
    t = outcar(folder)
    if "General timing and accounting" not in t:
        return "OUTCAR missing or unfinished"
    if "reached required accuracy" not in t:
        return "EDIFFG not reached (NSW exhausted?)"
    ionic = len(re.findall(r"^\s*-+\s*Iteration\s+(\d+)\(\s*1\)", t, re.M))
    if t.count("aborting loop because EDIFF is reached") < ionic:
        return "an ionic step with SCF not converged"
    return None


def single_ok(folder):
    t = outcar(folder)
    if "General timing and accounting" not in t:
        return "OUTCAR missing or unfinished"
    if "aborting loop because EDIFF is reached" not in t:
        return "EDIFF not reached"
    return None


def level(frame, name, projected=False):
    """(Vasprun, labels or None) of a finished single point, else None (problem recorded)."""
    folder = pkg / "outputs" / f"frame_{frame:04d}" / name
    why = single_ok(folder)
    if why:
        problems.append(f"frame {frame} {name}: {why}")
        return None, None
    vr = Vasprun(str(folder / "vasprun.xml"), parse_dos=False, parse_potcar_file=False,
                 parse_projected_eigen=projected)
    lab_file = pkg / "inputs" / f"frame_{frame:04d}" / f"KPOINTS.{name}.labels.json"
    labels = None
    if lab_file.exists():
        labels = json.loads(lab_file.read_text(encoding="utf-8"))
        ref = np.array([[float(x) for x in line.split()[:3]] for line in
                        (lab_file.parent / f"KPOINTS.{name}").read_text().splitlines()[3:]
                        if line.strip()])
        k = np.array(vr.actual_kpoints)
        if len(k) != len(labels) or not np.allclose(k, ref, atol=1e-6):
            problems.append(f"frame {frame} {name}: k-points differ from the package's list")
            return None, None
    return vr, labels


def edges(vr):
    """(eigenvalues [k, band], index of the lowest empty band, VBM k, CBM k, gap, direct gap)."""
    ((_, eig),) = vr.eigenvalues.items()  # ISPIN = 1
    e = eig[:, :, 0]
    cb = int(round(vr.parameters["NELECT"])) // 2
    kv, kc = int(np.argmax(e[:, cb - 1])), int(np.argmin(e[:, cb]))
    return (e, cb, kv, kc, float(e[kc, cb] - e[kv, cb - 1]),
            float((e[:, cb] - e[:, cb - 1]).min()))


def fit_hessian(k, E, labels, cell, label):
    """Least-squares E = E0 + g·dk + ½ dk·H·dk over the centre's first-step points (as 29/31)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    c = idx[f"{label} c"]
    pts = [i for lab, i in idx.items() if lab.startswith(f"{label} ") and lab.endswith("h0")]
    dk = bk.to_cartesian(cell, k[pts] - k[c])
    e = E[pts] - E[c]
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    A = np.column_stack([dk, *[(0.5 if i == j else 1.0) * dk[:, i] * dk[:, j] for i, j in pairs]])
    sol, *_ = np.linalg.lstsq(A, e, rcond=None)
    H = np.zeros((3, 3))
    for (i, j), v in zip(pairs, sol[3:], strict=True):
        H[i, j] = H[j, i] = v
    return H


report = {"package": str(pkg), "problems": problems}
f0 = pkg / "outputs" / "frame_0000"
# ---- common geometry ----
for lv in ("1_relax", "2_relax"):
    why = relax_ok(f0 / lv)
    if why:
        problems.append(f"frame 0 {lv}: {why}")
common = {}
try:
    st = parse_vasp_run(f0 / "3_static")
    s = Structure.from_file(str(f0 / "3_static" / "CONTCAR"))
    common = {"E_eV_per_fu": st["energy"] / (len(s) / 10),
              "a_conv_A": float((4 * s.volume / (len(s) / 10)) ** (1 / 3)),  # 4 f.u./cube
              "volume_per_fu_A3": s.volume / (len(s) / 10),
              "spg_0.01": SpacegroupAnalyzer(s, symprec=0.01).get_space_group_symbol(),
              "spg_0.1": SpacegroupAnalyzer(s, symprec=0.1).get_space_group_symbol(),
              "V_CN": sorted({len([n for n in s.get_neighbors(x, 2.4) if n.specie.symbol == "O"])
                              for x in s if x.specie.symbol == "V"}),
              "a_conv_start_A": meta["frames"][0].get("a_conv_start_A")}
    other = pkg.parents[1] / meta["reference_package"]
    om = json.loads((other / "package.json").read_text(encoding="utf-8"))
    of = next(f for f in om["frames"] if f["name"] == meta["reference"])
    ref = parse_vasp_run(other / "outputs" / f"frame_{of['frame']:04d}" / "3_static")
    common["ref_E_eV_per_fu"] = ref["energy"] / (of["natoms"] / 10)
    common["dE_meV_per_fu_vs_ref"] = (common["E_eV_per_fu"] - common["ref_E_eV_per_fu"]) * 1000
    if common["dE_meV_per_fu_vs_ref"] > 5:
        problems.append(f"Fm-3m cell {common['dE_meV_per_fu_vs_ref']:.1f} meV/f.u. above "
                        "ref_BiSc1 (> 5: say so in the note)")
except (OutputError, OSError, StopIteration, KeyError) as ex:
    problems.append(f"common geometry: {ex}")
report["common"] = common
# ---- B1 ----
b1 = {}
va, la = level(0, "a_edges")
vb, _ = level(0, "b_pbeu_mesh")
vc, _ = level(0, "c_hse06")
if va is not None:
    e, cb, kv, kc, g, gd = edges(va)
    kk = np.array(va.actual_kpoints)
    b1.update({"pbeu_gap_converged_eV": g, "pbeu_direct_gap_eV": gd,
               "vbm_at": f"{la[kv]} {np.round(kk[kv], 4).tolist()}",
               "cbm_at": f"{la[kc]} {np.round(kk[kc], 4).tolist()}"})
if vb is not None and vc is not None:
    gb, gc = edges(vb)[4], edges(vc)[4]
    b1.update({"pbeu_gap_mesh_eV": gb, "hse06_gap_mesh_eV": gc, "scissor_eV": gc - gb})
    if "pbeu_gap_converged_eV" in b1:
        b1["hse06_gap_estimate_eV"] = b1["pbeu_gap_converged_eV"] + gc - gb
report["B1"] = b1
# ---- B3 ----
b3 = {}
v6, l6 = level(1, "6_mass", projected=True)
if v6 is not None:
    e, cb, *_ = edges(v6)
    if not v6.projected_eigenvalues:
        problems.append("frame 1 6_mass: no projections in vasprun.xml (LORBIT 11 not applied?)")
        proj = None
    else:
        ((_, proj),) = v6.projected_eigenvalues.items()  # [k, band, ion, orbital]
    symbols = [x.specie.symbol for x in v6.final_structure]
    vions = [i for i, sym in enumerate(symbols) if sym == "V"]
    cell = v6.final_structure.lattice.matrix
    k = np.array(v6.actual_kpoints)
    stencil = [i for i, lab in enumerate(l6) if lab.startswith("G ")]
    for n in range(3):
        band = cb + n
        pts = {}
        for i in stencil if proj is not None else ():
            w = proj[i, band]
            vd = {o: float(w[vions, j].sum()) for o, j in D_ORB.items()}
            tot_vd = sum(vd.values()) or 1.0
            pts[l6[i]] = {"E_eV": round(float(e[i, band]), 5),
                          "V_d_share": round(tot_vd / (float(w.sum()) or 1.0), 3),
                          **{o: round(v / tot_vd, 3) for o, v in vd.items()}}
        H = fit_hessian(k[stencil], e[stencil, band], [l6[i] for i in stencil], cell, "G")
        b3[f"CB{n + 1}"] = {"points": pts, "mass_G": bk.masses(H)}
report["B3"] = b3

out.mkdir(parents=True, exist_ok=True)
(out / "scb_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
L = ["## Common geometry (frame 0)"]
if common:
    L.append(f"Fm-3m 10-atom cell: a_conv {common['a_conv_A']:.4f} A (start "
             f"{common['a_conv_start_A']}), {common['spg_0.01']} / {common['spg_0.1']}, V CN "
             f"{common['V_CN']}, {common['volume_per_fu_A3']:.1f} A^3/f.u.; E - E(ref_BiSc1) = "
             f"{common.get('dE_meV_per_fu_vs_ref', float('nan')):+.1f} meV/f.u.")
L.append("\n## B1: gaps (frame 0, no SOC)")
for key, val in b1.items():
    L.append(f"- {key}: {val:.3f}" if isinstance(val, float) else f"- {key}: {val}")
L.append("\n## B3: V d weights of the lowest three conduction bands on the Γ stencil (frame 1)")
L.append("| Band | Point | E (eV) | V d share | d_xy | d_yz | d_xz | d_z2 | d_x2-y2 |")
L.append("|---|---|---|---|---|---|---|---|---|")
for bname, b in b3.items():
    for lab, v in b["points"].items():
        if lab.endswith(" c") or lab.endswith("h0") and len(lab.split()[1]) == 2:
            L.append(f"| {bname} | {lab} | {v['E_eV']:.4f} | {v['V_d_share']:.2f} | {v['dxy']:.2f} "
                     f"| {v['dyz']:.2f} | {v['dxz']:.2f} | {v['dz2']:.2f} | {v['dx2']:.2f} |")
for bname, b in b3.items():
    m = b["mass_G"]
    L.append(f"- {bname} at Γ: principal {m['principal_me']} m_e, conductivity "
             f"{m['conductivity_me']} (sorted band; compare 31 frame 12)")
L += ["", "Problems: " + ("; ".join(problems) if problems else "none")]
(out / "scb_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
if problems:
    sys.exit(1)
