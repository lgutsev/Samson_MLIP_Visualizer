"""Analyse package 29 (CBM dispersion at relaxed geometries, PBE+U).

    PYTHONPATH=../../src micromamba run -n defects python cbm_analyze.py PACKAGE [--out DIR]

Per frame: every level finished (timing summary, EDIFF reached); the explicit k-points match the
package's labels; the gap and where the CBM and VBM are (over the uniform grid, the Γ-TRIM lines
and the stencils), and whether they sit at a TRIM; the electron and hole mass tensors at every
TRIM (least-squares quadratic fit to the stencil, Cartesian, from the cell VASP used), with the
larger-step axis curvature as a parabolicity check and the gap to the next band as a degeneracy
flag; the element and s/p/d character of the edge states (``7_char``). Then relaxed against cubic
reference per composition. Writes cbm_report.json and cbm_report.md to --out (default: a
``cbm_analysis`` folder next to the package, not inside it).
"""

import argparse
import json
import sys
from pathlib import Path

import band_kpoints as bk
import numpy as np
from pymatgen.io.vasp import Vasprun

p = argparse.ArgumentParser()
p.add_argument("package", type=Path)
p.add_argument("--out", type=Path)
args = p.parse_args()
pkg = args.package
out = args.out or pkg.parent / f"{pkg.name}_analysis"
meta = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
LEVELS = ("5_edges", "6_mass", "7_char")
DEGENERATE_MEV = 5.0
OFF_TRIM_MEV = 2.0


def finished(folder):
    text = (folder / "OUTCAR").read_text(errors="replace") if (folder / "OUTCAR").exists() else ""
    if "General timing and accounting" not in text:
        return "OUTCAR has no timing summary"
    if "aborting loop because EDIFF is reached" not in text:
        return "EDIFF not reached"
    return None


def load(frame, level):
    folder = pkg / "outputs" / f"frame_{frame:04d}" / level
    problem = finished(folder)
    if problem:
        return None, problem
    vr = Vasprun(
        str(folder / "vasprun.xml"),
        parse_dos=False,
        parse_potcar_file=False,
        parse_projected_eigen=(level == "7_char"),
    )
    labels = json.loads(
        (pkg / "inputs" / f"frame_{frame:04d}" / f"KPOINTS.{level}.labels.json").read_text(
            encoding="utf-8"
        )
    )
    k = np.array(vr.actual_kpoints)
    ref = np.array(
        [
            [float(x) for x in line.split()[:3]]
            for line in (pkg / "inputs" / f"frame_{frame:04d}" / f"KPOINTS.{level}")
            .read_text()
            .splitlines()[3:]
            if line.strip()
        ]
    )
    if len(k) != len(labels) or not np.allclose(k, ref, atol=1e-6):
        return None, "k-points in vasprun.xml differ from the package's list"
    ((spin, eig),) = vr.eigenvalues.items()  # ISPIN = 1
    return {
        "k": k,
        "labels": labels,
        "E": eig[:, :, 0],
        "cell": vr.final_structure.lattice.matrix,
        "nelect": vr.parameters["NELECT"],
        "vr": vr,
    }, None


def fit_hessian(run, label, band):
    """Least-squares E = E0 + g·dk + ½ dk·H·dk over the centre and the first-step points."""
    idx = {lab: i for i, lab in enumerate(run["labels"])}
    c = idx[f"{label} c"]
    pts = [i for lab, i in idx.items() if lab.startswith(f"{label} ") and lab.endswith("h0")]
    dk = bk.to_cartesian(run["cell"], run["k"][pts] - run["k"][c])
    e = run["E"][pts, band] - run["E"][c, band]
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    A = np.column_stack([dk, *[(0.5 if i == j else 1.0) * dk[:, i] * dk[:, j] for i, j in pairs]])
    sol, *_ = np.linalg.lstsq(A, e, rcond=None)
    H = np.zeros((3, 3))
    for (i, j), v in zip(pairs, sol[3:], strict=True):
        H[i, j] = H[j, i] = v
    resid = float(np.sqrt(np.mean((A @ sol - e) ** 2)) * 1000)
    # parabolicity: axis curvature with the second step against the fit's
    ratios = []
    for axis in "xyz":
        a, b = idx[f"{label} +{axis} h1"], idx[f"{label} -{axis} h1"]
        d = bk.to_cartesian(run["cell"], run["k"][a] - run["k"][c])
        h2 = float(d @ d)
        curv = (run["E"][a, band] + run["E"][b, band] - 2 * run["E"][c, band]) / h2
        u = d / np.sqrt(h2)
        ratios.append(round(float(curv / (u @ H @ u)), 3) if abs(u @ H @ u) > 1e-6 else None)
    return H, resid, ratios, float(np.linalg.norm(sol[:3]))


report, problems = [], []
for f in meta["frames"]:
    runs, row = {}, {k: f[k] for k in ("frame", "name", "kind", "natoms", "formula")}
    for level in LEVELS:
        run, problem = load(f["frame"], level)
        if problem:
            problems.append(f"frame {f['frame']} {f['name']} {f['kind']} {level}: {problem}")
        runs[level] = run
    if not all(runs.values()):
        row["status"] = "incomplete"
        report.append(row)
        continue
    nocc = int(round(runs["5_edges"]["nelect"])) // 2
    cb, vb = nocc, nocc - 1
    # edges over every point of every level (same SCF mesh and settings in each)
    mesh_spread = max(abs(runs[lv]["E"][0, vb] - runs["5_edges"]["E"][0, vb]) for lv in LEVELS)
    allE = [
        (runs[lv]["E"][i], runs[lv]["labels"][i], lv)
        for lv in LEVELS
        for i in range(len(runs[lv]["labels"]))
    ]
    cbm = min(allE, key=lambda t: t[0][cb])
    vbm = max(allE, key=lambda t: t[0][vb])
    trim_c = {lab: i for i, lab in enumerate(runs["6_mass"]["labels"]) if lab.endswith(" c")}
    E6 = runs["6_mass"]["E"]
    cb_trim = min(trim_c, key=lambda lab: E6[trim_c[lab], cb])
    vb_trim = max(trim_c, key=lambda lab: E6[trim_c[lab], vb])
    row.update(
        status="ok",
        nocc=nocc,
        mesh_consistency_meV=round(1000 * mesh_spread, 2),
        gap_eV=round(float(cbm[0][cb] - vbm[0][vb]), 4),
        cbm={"at": cbm[1], "level": cbm[2], "E": round(float(cbm[0][cb]), 4)},
        vbm={"at": vbm[1], "level": vbm[2], "E": round(float(vbm[0][vb]), 4)},
        cbm_trim=cb_trim.split()[0],
        vbm_trim=vb_trim.split()[0],
        cbm_off_trim_meV=round(1000 * float(E6[trim_c[cb_trim], cb] - cbm[0][cb]), 2),
        vbm_off_trim_meV=round(1000 * float(vbm[0][vb] - E6[trim_c[vb_trim], vb]), 2),
        direct_gap_at_cbm_eV=round(float(cbm[0][cb] - cbm[0][vb]), 4),
    )
    row["flags"] = []
    if row["cbm_off_trim_meV"] > OFF_TRIM_MEV:
        row["flags"].append(
            f"CBM {row['cbm_off_trim_meV']} meV below the best TRIM ({cbm[1]}): "
            "the TRIM masses are not the band-edge masses"
        )
    if row["vbm_off_trim_meV"] > OFF_TRIM_MEV:
        row["flags"].append(f"VBM {row['vbm_off_trim_meV']} meV above the best TRIM ({vbm[1]})")
    masses = {}
    for lab, i in trim_c.items():
        name = lab.split()[0]
        entry = {
            "E_cb": round(float(E6[i, cb]), 4),
            "E_vb": round(float(E6[i, vb]), 4),
            "cb_split_meV": round(1000 * float(E6[i, cb + 1] - E6[i, cb]), 2),
            "vb_split_meV": round(1000 * float(E6[i, vb] - E6[i, vb - 1]), 2),
        }
        for carrier, band, sign in (("electron", cb, 1), ("hole", vb, -1)):
            H, resid, ratios, grad = fit_hessian(runs["6_mass"], name, band)
            entry[carrier] = {
                **bk.masses(H, sign),
                "fit_rms_meV": round(resid, 3),
                "h1_over_h0_curvature": ratios,
                "slope_eV_A": round(grad, 4),
            }
        masses[name] = entry
    row["masses"] = masses
    e_edge, h_edge = masses[row["cbm_trim"]], masses[row["vbm_trim"]]
    row["electron"] = e_edge["electron"]
    row["hole"] = h_edge["hole"]
    if e_edge["cb_split_meV"] < DEGENERATE_MEV:
        row["flags"].append(
            f"CB degenerate at {row['cbm_trim']} ({e_edge['cb_split_meV']} meV to the "
            "next band): masses are of the sorted lowest band; read the per-direction values"
        )
    if h_edge["vb_split_meV"] < DEGENERATE_MEV:
        row["flags"].append(f"VB degenerate at {row['vbm_trim']} ({h_edge['vb_split_meV']} meV)")
    # edge character (LORBIT 10: s, p, d per ion) at the edge TRIMs
    vr = runs["7_char"]["vr"]
    ((spin, proj),) = vr.projected_eigenvalues.items()
    symbols = [s.specie.symbol for s in vr.final_structure]
    lab7 = runs["7_char"]["labels"]
    char = {}
    for which, trim, band in (("cbm", row["cbm_trim"], cb), ("vbm", row["vbm_trim"], vb)):
        w = proj[lab7.index(trim), band]  # ions x orbitals
        tot = w.sum() or 1.0
        parts = {}
        for ion, s in enumerate(symbols):
            for o, orb in enumerate("spd"[: w.shape[1]]):
                parts[f"{s} {orb}"] = parts.get(f"{s} {orb}", 0.0) + float(w[ion, o]) / tot
        char[which] = {
            k: round(v, 3) for k, v in sorted(parts.items(), key=lambda t: -t[1]) if v >= 0.03
        }
    row["character"] = char
    report.append(row)

# relaxed vs cubic reference, per composition
pairs = []
for name in sorted({r["name"] for r in report}):
    rel = next((r for r in report if r["name"] == name and r["kind"] == "relaxed"), None)
    cub = next((r for r in report if r["name"] == name and r["kind"] == "cubic_ref"), None)
    if rel and cub and rel.get("status") == cub.get("status") == "ok":
        pairs.append(
            {
                "name": name,
                "gap_relaxed": rel["gap_eV"],
                "gap_cubic": cub["gap_eV"],
                "me_cond_relaxed": rel["electron"]["conductivity_me"],
                "me_cond_cubic": cub["electron"]["conductivity_me"],
            }
        )

out.mkdir(parents=True, exist_ok=True)
(out / "cbm_report.json").write_text(
    json.dumps(
        {"package": str(pkg), "frames": report, "relaxed_vs_cubic": pairs, "problems": problems},
        indent=1,
    ),
    encoding="utf-8",
)


def fmt(m):
    return "—" if m is None else f"{m:.2f}"


lines = [
    "| Frame | Structure | Geometry | Gap (eV) | CBM at | m*e cond / DOS (mₑ) | m*e principal | "
    "m*h cond (mₑ) | CBM character | Flags |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for r in report:
    if r.get("status") != "ok":
        lines.append(f"| {r['frame']} | {r['name']} | {r['kind']} | incomplete | | | | | | |")
        continue
    e, h = r["electron"], r["hole"]
    ch = ", ".join(f"{k} {v:.2f}" for k, v in list(r["character"]["cbm"].items())[:3])
    pr = " / ".join(f"{m:.2f}" for m in e["principal_me"]) if e["principal_me"] else "not a minimum"
    lines.append(
        f"| {r['frame']} | {r['name']} | {r['kind']} | {r['gap_eV']:.3f} | {r['cbm_trim']} "
        f"({r['cbm_off_trim_meV']} meV) | "
        f"{fmt(e['conductivity_me'])} / {fmt(e['dos_me'])} | {pr} | "
        f"{fmt(h['conductivity_me'])} | {ch} | {'; '.join(r['flags']) or '—'} |"
    )
md = "\n".join(lines) + "\n\nProblems: " + ("; ".join(problems) or "none") + "\n"
(out / "cbm_report.md").write_text(md, encoding="utf-8")
print(md)
print(f"-> {out}")
sys.exit(1 if problems else 0)
