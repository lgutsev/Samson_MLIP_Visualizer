"""Analyse package 38 (PBE+U LOPTICS of cubic and R3 Ba₂BiVO₆).

    PYTHONPATH=../../src micromamba run -n defects python optics_analyze.py PACKAGE [--out DIR]

Per frame: ``1_optics`` finished with EDIFF reached (``parse_vasp_run``) and a density-density
dielectric function in vasprun.xml. From the eigenvalues, the PBE+U gap on the mesh (fundamental
and smallest direct). From ε(E) (trace/3, the isotropic average a polycrystalline film sees; the
diagonal components are written too): n + iκ = √ε, α = 4πκ/λ, on 0-5 eV.

Reported at gap + 0.1, 0.2 and 0.5 eV: α and the single-pass absorptance of a 500 nm film,
1 − exp(−αd). Twice: as computed (PBE+U gap) and with ε rigidly shifted up by the frame's HSE06 −
PBE+U gap difference (package 30), which is **a scissor estimate**, not a hybrid optics result.
For a phase computed on two meshes, the relative change of α at those energies is the
k-convergence check.
"""

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from samson_mlip_visualizer.vasp_labeling import OutputError, parse_vasp_run

HC_EV_CM = 1.239841984e-4  # h c in eV cm
FILM_CM = 500e-7
OFFSETS = (0.1, 0.2, 0.5)


def dielectric(root):
    """(E, eps complex [n, 6]) from the density-density block (VASP 6 also writes
    current-current)."""
    blocks = root.findall(".//dielectricfunction")
    if not blocks:
        raise OutputError("no <dielectricfunction> in vasprun.xml (LOPTICS not applied?)")
    blk = next((b for b in blocks if b.get("comment", "") == "density-density"), blocks[0])

    def table(tag):
        rows = blk.find(f"{tag}/array/set").findall("r")
        return np.array([[float(x) for x in r.text.split()] for r in rows])

    im, re_ = table("imag"), table("real")
    return im[:, 0], re_[:, 1:7] + 1j * im[:, 1:7]


def gaps(root):
    """Fundamental and smallest direct gap (eV) from the last calculation's eigenvalues."""
    calc = root.findall(".//calculation")[-1]
    vb_k, cb_k = [], []
    for spin in calc.find("eigenvalues/array/set").findall("set"):
        for i, k in enumerate(spin.findall("set")):
            eo = np.array([[float(x) for x in r.text.split()] for r in k.findall("r")])
            v, c = eo[eo[:, 1] >= 0.5, 0].max(), eo[eo[:, 1] < 0.5, 0].min()
            if i < len(vb_k):
                vb_k[i], cb_k[i] = max(vb_k[i], v), min(cb_k[i], c)
            else:
                vb_k.append(v)
                cb_k.append(c)
    vb_k, cb_k = np.array(vb_k), np.array(cb_k)
    return float(cb_k.min() - vb_k.max()), float((cb_k - vb_k).min())


def absorption(e, eps):
    """α (cm⁻¹) from the isotropic average ε = trace/3."""
    nk = np.sqrt(eps)
    return 4 * np.pi * np.abs(nk.imag) * e / HC_EV_CM


def shifted(e, eps, delta):
    """ε(E − Δ) on the same grid, zero below Δ: a rigid scissor shift (estimate)."""
    return (np.interp(e - delta, e, eps.real, left=eps.real[0])
            + 1j * np.interp(e - delta, e, eps.imag, left=0.0))


def at_edge(e, alpha, edge):
    return {f"+{x}": {"alpha_cm-1": float(np.interp(edge + x, e, alpha)),
                      "absorptance_500nm": float(1 - np.exp(-np.interp(edge + x, e, alpha)
                                                            * FILM_CM))}
            for x in OFFSETS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("package", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    meta = json.loads((a.package / "package.json").read_text(encoding="utf-8"))
    out = a.out or a.package / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows, problems, curves = [], [], {}
    for f in meta["frames"]:
        folder = a.package / "outputs" / f"frame_{f['frame']:04d}" / "1_optics"
        try:
            parse_vasp_run(folder)
            root = ET.parse(folder / "vasprun.xml").getroot()
            e, eps6 = dielectric(root)
        except (OutputError, OSError, ET.ParseError) as ex:
            problems.append(f"{f['name']}: {ex}")
            continue
        eg, eg_direct = gaps(root)
        eps = eps6[:, :3].mean(axis=1)
        alpha = absorption(e, eps)
        d = f["scissor_eV"]
        eps_s = shifted(e, eps, d)
        alpha_s = absorption(e, eps_s)
        curves[f["name"]] = (e, alpha)
        rows.append({"frame": f["frame"], "name": f["name"], "phase": f["phase"],
                     "kmesh": f["kmesh"], "gap_pbeu_eV": eg, "gap_direct_pbeu_eV": eg_direct,
                     "pbeu": at_edge(e, alpha, eg),
                     "scissor_eV": d, "gap_scissor_eV": eg + d,
                     "scissor_estimate": at_edge(e, alpha_s, eg + d)})
        keep = e <= 5.0
        with open(out / f"optics_{f['name']}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["E_eV", "eps1", "eps2", "alpha_cm-1", "eps2_xx", "eps2_yy", "eps2_zz",
                        "E_scissor_eV", "eps1_scissor", "eps2_scissor", "alpha_scissor_cm-1"])
            for i in np.flatnonzero(keep):
                w.writerow([f"{e[i]:.4f}", f"{eps[i].real:.5f}", f"{eps[i].imag:.5f}",
                            f"{alpha[i]:.4e}", *(f"{eps6[i, j].imag:.5f}" for j in range(3)),
                            f"{e[i]:.4f}", f"{eps_s[i].real:.5f}", f"{eps_s[i].imag:.5f}",
                            f"{alpha_s[i]:.4e}"])
    # k-convergence: the same phase on two meshes, α at the coarser run's edge offsets
    kconv = {}
    for phase in {r["phase"] for r in rows}:
        rs = sorted((r for r in rows if r["phase"] == phase), key=lambda r: r["kmesh"][0])
        if len(rs) == 2:
            (e0, a0), (e1, a1) = curves[rs[0]["name"]], curves[rs[1]["name"]]
            kconv[phase] = {}
            for x in OFFSETS:  # None where the coarse mesh gives no absorption yet
                c = np.interp(rs[0]["gap_pbeu_eV"] + x, e0, a0)
                f_ = np.interp(rs[0]["gap_pbeu_eV"] + x, e1, a1)
                kconv[phase][f"+{x}"] = float(f_ / c - 1) if c > 0 else None
            kconv[phase]["meshes"] = [rs[0]["kmesh"], rs[1]["kmesh"]]
            kconv[phase]["gap_change_eV"] = rs[1]["gap_pbeu_eV"] - rs[0]["gap_pbeu_eV"]
    report = {"rows": rows, "k_convergence_relative_alpha": kconv, "problems": problems,
              "note": "independent-particle PBE+U; the scissor columns are an estimate "
                      "(rigid shift by HSE06 - PBE+U from package 30), not hybrid optics"}
    (out / "optics_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = ["| Frame | Name | Mesh | Gap / direct (eV) | α at +0.1/+0.2/+0.5 eV (cm⁻¹) | "
             "A(500 nm) at +0.1/+0.2/+0.5 | Scissor gap (eV) | α scissor (cm⁻¹) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        al = " / ".join(f"{v['alpha_cm-1']:.2e}" for v in r["pbeu"].values())
        ab = " / ".join(f"{v['absorptance_500nm']:.2f}" for v in r["pbeu"].values())
        als = " / ".join(f"{v['alpha_cm-1']:.2e}" for v in r["scissor_estimate"].values())
        lines.append(f"| {r['frame']} | {r['name']} | {'×'.join(map(str, r['kmesh']))} | "
                     f"{r['gap_pbeu_eV']:.3f} / {r['gap_direct_pbeu_eV']:.3f} | {al} | {ab} | "
                     f"{r['gap_scissor_eV']:.3f} | {als} |")
    for phase, k in kconv.items():
        lines.append(f"\nk-convergence {phase} {k['meshes'][0]} -> {k['meshes'][1]}: α change "
                     + ", ".join(f"{x} eV " + ("n/a (no absorption on the coarse mesh)"
                                               if k[f"+{x}"] is None else f"{k[f'+{x}']:+.1%}")
                                 for x in OFFSETS)
                     + f"; gap change {k['gap_change_eV']:+.3f} eV")
    lines += ["", "Scissor columns: rigid shift by HSE06 - PBE+U (package 30); an estimate.",
              "Problems: " + ("; ".join(problems) if problems else "none")]
    (out / "optics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
