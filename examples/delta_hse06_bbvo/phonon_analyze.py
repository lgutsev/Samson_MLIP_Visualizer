"""Analyse package 18 (PBE+U phonons of cubic Ba₂BiVO₆ + the cubic -> minimum path scan).

    PYTHONPATH=../../src micromamba run -n defects python phonon_analyze.py PACKAGE [--out DIR]
    ... phonon_analyze.py PACKAGE --mace          # same pipeline on MACE-MP-0 forces (a preview, NOT DFT)
    ... phonon_analyze.py PACKAGE --scan SCANDIR  # also write a frozen-mode scan package for the soft modes

Checks every frame before using it, and stops on the first problem rather than guessing:
- ``samson_mlip_visualizer.vasp_labeling.parse_vasp_run``: OUTCAR finished, EDIFF reached
  (not NELM), vasprun.xml readable; energy = force-consistent (e_fr_energy);
- the geometry in vasprun.xml equals the displaced supercell phonopy expects (0.001 Å, minimum image);
- the net force (drift) per frame, reported and removed by phonopy's symmetrization.

Reports, per set: frequencies at the commensurate points (Γ, X; L for sc80), the unstable
modes (f < −0.1 THz; smaller |f| is treated as numerical noise and reported separately), each
mode's weight on Ba/Bi/V/O and Γ irreps, the dispersion (interpolated between commensurate
points, so only the commensurate points are exact), the pressure series, and E(λ) on the path.

What the results can and cannot say:
- imaginary modes at the cubic point = negative curvature (local instability) at this level;
- no imaginary modes does NOT make the lower-energy 08 structure an artefact: a locally stable
  cubic phase can coexist with a deeper minimum behind a barrier. The path scan (and a
  frozen-mode scan) tell the two cases apart: E(λ) falling from λ = 0 vs rising first.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import phonopy
from ase import Atoms
from ase.io import read

p = argparse.ArgumentParser()
p.add_argument("package", type=Path)
p.add_argument("--out", type=Path)
p.add_argument("--mace", action="store_true", help="forces from MACE-MP-0 instead of the VASP outputs (preview)")
p.add_argument("--scan", type=Path, help="write a frozen-mode scan package here (dispatcher layout)")
p.add_argument("--noise", type=float, default=0.1, help="THz; |f| below this at imaginary is reported as noise")
args = p.parse_args()
pkg = args.package
meta = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
out = args.out or (pkg / ("analysis_mace_preview" if args.mace else "analysis"))
out.mkdir(parents=True, exist_ok=True)
calc = None
if args.mace:
    sys.path.insert(0, str(Path(__file__).parent))
    from common import mace_mp0
    calc = mace_mp0()


def frame_result(i):
    """(atoms with forces, energy per atom, notes) of frame i, or SystemExit with the reason."""
    folder = pkg / "outputs" / f"frame_{i:04d}" / "pbe_u"
    inp = read(pkg / "inputs" / f"frame_{i:04d}" / "POSCAR", format="vasp")
    if calc is not None:
        inp.calc = calc
        return inp.get_forces(), inp.get_potential_energy(), inp
    from ase.geometry import find_mic
    from samson_mlip_visualizer.vasp_labeling import OutputError, parse_vasp_run
    try:
        run = parse_vasp_run(folder)  # OUTCAR finished, EDIFF reached, vasprun readable
    except OutputError as ex:
        sys.exit(f"frame {i}: {ex} ({folder})")
    if run["symbols"] != inp.get_chemical_symbols() or np.abs(run["cell"] - inp.cell[:]).max() > 1e-4:
        sys.exit(f"frame {i}: atoms or cell differ from the input")
    shift, _ = find_mic(run["positions"] - inp.positions, inp.cell, True)
    if np.linalg.norm(shift, axis=1).max() > 1e-3:
        sys.exit(f"frame {i}: geometry differs from its input")
    return run["forces"], run["energy"], inp


report = {"package": str(pkg), "forces_from": "MACE-MP-0 (preview, not DFT)" if args.mace else "VASP PBE+U", "sets": {}}
QPTS = {"G": [0, 0, 0], "X": [0.5, 0.0, 0.5], "L": [0.5, 0.5, 0.5], "W": [0.5, 0.25, 0.75]}
COMM = {"sc40": ("G", "X"), "sc80": ("G", "X", "L")}
PATH = [[[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.25, 0.75], [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
LABELS = ["$\\Gamma$", "X", "W", "K", "$\\Gamma$", "L"]
soft_modes = []
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
for name, s in meta["sets"].items():
    if name == "path08":
        continue
    ph = phonopy.load(pkg / s["phonopy_yaml"], produce_fc=False, log_level=0)
    f0, f1 = s["frames"]
    forces, drift = [], []
    for i in range(f0, f1 + 1):
        f, _, _ = frame_result(i)
        drift.append(float(np.abs(f.sum(axis=0)).max()))
        forces.append(f)
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    entry = {"atoms": s["atoms"], "lattice_scale": s["lattice_scale"], "max_drift_eV_A": max(drift), "points": {}}
    masses = ph.primitive.masses
    symbols = ph.primitive.symbols
    for lab in COMM[s["supercell"]]:
        q = QPTS[lab]
        freqs, vecs = ph.get_frequencies_with_eigenvectors(q)
        modes = []
        for b, fr in enumerate(freqs):
            e = vecs[:, b].reshape(-1, 3)
            w = {}
            for sym, v in zip(symbols, e):
                w[sym] = w.get(sym, 0.0) + float(np.vdot(v, v).real)
            modes.append({"band": b, "THz": round(float(fr), 3), "weights": {k: round(v, 3) for k, v in w.items()}})
        unstable = [m for m in modes if m["THz"] < -args.noise]
        noise = [m for m in modes if -args.noise <= m["THz"] < 0]
        irreps = None
        if lab == "G":
            try:
                ph.set_irreps(q, degeneracy_tolerance=1e-3)
                ir = ph.irreps  # labels per degenerate set (phonopy keeps them private)
                irreps = {}
                for label, deg in zip(getattr(ir, "_ir_labels", None) or [], getattr(ir, "_degenerate_sets", [])):
                    for b in deg:
                        irreps[int(b)] = label
                for m in modes:
                    m["irrep"] = irreps.get(m["band"])
            except Exception as ex:
                irreps = f"irreps failed: {ex}"
        entry["points"][lab] = {"lowest": modes[:6], "n_unstable": len(unstable), "unstable": unstable,
                                "n_noise": len(noise)}
        for m in unstable:
            soft_modes.append({"set": name, "q": lab, **m})
    ph.run_band_structure(PATH, labels=LABELS)
    bs = ph.get_band_structure_dict()
    if name in ("sc40", "sc80"):
        for dist, fr in zip(bs["distances"], bs["frequencies"]):
            ax.plot(dist, fr, color="#2c3e50" if name == "sc40" else "#c0392b", lw=0.7,
                    label=name if dist is bs["distances"][0] else None)
    entry["min_freq_on_path_THz"] = float(min(np.min(f) for f in bs["frequencies"]))
    report["sets"][name] = entry
    print(f"{name}: drift {max(drift):.1e} eV/A; " + "; ".join(
        f"{lab}: {v['n_unstable']} unstable, lowest {v['lowest'][0]['THz']} THz" for lab, v in entry["points"].items()))
ax.axhline(0, color="0.5", lw=0.5)
ax.set_ylabel("frequency (THz)")
ax.set_title(("MACE-MP-0 PREVIEW (not DFT)" if args.mace else "PBE+U") + ": cubic Ba$_2$BiVO$_6$; exact only at commensurate q", fontsize=8)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(out / "dispersion.png", dpi=150)

if "path08" in meta["sets"]:
    s = meta["sets"]["path08"]
    E = []
    for i, lam in zip(range(s["frames"][0], s["frames"][1] + 1), s["lambdas"]):
        _, e, _ = frame_result(i)
        E.append(e)
    E = (np.array(E) - E[0]) / 4 * 1000  # meV per formula unit (40 atoms = 4 f.u.)
    lam = np.array(s["lambdas"])
    c2 = np.polyfit(lam[:4], E[:4], 2)[0]  # curvature near the cubic point from λ <= 0.2
    shape = ("falls from the cubic point (no barrier along this path)" if E[1] < 0 and E[2] < E[1]
             else "rises first: a barrier along this path (cubic locally stable along it)" if E[1] > 0
             else "flat near the cubic point")
    report["path08"] = {"lambda": s["lambdas"], "E_meV_per_fu": [round(float(x), 2) for x in E],
                        "curvature_near_0_meV_per_fu": round(float(c2), 1), "shape": shape}
    print(f"path08: E(λ) = {', '.join(f'{x:.1f}' for x in E)} meV/f.u.; {shape}")
(out / "phonon_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

if args.scan:
    if not soft_modes:
        print("no unstable modes: no frozen-mode scan written (use the path scan)")
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from common import KSPACING
        from loni_chain import write_frame, write_manifest, write_script
        if args.scan.exists():
            sys.exit(f"{args.scan} exists")
        frames = []
        base = {"sc40": meta["sets"]["sc40"], "sc80": meta["sets"]["sc80"]}
        args.scan.mkdir(parents=True)
        # one mode per degenerate set, from the reference cells only (sc40 for Γ/X, sc80 for L)
        chosen, seen = [], set()
        for m in sorted(soft_modes, key=lambda m: m["THz"]):
            if m["set"] != ("sc80" if m["q"] == "L" else "sc40"):
                continue
            key = (m["q"], round(m["THz"], 2))
            if key not in seen:
                seen.add(key); chosen.append(m)
        for m in chosen[:4]:
            s = base[m["set"]]
            ph = phonopy.load(pkg / s["phonopy_yaml"], produce_fc=False, log_level=0)
            ph.forces = np.array([frame_result(i)[0] for i in range(s["frames"][0], s["frames"][1] + 1)])
            ph.produce_force_constants(); ph.symmetrize_force_constants()
            for amp in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):  # phonopy amplitude (Å·√amu, normalised by N)
                ph.run_modulations(ph.supercell_matrix, [[QPTS[m["q"]], m["band"], amp, 0]])
                cell = ph.get_modulated_supercells()[0]
                img = Atoms(cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)
                write_frame(args.scan, len(frames), img, {"pbe_u": {"EDIFF": "1E-7", "NCORE": 4, "KPAR": 4, "LWAVE": ".FALSE.", "LCHARG": ".FALSE."}},
                            KSPACING, None, f"mode {m['q']} band {m['band']} amp {amp}")
                frames.append({"frame": len(frames), "q": m["q"], "band": m["band"], "THz": m["THz"], "amplitude": amp})
        write_script(args.scan, len(frames), {"pbe_u": "poscar"}, name="bbvo-modescan", time="02:00:00", throttle=20)
        write_manifest(args.scan, frames, {"package": "frozen-mode scan of package 18", "forces_from": report["forces_from"]})
        print(f"frozen-mode scan: {len(frames)} single points -> {args.scan}")
print(f"-> {out}")
