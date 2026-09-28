"""Read smoke test 12 (UMA on LONI) back on the desktop.

    python check_loni_results.py [PACKAGE]

Run with SAMSON's Python after copying ``outputs/`` back into the package
(default ``D:\\MLIP_Work_Folder\\hpc_smoke_tests\\12_uma_sn2_large``). Prints:

1. PASS/FAIL for the control: the cluster's uma-s-1p1 against the same script on
   the laptop (``<work>/uma/loni_laptop_test/uma-s-1p1``). Energies on the 46
   frames must agree within 5 meV, and the barrier within 0.1 kcal/mol.
2. Every checkpoint in ``outputs/`` against ωB97X-D/def2-TZVPD on the labeled
   frames (from the desktop's cache; nothing is computed): the barrier, the largest
   energy error and RMSE, and the force RMSE. Also the stationary points and,
   with the references file, energies relative to F⁻ + CH₃Cl.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sn2_f_ch3cl"))

import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from sn2_common import KCAL, WORK, C, F, reference_cache  # noqa: E402

from samson_mlip_visualizer.reference_cache import geometry_key  # noqa: E402

PACKAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\MLIP_Work_Folder\hpc_smoke_tests\12_uma_sn2_large")
LAPTOP = WORK / "uma" / "loni_laptop_test" / "uma-s-1p1"


def frames_of(folder):
    return read(folder / "labeled_frames_uma.extxyz", ":")


def control():
    cluster = PACKAGE / "outputs" / "uma-s-1p1"
    if not (cluster / "results.json").exists():
        return "NOT RUN", "outputs/uma-s-1p1 is missing"
    a, b = frames_of(cluster), frames_of(LAPTOP)
    energy = max(abs(x.info["uma_energy"] - y.info["uma_energy"]) for x, y in zip(a, b,
                                                                                    strict=True))
    barriers = [json.loads((f / "results.json").read_text())["barrier_kcal"]
                for f in (cluster, LAPTOP)]
    ok = energy < 5e-3 and abs(barriers[0] - barriers[1]) < 0.1
    detail = (f"max |ΔE| on 46 frames {energy * 1000:.2f} meV; barrier {barriers[0]:.3f} "
              f"(cluster) vs {barriers[1]:.3f} kcal/mol (laptop)")
    return "PASS" if ok else "FAIL", detail


def against_wb97xd(folder, reference):
    rows, cf = {}, {}
    for image in frames_of(folder):
        label = reference.cache.get(geometry_key(image.numbers, image.positions,
                                                 reference.charge))
        if label is None:
            raise SystemExit(f"frame {image.info['path']}:{image.info['frame']} has no "
                             "ωB97X-D label in the cache (it should: nothing new is computed)")
        rows.setdefault(image.info["path"], []).append(
            (image.info["uma_energy"], label["energy_ev"], image.arrays["uma_forces"],
             np.asarray(label["forces"])))
        cf.setdefault(image.info["path"], []).append(image.get_distance(C, F))
    out = {}
    for path, values in rows.items():
        model = np.array([v[0] for v in values])
        ref = np.array([v[1] for v in values])
        start = 0 if cf[path][0] > cf[path][-1] else -1  # the F⁻···CH₃Cl end
        error = (model - model[start]) - (ref - ref[start])
        forces = np.concatenate([(v[2] - v[3]).ravel() for v in values])
        out[path] = {
            "barrier_model_kcal": float((model.max() - model[start]) * KCAL),
            "barrier_reference_kcal": float((ref.max() - ref[start]) * KCAL),
            "max_abs_energy_error_meV": float(np.abs(error).max() * 1000),
            "energy_rmse_meV": float(np.sqrt((error**2).mean()) * 1000),
            "force_rmse_ev_per_A": float(np.sqrt((forces**2).mean())),
        }
    return out


def main():
    status, detail = control()
    print(f"control (uma-s-1p1 on the cluster vs the laptop): {status}  {detail}\n")
    outputs = sorted(p for p in (PACKAGE / "outputs").glob("*") if (p / "results.json").exists())
    if not outputs:
        print("no outputs yet: copy outputs/ back from the cluster")
        return
    reference = reference_cache()
    for folder in outputs:
        results = json.loads((folder / "results.json").read_text())
        ts = results["stationary"]["ts"]
        rc = results["stationary"]["reactant_complex"]
        print(f"== {folder.name} ({results['device']}, load {results['load_s']:.0f} s, "
              f"{results['seconds_per_frame'] * 1000:.0f} ms per frame)")
        print(f"   barrier from F⁻···CH₃Cl {results['barrier_kcal']:.2f} kcal/mol (CCSD(T) 3.39), "
              f"complex to complex {results['complex_to_complex_kcal']:.1f} (ωB97X-D −27.1)")
        print(f"   TS r(C–F)/r(C–Cl) {ts['r_CF']:.3f}/{ts['r_CCl']:.3f} Å (CCSD(T) 2.025/2.112), "
              f"imaginary {', '.join(f'{w:.0f}' for w in ts['imaginary_cm'])} cm⁻¹ "
              f"(ωB97X-D −450), converged {ts['converged']}")
        print(f"   F⁻···CH₃Cl r(C–F)/r(C–Cl) {rc['r_CF']:.3f}/{rc['r_CCl']:.3f} Å "
              "(CCSD(T) 2.498/1.843)")
        for path, row in against_wb97xd(folder, reference).items():
            print(f"   vs ωB97X-D, {path}: barrier {row['barrier_model_kcal']:.2f} "
                  f"({row['barrier_reference_kcal']:.2f}), max |ΔE| "
                  f"{row['max_abs_energy_error_meV']:.1f} meV, RMSE "
                  f"{row['energy_rmse_meV']:.1f} meV, force RMSE "
                  f"{row['force_rmse_ev_per_A']:.3f} eV/Å")
        relative = results.get("relative_to_reactants_kcal")
        table = json.loads((WORK / "uma" / "fragments_wb97mv.json").read_text()).get(
            "table", {}) if (WORK / "uma" / "fragments_wb97mv.json").exists() else {}
        if not relative and {"F-", "Cl-"} <= set(table):
            # No references file on the cluster: UMA's table ions from the desktop run.
            fragments = results["fragments_ev"]
            energy = {name: point["energy_ev"] for name, point in results["stationary"].items()}
            zero = fragments["CH3Cl"] + table["F-"]
            relative = {"reactant_complex": (energy["reactant_complex"] - zero) * KCAL,
                        "ts": (energy["ts"] - zero) * KCAL,
                        "product_complex": (energy["product_complex"] - zero) * KCAL,
                        "products": (fragments["CH3F"] + table["Cl-"] - zero) * KCAL}
            results["errors"] = [e for e in results.get("errors", []) if not e.startswith(
                ("F-:", "Cl-:"))] + [f"ions from UMA's table via {table['from']} (desktop)"]
        if relative:
            print("   relative to F⁻ + CH₃Cl (kcal/mol; CCSD(T) −15.6, −12.2, −41.6, −31.9): "
                  + ", ".join(f"{k} {v:.1f}" for k, v in relative.items()))
        for error in results.get("errors", []):
            print(f"   note: {error}")


if __name__ == "__main__":
    main()
