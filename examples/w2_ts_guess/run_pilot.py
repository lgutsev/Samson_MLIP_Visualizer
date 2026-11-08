"""Run the three Fe2 pilot cases two ways and write ``pilot_results.json``.

    PYTHONPATH=../../src python run_pilot.py [--out D:/MLIP_Work_Folder/w2_ts_guess]

1. ``frame``: P-RFO from the stand-in TS frame alone, barriers from an MLIP IRC
   (what happens with a failed run's best index-1 frame and nothing else).
2. ``frame+ends``: the same frame plus the two W2 minima (``--strategy auto``:
   P-RFO, then QST3 if the saddle does not connect the minima on the MLIP).

Also records the MLIP and UBPW91 energy differences between the two W2 minima at
their UBPW91 geometries, as a check on how far the MLIP's energetics can be
trusted for each case.
"""

import argparse
import json
from pathlib import Path

from ase.io import read
from refine_ts_guess import DEFAULT_MODEL, load_calculator, main, tag

HERE = Path(__file__).resolve().parent
CASES = ("fe2h2_m9", "fe2h2o2_m7", "fe2h2o3_m7")
KEEP = ("ok", "saddle", "method", "converged", "steps", "n_imaginary", "imaginary_cm",
        "barrier_forward_ev", "barrier_reverse_ev", "rmsd_from_start", "endpoint_rmsd",
        "irc_connects_endpoints", "notes", "attempts")


def compact(entry):
    out = {k: entry[k] for k in KEEP if k in entry}
    for key in ("barrier_forward_ev", "barrier_reverse_ev", "rmsd_from_start"):
        if out.get(key) is not None:
            out[key] = round(out[key], 3)
    out["attempts"] = [{k: a[k] for k in ("method", "n_imaginary", "imaginary_cm",
                                           "irc_connects_endpoints")}
                       for a in out.get("attempts", [])]
    return out


def endpoint_check(calc):
    rows = {}
    for name in CASES:
        energies = {}
        for role in ("reactant", "product"):
            atoms = read(HERE / "cases" / name / f"{role}.extxyz")
            tag(atoms, atoms.info["charge"], atoms.info["multiplicity"])
            atoms.calc = calc
            energies[role] = (float(atoms.get_potential_energy()), atoms.info["REF_energy"])
        rows[name] = {
            "dE_product_minus_reactant_UBPW91_eV":
                round(energies["product"][1] - energies["reactant"][1], 3),
            "dE_product_minus_reactant_MLIP_eV":
                round(energies["product"][0] - energies["reactant"][0], 3),
        }
    return rows


def run():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(r"D:\MLIP_Work_Folder\w2_ts_guess"))
    args = parser.parse_args()
    template = str(HERE / "w2_ts_template_example.gjf")
    results = {"model": str(DEFAULT_MODEL), "cases": {}}
    for name in CASES:
        folder = HERE / "cases" / name
        main([str(folder / "guess.extxyz"), "--route-from", template,
              "--out", str(args.out / "frame"), "--write-failed"])
        main([str(folder / "guess.extxyz"), "--reactant", str(folder / "reactant.extxyz"),
              "--product", str(folder / "product.extxyz"), "--route-from", template,
              "--out", str(args.out / "frame_ends")])
        frame = json.loads((args.out / "frame" / "report.json").read_text())["jobs"]
        ends = json.loads((args.out / "frame_ends" / "report.json").read_text())["jobs"]
        results["cases"][name] = {
            "reaction": read(folder / "guess.extxyz").info["reaction"],
            "frame": compact(frame[f"{name}_guess"]),
            "frame+ends": compact(ends[name]),
        }
    calc = load_calculator(DEFAULT_MODEL, "cpu")
    for name, row in endpoint_check(calc).items():
        results["cases"][name]["W2_minima_at_UBPW91_geometries"] = row
    (HERE / "pilot_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    run()
