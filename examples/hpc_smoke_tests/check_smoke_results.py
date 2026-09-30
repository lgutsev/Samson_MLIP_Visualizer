"""Check what came back from the HPC smoke tests.

Run in SAMSON's Python (where samson-mlip-visualizer and mace-torch are
installed), from the smoke-test folder, after copying back each package's
outputs/ (labeling) or runs/ (training):

    python check_smoke_results.py              # all four
    python check_smoke_results.py 02 04        # only the tests starting with 02 or 04
    python check_smoke_results.py --no-write   # install nothing, write no smoke_results.json

It prints PASS / FAIL / NOT RUN per test and writes smoke_results.json. The
training checks install the models into installed_models/ (with --no-write: into
a temporary folder instead).
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
# Gaussian (no density fitting) and ORCA (RI-J) against Psi4 (DF), same functional
# and basis: relative energies agree to a few meV, forces to ~0.01 eV/Å.
ENERGY_TOLERANCE, FORCE_TOLERANCE = 0.02, 0.05
results = {}
NO_WRITE = "--no-write" in sys.argv
SELECTED = [arg for arg in sys.argv[1:] if not arg.startswith("--")]


def record(name, status, **detail):
    results[name] = {"status": status, **detail}
    extra = ", ".join(f"{key} {value}" for key, value in detail.items())
    print(f"{status:8s} {name}" + (f": {extra}" if extra else ""))


def check_labeling(name):
    from samson_mlip_visualizer.finetune import Manifest
    from samson_mlip_visualizer.labeling import collect_labels

    package = HERE / name
    if not (package / "outputs").is_dir():
        return record(name, "NOT RUN", reason="no outputs/ copied back")
    result = collect_labels(package)
    if result.rejected or len(result.labeled) != 3:
        return record(name, "FAIL", rejected=result.report()["rejected"])
    names = Manifest.load(package / "manifest.json").notes["names"]
    energies = {n: f.info["REF_energy_raw"] for n, f in zip(names, result.labeled, strict=True)}
    forces = {n: f.arrays["REF_forces"] for n, f in zip(names, result.labeled, strict=True)}
    detail = {
        "code": result.labeled[0].info["code"],
        "barrier_eV": round(energies["TS"] - energies["HCN"], 4),
        "HNC_minus_HCN_eV": round(energies["HNC"] - energies["HCN"], 4),
    }
    reference_file = HERE / "reference_psi4_pbe_def2tzvp.json"
    if not reference_file.is_file():
        return record(name, "PASS", **detail, note="no Psi4 reference to compare with")
    reference = json.loads(reference_file.read_text())
    energy_error = max(
        abs((energies[n] - energies["HCN"])
            - (reference[n]["energy_ev"] - reference["HCN"]["energy_ev"]))
        for n in names
    )
    force_error = max(
        float(np.abs(forces[n] - np.array(reference[n]["forces_ev_per_A"])).max()) for n in names
    )
    ok = energy_error < ENERGY_TOLERANCE and force_error < FORCE_TOLERANCE
    record(name, "PASS" if ok else "FAIL", **detail,
           relative_energy_vs_psi4_eV=round(energy_error, 4),
           max_force_vs_psi4_eV_per_A=round(force_error, 4))


def check_training(name, expect_heads):
    from ase import Atoms
    from ase.build import molecule

    from samson_mlip_visualizer.calculators import create_calculator
    from samson_mlip_visualizer.compat import supported_species
    from samson_mlip_visualizer.training import install_models

    package = HERE / name
    if not (package / "runs").is_dir():
        return record(name, "NOT RUN", reason="no runs/ copied back")
    destination = Path(tempfile.mkdtemp()) if NO_WRITE else HERE / "installed_models"
    installed = install_models(package, destination=destination)
    if not installed:
        return record(name, "FAIL", reason="no .model file in runs/seed*/")
    calc = create_calculator("mace", str(installed[0]), device="cpu")
    heads = list(getattr(calc.models[0], "heads", []))
    elements = len(supported_species(calc) or [])
    foundation = next((package / "foundation").glob("*"), None)
    expected = 89  # MACE-MP-0
    if foundation is not None:
        expected = len(supported_species(create_calculator("mace", str(foundation), device="cpu")))
    hcn = Atoms("CNH", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.07]])
    water = molecule("H2O")
    energies = []
    for atoms in (hcn, water):
        atoms.calc = calc
        energies.append(atoms.get_potential_energy())
    card = json.loads(Path(str(installed[0]) + ".json").read_text())
    ok = elements == expected and all(np.isfinite(energies)) and len(heads) >= expect_heads
    record(name, "PASS" if ok else "FAIL", elements=f"{elements} of {expected}", heads=heads,
           errors=card.get("training_errors_meV"), model=str(installed[0]))


CHECKS = {
    "01_label_gaussian": check_labeling,
    "02_label_orca": check_labeling,
    "03_train_plain": lambda name: check_training(name, expect_heads=1),
    "04_train_multihead_mp": lambda name: check_training(name, expect_heads=2),
}
for test, check in CHECKS.items():
    if not SELECTED or any(test.startswith(prefix) for prefix in SELECTED):
        check(test)
if not NO_WRITE:
    (HERE / "smoke_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {HERE / 'smoke_results.json'}")
