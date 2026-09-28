"""Step 3 (laptop, after the campaign): train on the HSE06 labels.  -> models/<name>/

Reads ``campaign/labeled.extxyz`` (written by ``collect_vasp_labels`` on the
returned outputs), or ``dry_run/labeled.extxyz`` with ``--dry-run``
(``fake_labels.py``), plus the doped series when it has been collected
(``doped_campaign/labeled.extxyz``, or ``dry_run/labeled_doped.extxyz``). The
``md600`` frames and the doped ``*_test`` frames are held out; the rest train:

- ``mace-mp0+delta``: a small MACE from scratch on HSE06 − MACE-MP-0 (energy,
  forces, stress); its card names MACE-MP-0 (file and SHA-256) as the baseline,
  so the tool always runs it as MACE-MP-0 + correction;
- ``direct``: MACE-MP-0 fine-tuned on HSE06 (plain; per-element offsets move the
  VASP HSE06 energies onto the foundation scale), for comparison.

Three seeds each. ``--quick`` trains for a tenth of the steps (a pipeline check).

``--package DIR`` writes the two trainings as HPC packages instead (one SLURM
array each, one task per seed, GPU): the residual labels are computed here, so
the cluster needs only mace-torch. ``--smoke`` makes them 3 epochs and 1 seed, a
check that stress training and the 40-atom fine-tune fit the cluster's GPU. Copy
the ``runs/`` folders back and install them into ``<run>/models`` with
``install_models``; then run ``evaluate.py`` here.

Stress weights: the Δ stresses are small (a few meV/Å³, a fraction of a GPa), so
with mace-torch's usual stress weight they add ~1 % to the loss and are not
learned at all; forces barely constrain them, since pair contributions cancel in
the forces of a near-perfect crystal but add up in the stress. The dry run
needed a weight of 10⁴ (stress error 0.56 → 0.02 GPa, forces unchanged). The
direct fine-tune's full stresses are ~10× larger: 10³. Its seeds train one at a
time: three MACE-MP-0 fine-tunes with stress on 40 atoms do not fit in 8 GB.
"""

import sys
from pathlib import Path

from ase.io import write
from common import (
    BATCH,
    FOUNDATION,
    STEPS_PER_MODEL,
    VALID_FRACTION,
    epochs,
    held_out,
    labels,
    mace_mp0,
    run_dir,
)

from samson_mlip_visualizer.delta import delta_e0s, delta_labels, mace_baseline_card
from samson_mlip_visualizer.finetune import fit_element_offsets
from samson_mlip_visualizer.training import (
    GpuSlurmSettings,
    TrainingSpec,
    install_models,
    train_local,
    write_training_package,
)

DRY = "--dry-run" in sys.argv
PACKAGE = Path(sys.argv[sys.argv.index("--package") + 1]) if "--package" in sys.argv else None
SMOKE = "--smoke" in sys.argv
STEPS = STEPS_PER_MODEL // (10 if "--quick" in sys.argv else 1)
run = run_dir(DRY)
labeled = labels(DRY)
train = [f for f in labeled if not held_out(f)]
print(f"{len(train)} training frames, {len(labeled) - len(train)} held out", flush=True)
data = run / "data"
data.mkdir(exist_ok=True)

base = mace_mp0()
delta = delta_labels(train, base, energy_key="HSE06_energy", forces_key="HSE06_forces",
                     stress_key="HSE06_stress")
write(data / "delta.extxyz", delta)
foundation_energies = [f.info["BASE_energy"] for f in delta]
offsets = fit_element_offsets(train, [f.info["HSE06_energy"] for f in train],
                              foundation_energies)
direct = []
for frame in train:
    image = frame.copy()
    image.info["REF_energy"] = offsets.to_foundation_scale(frame.info["HSE06_energy"], frame)
    image.info["REF_stress"] = frame.info["HSE06_stress"]
    image.arrays["REF_forces"] = frame.arrays["HSE06_forces"]
    direct.append(image)
write(data / "direct.extxyz", direct)

card = {"reference": "HSE06 (VASP, PAW PBE, 520 eV, Γ-centered 0.3 Å⁻¹ mesh)",
        "elements": ", ".join(sorted({s for f in train for s in f.get_chemical_symbols()})),
        "scope": "Ba2BiVO6 (and V-site Nb/Ta if in the data); not a general-purpose model",
        "trained_on": f"{len(train)} frames" + (" (SYNTHETIC dry-run labels)" if DRY else "")}
common = dict(seeds=(1, 2, 3), batch_size=BATCH, valid_fraction=VALID_FRACTION,
              device="cuda", epochs=epochs(len(train), STEPS))
specs = [
    TrainingSpec(name="mace-mp0+delta", foundation="", train_file=str(data / "delta.extxyz"),
                 mode="scratch", e0s=delta_e0s(delta), energy_key="DELTA_energy",
                 forces_key="DELTA_forces", stress_key="DELTA_stress", lr=0.01,
                 stress_weight=1e4,
                 card={**card, "target": "HSE06 − MACE-MP-0",
                       "delta_baseline": mace_baseline_card(FOUNDATION)}, **common),
    TrainingSpec(name="direct", foundation=FOUNDATION, train_file=str(data / "direct.extxyz"),
                 stress_weight=1e3, card={**card, "energy_offsets_eV": offsets.offsets},
                 **common),
]
if PACKAGE:
    slurm = GpuSlurmSettings(time="01:00:00" if SMOKE else "12:00:00", memory_gb=32)
    for spec in specs:
        if SMOKE:
            spec = TrainingSpec.from_json({**spec.to_json(), "epochs": 3, "seeds": (1,)})
        write_training_package(spec, PACKAGE / spec.name, slurm=slurm)
    (PACKAGE / "submit_all.sh").write_text(
        "#!/bin/bash\nset -e\n" + "".join(f"(cd {spec.name} && sbatch run_train.slurm)\n"
                                           for spec in specs), newline="\n")
    (PACKAGE / "README.md").write_text(
        f"# Ba2BiVO6 HSE06 training: {', '.join(s.name for s in specs)}\n\n"
        + ("Smoke test on SYNTHETIC dry-run labels, 3 epochs, 1 seed: checks that "
           "mace-torch trains with stress on the cluster GPU (the direct 40-atom "
           "fine-tune did not fit three at a time in 8 GB).\n\n" if SMOKE else "")
        + "Fill in the placeholders in each `run_train.slurm`, then `bash submit_all.sh`. "
        "Copy each `<model>/runs/` back and install it into the run's `models/` folder "
        "with `install_models`, then run `evaluate.py` on the desktop.\n",
        encoding="utf-8")
    print(f"{len(specs)} training packages -> {PACKAGE}")
    raise SystemExit
for spec in specs:
    runs = train_local(spec, run / "runs" / spec.name, parallel=spec.mode == "scratch")
    print(f"{spec.name}: " + ", ".join(f"seed {r.seed} {'ok' if r.ok else 'FAILED'} "
                                       f"{r.seconds:.0f} s" for r in runs), flush=True)
    install_models(run / "runs" / spec.name, destination=run / "models")
