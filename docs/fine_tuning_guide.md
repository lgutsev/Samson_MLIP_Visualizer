# Fine-tuning guide

How this tool fine-tunes a MACE foundation model for your own chemistry, and
how to run each step. The worked example, HCN ⇌ HNC, is written up with all
its numbers in [fine_tuning.md](fine_tuning.md).

## When to fine-tune

Fine-tune when the foundation model is wrong for your system, not by default.
The [path benchmark](../README.md#benchmarking-a-model-along-a-reaction-path)
shows where it is wrong: for HCN, MACE-MP-0 was right at both minima but
0.5–0.65 eV too high across the whole bent region, because its training data
(Materials Project crystals) never contain a hydrogen shifting between two
atoms. That region is where new training data belongs.

## The loop

![The fine-tuning workflow](images/finetuning_workflow.svg)

Each round costs a few DFT calculations and a few minutes of GPU time. The
MLIP steps (teal) run on the desktop in seconds to minutes; the reference
calculations (coral) run locally with Psi4 for small molecules, or on the HPC
from a package this tool writes. Nothing is ever submitted to a cluster by the
tool: you copy the package, run it, and copy the results back.

1. **Explore** with the current model: TS searches, IRCs, scans, and MD
   (the panel, `samson-mlip`, or the bridge). Run the path benchmark
   (`samson-mlip-benchmark`) to see where the model disagrees with the
   reference.
2. **Select frames** to label (`finetune.select_for_labeling`), from three
   sources: where a committee disagrees, frames farthest from everything
   labeled so far (diversity), and a few random spot checks. Every frame is
   kept a minimum distance from the others. Disagreement never selects alone:
   a committee of seeds from one foundation model is overconfident.
3. **Label with DFT.** Locally with the Psi4 backend, or as a Gaussian or ORCA
   package for SLURM (`labeling.write_label_package`). Back on the desktop,
   `labeling.collect_labels` checks every output (normal termination, SCF
   converged, the geometry is the frame it was made for) before accepting it.
4. **Fine-tune** MACE-MP-0 on the labels (below).
5. **Evaluate** on DFT frames the model has not seen, reported next to their
   distance from the training data, plus the chemistry the model must not
   forget.
6. **Repeat or stop.** Stop when the properties you care about are within
   your tolerance of the reference and the off-path error has stopped
   improving, never on committee spread alone. Then install the model with its
   model card.

**Automated: `samson-mlip-finetune config.json`** runs these steps as rounds
(`samson_mlip_visualizer.active_learning`). Each round trains a committee (or
reuses it), explores from a TS guess (P-RFO with an exact Hessian,
frequencies, IRC, and an optional bond scan), evaluates against the reference
on frames of that round's own IRC and scan that it was not trained on, and
then stops or selects (the three sources above; the held-out frames are never
candidates), labels through a cached reference (`reference_cache`: each
structure computed once), and adds the labels to the training set. It stops
only when the barrier, the largest IRC error and the largest scan error are all
within your tolerances; the committee spread is logged next to the real error
every round and used for selection, never for stopping. Two model plug-ins:
MACE (a committee of seeds, `training.train_local`) and AIMNet2 (members
fine-tuned from different AIMNet2 ensemble members in the aimnet environment,
charge per structure). Every round writes its training set, models,
exploration, evaluation, selection manifest and labels under `round_NN/` plus a
`rounds.json`, and a restarted run picks up where it stopped. Worked example,
with the round table:
[examples/sn2_f_ch3cl](../examples/sn2_f_ch3cl/README.md#active-learning-with-the-library-loop)
(converged in one round of selection, about 5 minutes of DFT).

## Inside the fine-tuning step

![Multihead fine-tuning of MACE](images/finetuning_multihead.svg)

MACE has a shared body, which turns each atom's neighborhood into features,
and small readout heads, which turn features into energies. Two modes:

- **Plain** (left side only): the whole model is trained on your labels. It is
  fast and effective on the target, but the shared body drifts: the HCN model
  got C≡C 0.04 Å too short.
- **Multihead** (both sides): your labels train a new `Default` head, while a
  replay set of Materials Project structures, the data MACE-MP-0 was trained
  on, keeps training the original head (`pt_head`). Because the shared body
  must keep serving both, it cannot drift far from what it already knew. The
  calculator uses `Default` automatically.

Both modes keep all 89 foundation elements
(`--foundation_model_elements=True`). Without that flag mace-torch rebuilds
the element table from the data: the plain HCN model knew only H, C and N,
and a multihead run kept 83 of 89.

```python
from samson_mlip_visualizer.paths import foundation_model
from samson_mlip_visualizer.training import TrainingSpec, install_models, train_local

spec = TrainingSpec(
    name="my_reaction_r1",
    foundation=str(foundation_model()),        # MACE-MP-0 small
    train_file="round1/gaussian/labeled.extxyz",
    mode="multihead", replay="mp",             # or mode="plain"
    replay_samples=10000, seeds=(1, 2, 3), epochs=120,
    card={"reference": "PBE/def2-TZVP (Gaussian 16)", "scope": "my reaction"},
)
train_local(spec, "round1/train")             # desktop GPU, seeds in parallel
install_models("round1/train")                # into the MACE folder, with model cards
```

For the HPC, `training.write_training_package(spec, folder)` writes the same
run as a SLURM GPU array (one task per seed), a login-node script that
downloads the replay set (compute nodes often have no internet), and a README.
In multihead mode mace-torch sets its own learning rate (10⁻⁴).

### What training looks like

![Validation error while fine-tuning on HCN](images/hcn_finetune_learning_curves.png)

The HCN run: three models (seeds 1–3) trained side by side on the laptop GPU,
120 epochs in 257 s. Epoch 0 is essentially MACE-MP-0 on this data (about
250 meV/atom and 260 meV/Å). The error falls by an order of magnitude within
ten epochs, then improves slowly and noisily, and mace-torch keeps the best
epoch (here 110–117). These validation frames are a random 10 % of the
training set, so they sit right next to training frames: the curve shows that
training converged, not how good the model is. That is what the evaluation on
held-out frames is for.

## Energy scales

Gaussian, ORCA, and Psi4 energies are all-electron or on other pseudopotential
scales; MACE-MP-0 was trained on VASP. They differ by roughly a constant per
atom of each element. `finetune.fit_element_offsets` fits those offsets from
a few structures labeled by both codes and reports the residual; pass the fit
to `collect_labels` to put `REF_energy` on the foundation scale (the raw
energy is kept as `REF_energy_raw`). Do this check before the first labeling
round. With one composition only (as for HCN) the offsets are not unique and
the fit says so.

## Where things are stored

- The MACE folder, `~/.cache/mace`, holds the foundation model and installed
  fine-tuned models (`finetuned/<name>/`). The panel's default model comes
  from here, so it works with no other drive attached.
- An optional mirror folder on a second drive (`paths.set_mirror_dir(...)`),
  here `D:\MLIP_Work_Folder\cache\mace`, gets a copy of every installed model
  and takes the large downloads: the Materials Project replay set is about
  595 MB. It is used only while the drive is connected.
- Labeling and training packages, run folders, and figures go wherever you
  write them; on this machine, `D:\MLIP_Work_Folder`.

## Before the first real campaign: HPC smoke tests

`examples/hpc_smoke_tests/make_smoke_tests.py` writes four small jobs:
- Gaussian and ORCA labeling of three HCN/HNC geometries, with a Psi4 PBE
  reference to compare against;
- plain and multihead fine-tuning on a GPU node.

Run them on the cluster, copy the results back, and run
`check_smoke_results.py`, which prints PASS or FAIL per test. Both training
tests pass when run locally exactly as SLURM would run them (28 s and 184 s).
The labeling collectors have so far been tested only on outputs written to the
Gaussian 16 and ORCA 5 formats, so the smoke tests are also their first check
against real output.

## Δ-learning instead of fine-tuning

When a cheap quantum method gets the shape of the surface right, train a
small MACE from scratch on the residual E_ref − E_xTB instead of fine-tuning
the foundation model (`TrainingSpec(mode="scratch", e0s=delta_e0s(frames),
energy_key="DELTA_energy", forces_key="DELTA_forces")`, labels from
`samson_mlip_visualizer.delta.delta_labels`). Put the baseline in the model
card (`card={"delta_baseline": xtb_baseline_card("gfn1")}`) and the tool will
always evaluate the model as xTB + correction. For HCN it needed about a
quarter of the labels for better accuracy near the path. Like any MLIP, the
correction must be trained where it will be used, and bond breaking needed
labels of its own:
[examples/delta_xtb_hcn](../examples/delta_xtb_hcn/README.md).

The baseline can also be a MACE model (`mace_baseline_card(model)`, which
records the file name and SHA-256): MACE-MP-0 plus a correction trained on
HSE06 − MACE-MP-0 gives hybrid-functional quality with no DFT at run time.
Stress is part of the labels and the model for periodic cells. Give it a
large weight: a residual's stresses are a fraction of a GPa, so at mace-torch's
usual weights they add about 1 % to the loss and are simply not learned
(forces do not pin them down either, because pair contributions cancel in a
near-perfect crystal's forces but add up in its stress). In the BBVO dry run,
stress weight 10 left the stress error at 0.56 GPa; 10⁴ brought it to 0.02 GPa
with unchanged forces. `TrainingSpec(stress_weight=...)` switches mace-torch to
its stress loss (`--loss=stress --compute_stress=True`). Two more worked
examples: [a metal carbonyl](../examples/delta_nico4/README.md) (Ni(CO)₄,
PBE0, three baselines, plus an ORCA package for Ni porphine on an HPC) and
[a double perovskite](../examples/delta_hse06_bbvo/README.md) (Ba₂BiVO₆,
HSE06 labels from VASP on an HPC).

## VASP labeling (periodic frames)

`samson_mlip_visualizer.vasp_labeling.write_vasp_package` turns
`frames.extxyz` + `manifest.json` into a VASP package in which every frame is
computed at several levels (default PBE+U, then HSE06 restarted from its
WAVECAR), all sharing one Γ-centered KPOINTS, ENCUT, PREC, LREAL, and ISYM, so
that differences between the levels come from the method alone. POTCARs and U
follow pymatgen's `MPRelaxSet.yaml`, the settings of MACE-MP-0's training
data; POTCARs are concatenated on the cluster from `<POTPAW_PBE_DIR>`, never
copied. `collect_vasp_labels` keeps a frame only when every level finished,
reached EDIFF, and matches the frame's cell and geometry, and writes
`<LEVEL>_energy` / `_forces` / `_stress`. The parser has only been tested on
synthetic data so far, so run the BBVO smoke test (`hpc_smoke_tests/05_vasp_bbvo`)
before a campaign.

## Not built yet

- An evaluation report: errors binned by distance from the training data,
  and a forgetting table (the loop already logs committee spread against the
  real error every round).
- Panel buttons for exporting frames, loading labels, and running the loop.
- The HCN example rebuilt on the new modules, as an end-to-end regression
  test.
- A test of the VASP collector on real vasprun.xml/OUTCAR files (the BBVO
  smoke test).
