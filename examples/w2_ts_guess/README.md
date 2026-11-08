# Restarting failed W2 Gaussian saddle searches from MLIP guesses (Fe₂ pilot)

ClusterMLIP's Warehouse 2 saddle campaign (`gaussian_spin_qb3_g09_v3`, G09
UBPW91/6-311++G* on LONI QB3) left 16 + 5 TS searches that should restart from
their best index-1 frame. Most are Fe₂ systems. This pilot refines such a frame
on the charge/spin-embedded Fe₂ MACE from
[`../fe_spin_ladder/train_fe2_spin.py`](../fe_spin_ladder/train_fe2_spin.py),
checks it on the MLIP, and writes a Gaussian 09 TS restart that keeps the
campaign's own route. **It does not run DFT. The Gaussian Freq run decides
whether the saddle is real; the MLIP only proposes it.**

Nothing here is new machinery. [`ts_guess.refine_guess`](../../src/samson_mlip_visualizer/ts_guess.py)
chains `ts.prfo_search` (Sella P-RFO with an exact initial Hessian, like
`CalcFC`), `reaction_path.qst` (IDPP → CI-NEB → P-RFO), `reaction_path.irc` and
`vibrations.harmonic_frequencies`. [`qm_export.gaussian_ts_input_from_template`](../../src/samson_mlip_visualizer/qm_export.py)
writes the restart.

## Run it

```
PYTHONPATH=../../src python refine_ts_guess.py FRAMES.extxyz --route-from W2_JOB.gjf --out DIR
PYTHONPATH=../../src python refine_ts_guess.py FRAME.extxyz --reactant R.extxyz --product P.extxyz \
    --route-from W2_JOB.gjf --out DIR
```

Use any Python with mace-torch and sella (SAMSON's works). The model path
defaults to `D:\MLIP_Work_Folder\fe_spin_ladder\fe2_spin\seed1\fe2_spin_seed1.model`
and can be set with `--model` or `FE2_SPIN_MODEL`.

- **Input**: extxyz frames carrying `charge` and `multiplicity` in `info`
  (`mult`, or W2's `spin` = multiplicity, also work; or pass
  `--charge`/`--multiplicity`). Every frame of every file is its own guess.
  With `--reactant`/`--product` (IRC endpoints or the two minima, in the
  frame's atom order) give at most one frame.
- **Refinement**: with a frame only, P-RFO from it, then an MLIP IRC to both
  sides for the barriers. With a frame and endpoints (`--strategy auto`), P-RFO
  first. If that saddle's IRC does not reach both endpoints (0.25 Å RMSD), it
  tries QST3, and keeps whichever passes; if neither passes, it keeps the
  P-RFO saddle, the one closest to the DFT frame. Endpoints alone give QST2.
  The endpoints are relaxed on the MLIP first.
- **Check**: converged to `--fmax` (0.01 eV/Å), exactly one imaginary MLIP
  mode, and, with endpoints, an IRC that connects them. The output reports the
  MLIP barrier from both sides and the RMSD from the starting frame (after
  superposition).
- **Output** in `DIR`: `<name>.ts.gjf` (only for guesses that pass, unless
  `--write-failed`), `<name>.mlip_ts.extxyz` and `report.json` (per job:
  imaginary modes, barriers, RMSDs, notes, any rejected attempt, and every
  change made to the template route).

`run_pilot.py` runs the three cases below both ways and writes
`pilot_results.json`. `make_cases.py` rebuilt the cases from the W2 key frames.

## The Gaussian restart

The route of `--route-from` is copied token for token, with these changes only:

- the Opt keyword, whatever it was, becomes `Opt=(TS,CalcFC,NoEigenTest)`
  (any `MaxCycles` etc. in the old Opt goes; each removal is logged in
  `route_changes`), or is added if missing;
- `Freq` is added if missing (an existing `Freq=...` stays as it is);
- `Geom=Checkpoint`/`AllCheck` is removed, because the geometry is written as
  explicit Cartesians. Gaussian dies in l101 when both are given.

Method/basis, `IOP(...)`, `SCF=(...)`, `Int=`, `Pop=`, `Guess=` and all Link 0
lines (`%mem`, `%nprocshared`, ...) stay as they are. A template without
`IOP(5/13=1)` (alone or in a combined `IOP(5/13=1,5/36=1,...)`) is refused
before any MLIP work. `%chk` is renamed to `<name>.ts.chk` so that restarts do
not overwrite each other. If the route has `Guess=Read`, the old checkpoint is
kept as `%oldchk`, so it must be next to the job (or drop `Guess=Read`).
Charge and multiplicity come from the frame, not the template, with a note when
they differ. Only the first job of a `--Link1--` template is used.

`w2_ts_template_example.gjf` is an **illustrative** W2-style template (the
route from the task description). Use each failed job's real `.gjf` instead.

## Pilot cases

No failed-saddle frames have been exported yet, so each case pairs two W2
UBPW91 minima (same charge and multiplicity, one H shift apart) with their IDPP
midpoint, a stand-in for a failed run's best index-1 frame (`cases/`).

Fe₂ spin-MACE, float64, CPU, about 1–2 min per case (`pilot_results.json`):

| Case (q = 0) | Reaction | P-RFO from frame | Imag. mode (cm⁻¹) | RMSD from frame (Å) | Barrier fwd / rev (eV) | MLIP IRC reaches the W2 minima? | `.gjf` |
|---|---|---|---|---|---|---|---|
| Fe₂H₂, M = 9 | terminal Fe–H → Fe–H–Fe | converged, 27 steps | −395 (one) | 0.44 | 0.26 / 0.26 | **no**: Fe–Fe stretch saddle, one IRC end at Fe–Fe 1.49 Å; QST3 gives 2 imaginary modes | not written |
| Fe₂H₂O₂, M = 7 | μ-O–H → Fe–H | converged, 113 steps | −793 (one) | 0.37 | 0.78 / 0.56 | **no**: H leaves the OH group without reaching Fe; QST3 gives another saddle (−1433) | not written |
| Fe₂H₂O₃, M = 7 | μ-O–H → Fe–H | converged, 13 steps | −520 (one) | 0.28 | 0.61 / 0.03 | **yes** (0.19 / 0.20 Å); QST3 finds the same saddle | written |

Barriers are measured from the endpoints after MLIP relaxation. With the frame
alone, P-RFO gives the same three saddles, and the barriers then come from the
minima its own IRC reaches. In that mode all three pass the MLIP check (one
imaginary mode), and nothing shows that two of them are the wrong reaction.
**When ClusterMLIP exports frames, export the IRC endpoints too.**

**Can the MLIP's energies be trusted here?** Product − reactant at the W2
UBPW91 geometries:

| | UBPW91 | Fe₂ spin-MACE |
|---|---|---|
| Fe₂H₂ M9 | −0.38 | −0.01 |
| Fe₂H₂O₂ M7 | +0.15 | +0.23 |
| Fe₂H₂O₃ M7 | −0.46 | +0.03 |

It misses by up to 0.5 eV on these isomer pairs. The Fe₂H₂O₃ "reactant" frame
is itself odd: its H sits halfway between O and Fe (O–H 1.44 Å, Fe–H 1.58 Å),
which is close to the MLIP saddle (O–H 1.95 Å, Fe–H 1.55 Å). So the geometry is
a reasonable TS guess, but the 0.61 eV barrier is not a prediction. The DFT run
decides.

## Caveats

- **UBPW91 is the reference.** UMA is not used here. The Fe spin-ladder
  benchmark found UMA spin gaps uncorrelated with UBPW91 (off by 12–14 eV on
  Fe16), so OMol25/hybrid-trained models are not a better reference. If UMA is
  ever added, it is only for comparison and never gives labels.
- **The spin-MACE was trained on relaxation frames near minima**, not on
  transition states. Its saddles are guesses, and its barriers can be off by
  tenths of an eV or worse (see the endpoint check above). At UMA's Fe₂O₄ + N₂
  TS it put the saddle below the intermediate (`../fe_spin_ladder`).
- **Out of scope**: Fe16 (a different model, ClusterMLIP's Fe16 v1) and anything
  outside Fe₂XY (X, Y = H, N, O). The script adds a note when a frame is
  outside that set.
- **One seed, no committee**: there is no uncertainty estimate on these guesses.
- RMSDs use a fixed atom order (no permutation of identical atoms), so a
  frame whose H labels differ from the endpoints' looks farther than it is.
