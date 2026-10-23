# UMA presets: GPU validation and benchmark (2026-10-23)

- **Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (Blackwell, sm 12.0, 8 GB),
  driver 591.44, Windows 11.
- **Software:** fairchem-core 2.24.0, torch 2.13.0+cu130 ([uma.md](uma.md)).
- **Path tested:** the bridge end to end. `UMACalculator` → shared worker →
  `uma_config.build_calculator`, with the bridge's own `engine.relax` (FIRE) and
  `md.run_md` (Langevin).
- **Script:** [scripts/uma_validate.py](../scripts/uma_validate.py).
- **Raw output:** `D:\MLIP_Work_Folder\uma_nitro_validation\uma_validation.{json,log}`.

These are integration and numerical-consistency checks. They are not accuracy
benchmarks: accuracy needs DFT references at each preset's level of theory, and
energies from different presets or tasks are different levels of theory, not
implementation errors. fairchem's ~6 ns/day for 1,000 atoms on an H200 applies
only to Nitro on that GPU and was not extrapolated here.

## What ran

| | Requested `nitro` | Requested `high-precision` |
|---|---|---|
| Effective mode | **`multigpu`** (merged MoLE, TF32), `umas_fast_pytorch`, no compile | `high-precision` (merged MoLE, FP32), `umas_fast_pytorch`, no compile |
| Fallback reasons (logged and in provenance) | Nitro's kernels need Triton, which has no official Windows build; `umas_fast_gpu` and `torch.compile` need Triton too | the same Triton reason for the backend and compile |

**UMA Nitro did not run on this machine.** The checkpoint is on Nitro's
allowlist, so the reason is the platform, not the model. The comparison below is
therefore fairchem's reduced-precision fallback (TF32) against FP32, on the same
checkpoint (`uma-s-1p2p1.pt`, md5 `3497615f…`), task, dispersion and
coordinates.

The fine-tuned checkpoints (`-ft-d4rev`, `-ft-r2scan`) were **not run**.
`facebook/UMA` answered `GatedRepoError 401` because no Hugging Face credentials
are configured here. Their presets are covered only by the rule checks and the
fake-worker tests.

## Requested `nitro` (effective `multigpu`) vs `high-precision`

| Structure | Task / D3 | Atoms | ΔE (meV) | ΔE/atom (meV) | max \|ΔF\| (meV/Å) | RMS ΔF (meV/Å) |
|---|---|---|---|---|---|---|
| NH₃ (`nh3_ts_guess.xyz`) | omol | 4 | 0.04 | 0.01 | 0.97 | 0.52 |
| Water dimer (closest pair from `water_box_64.xyz`) | omol | 6 | 0.01 – 0.04 | ≤ 0.006 | 1.9 – 2.0 | 0.91 |
| Ba₂BiVO₆ 40-atom cubic cell (d⁰, nonmagnetic) | omat | 40 | 0.96 – 1.55 | 0.02 – 0.04 | 1.1 – 1.4 | 0.35 |
| Ba₂BiVO₆ 40-atom cubic cell | omat + pbe-d3bj | 40 | 1.35 – 1.46 | 0.03 – 0.04 | 1.1 – 1.3 | 0.34 – 0.39 |

The ranges span the two runs. TF32 differences are at most 0.04 meV/atom and
2 meV/Å, the scale of the fmax = 0.05 eV/Å convergence noise. Use
`high-precision` for Hessians, phonons and tight TS work. PBE-D3(BJ) lowers the
Ba₂BiVO₆ energy by 9.25 eV (−275.044 vs −265.792 eV, 0.23 eV/atom). That is
the D3 correction itself, not a discrepancy.

## Speed and memory (this GPU only)

The Ba₂BiVO₆ rows are `omat` without D3.

| | NH₃ (4) | Water dimer (6) | Ba₂BiVO₆ (40) | Ba₂BiVO₆ + D3 (40) |
|---|---|---|---|---|
| Checkpoint load (s), run 1 / run 2 | 11.9 / 10.3 | 10.0 / 9.1 | 7.4 / 33.8 | 10.2 / 41.8 |
| First evaluation (s) | 1.3 – 9.8 | 1.2 – 1.8 | 1.9 – 18 | 1.8 – 11 |
| Warm evaluation, `multigpu` (ms, median of 20) | 95 – 98 | 95 – 96 | 48 – 106 | 109 – 113 |
| Warm evaluation, `high-precision` (ms) | 95 – 103 | 100 – 108 | 106 – 108 | 118 |
| Peak GPU memory (MB, allocated by torch) | 115 | 118 | 543 | 547 |

- Initialization (worker start, checkpoint load and first evaluation) is
  measured apart from warm evaluations, which are timed inside the worker
  without the JSON round trip. Cold starts varied 3–10× between runs (first
  CUDA context, disk cache, a shared laptop), so treat them as ranges. The
  shared worker pays the load once per checkpoint and mode, not once per run.
- Warm cost is about 0.1 s per evaluation whether the system has 4 or 40 atoms,
  so at these sizes per-call overhead dominates. That overhead is graph
  construction and the un-compiled PyTorch backend, not the GPU arithmetic.
  These are not Nitro numbers.
- Peak memory is set by the merged model plus the graph (~0.5 GB for 40 atoms in
  a periodic cell), well inside the 8 GB card.

## Optimization and MD through the bridge

| Run | Result |
|---|---|
| NH₃ relax (FIRE, fmax 0.05) | converged in 17 steps, 1.8–2.1 s |
| Water dimer relax | converged in 142–144 steps, 16–17 s |
| Ba₂BiVO₆ relax (positions, fixed cell) | converged in 1 step (already at a minimum) |
| Water dimer Langevin MD, 300 K, 0.5 fs, 200 steps | completed; 110–118 ms/step; mean T 350 K (expected spread for a 6-atom system) |

## Rule and fallback checks (all pass)

- `d4rev` + PBE-D3 or r²SCAN-D3: rejected as double dispersion.
- `standard` with `omol`, `omc`, `odac` or `oc25` + PBE-D3: rejected as double
  dispersion.
- `standard` `oc20` (RPBE) + PBE-D3: rejected because the functional does not
  match.
- `d4rev` with task `omat`, and `r2scan` with task `omol`: rejected.
- `r2scan` with `magnetic=True`, with multiplicity 3, or with a NiO cell
  carrying initial moments: rejected before the worker is asked.
- Nitro requested on the CPU: effective `multigpu`, with the reason in the log.
- Composition change on a fast-path predictor: `Rebuilt the UMA predictor:
  composition changed`.

The regression tests in `tests/test_uma_backend.py` pass: 12 offline tests and
2 real-GPU tests (anion forces vs. finite differences, plus a charge/spin change;
PBE-D3 on Si).
