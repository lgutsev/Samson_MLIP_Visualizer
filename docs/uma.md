# UMA in the bridge: presets, Nitro, installation

UMA runs in its own Python environment, in a worker process
([uma_worker.py](../src/samson_mlip_visualizer/uma_worker.py)). SAMSON's Python
(numpy 1.24) never imports fairchem. All configuration and calculator construction
is in [uma_config.py](../src/samson_mlip_visualizer/uma_config.py), which imports
only the standard library at import time. The panel, the CLI, bridge jobs, and
later InterfaceForge share that one definition.

## Presets

| Preset | Checkpoint | Task | Dispersion |
|---|---|---|---|
| `standard` | `uma-s-1p2p1` | any: `omol omat oc20 odac omc oc25 oc22` (default `omol`) | `none`, or `pbe-d3bj` on a PBE task (`omat`, `oc22`) only |
| `d4rev` | `uma-s-1p2p1-ft-d4rev` | `omol` only | in the labels (ωB97M-D4rev); any added D3 is rejected |
| `r2scan` | `uma-s-1p2p1-ft-r2scan` | `omat` only | `r2scan-d3bj` (fairchem `DFTD3Calculator(functional="r2scan")`), or `none` |
| `custom` | any `.pt` / fairchem name | any | as `standard` |

The bridge checks each configuration before the worker starts (`UMAConfigError`,
a `ValueError`):

- Charge and multiplicity apply only to `omol` (`standard`, `d4rev`, `custom`).
- **No double dispersion.** D3 is refused on `d4rev` and on tasks whose labels
  already contain dispersion: `omol` (VV10), `omc`, `odac` (PBE+D3) and `oc25`
  (RPBE+D3). PBE-D3(BJ) is also refused on RPBE tasks (`oc20`).
- **`r2scan` is not spin-polarized.** It treats every system as nonmagnetic. The
  panel says so under the preset and every run logs it. `magnetic=True`, a
  multiplicity other than 1, or a structure with nonzero initial magnetic moments
  is refused. Cr/Mn/Fe/Co/Ni/... and single atoms only warn, because composition
  or multiplicity alone cannot show whether a system is magnetic. Do not use this
  preset for Fe clusters, antiferromagnetic NiO, spin ladders, or cohesive
  energies with magnetic isolated atoms. Use `standard` with `omat` for those.
- A fine-tuned checkpoint loaded by path under `custom` gets its own preset's
  rules.
- No isolated-atom references ship with the two fine-tunes, so single-atom
  energies from them may be unreliable.

D3 settings are recorded in provenance (`uma_d3_settings`). Both presets use BJ
damping, a 15 Å cutoff, C5 smoothing over the outer 20 % and no three-body term.
The parameters are a1, a2 (Bohr), s6 and s8:

| `dispersion` | a1 | a2 | s6 | s8 | Source |
|---|---|---|---|---|---|
| `pbe-d3bj` | 0.4289 | 4.4407 | 1.0 | 0.7875 | Grimme, Ehrlich, Goerigk 2011 |
| `r2scan-d3bj` | 0.49484001 | 5.73083694 | 1.0 | 0.78981345 | Ehlert *et al.* 2021 |

This is not equivalent to every VASP "PBE+D3" setup (IVDW = 11/12 and their
cutoffs and three-body options differ).

## Inference modes

| Requested | fairchem preset | On Linux, Ampere/Hopper/Blackwell | On this Windows laptop (no Triton) |
|---|---|---|---|
| `nitro` (default) | `default` | UMA Nitro for the three `uma-s-1p2p1` checkpoints; `multigpu` (TF32) otherwise | `multigpu` settings, `umas_fast_pytorch` backend, no `torch.compile` |
| `high-precision` | `high-precision` | FP32, merged MoLE + compile | FP32, merged MoLE, `umas_fast_pytorch`, no compile |

`turbo` was removed in fairchem 2.24. A saved or requested `turbo` maps to `nitro`
with a note.

Every run logs the requested and effective mode and the reason for each
fallback, for example:

```
UMA: UMA loaded on cuda: requested nitro, effective multigpu (tf32 on, backend
umas_fast_pytorch, compile off); fairchem: UMA Nitro's kernels need Triton, which
has no official build for this platform (Windows): Nitro cannot run | ...
```

The same appears in the result's provenance as `mlip_uma_inference_requested`,
`mlip_uma_inference_effective`, `mlip_uma_fallback`, `mlip_uma_tf32`,
`mlip_uma_version_*` and `mlip_uma_gpu`.

UMA's fast paths are specialised to one composition, task, charge and spin, and
fairchem falls back for good when one of them changes. The worker instead
rebuilds the predictor for the new system and logs `Rebuilt the UMA predictor:
composition changed`. Predictors stay loaded between runs: all calculators with
the same fairchem Python share one worker, which keeps at most two predictors.

## Installation (reproducible)

Validated on 2026-10-23: Windows 11, RTX 5070 Laptop GPU (sm 12.0, 8 GB), driver
591.44. The full freeze is [uma_nitro_freeze.txt](uma_nitro_freeze.txt).

```bash
D:/MLIP_Work_Folder/envs/mlip/python.exe -m venv D:/MLIP_Work_Folder/envs/uma_nitro
```

```bash
D:/MLIP_Work_Folder/envs/uma_nitro/Scripts/python.exe -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
```

```bash
D:/MLIP_Work_Folder/envs/uma_nitro/Scripts/python.exe -m pip install "fairchem-core==2.24.0" "truststore==0.10.4"
```

- The base interpreter is Python 3.12.14 (conda-forge). fairchem 2.24 needs
  Python 3.11–3.14 and `torch~=2.13.0`. It brought numpy 2.4.6, ase 3.29.0,
  e3nn 0.6.0, nvalchemi-toolkit-ops 0.5.0 and huggingface_hub 2.3.0.
- `truststore` makes downloads use the Windows certificate store. Without it,
  `requests` (certifi) fails behind this network's TLS inspection, which breaks
  the D3 parameter download and Hugging Face checkpoint downloads.
- The previous environment (`envs/mlip_gpu`, fairchem 2.23.0) is untouched.
- Point the panel's model field, `--backend uma` with `auto`, or
  `FAIRCHEM_PYTHON` at `D:/MLIP_Work_Folder/envs/uma_nitro/Scripts/python.exe`.

Checkpoints: `facebook/UMA` on Hugging Face is gated. Accept its licence, then
log in once (`hf auth login`, or set `HF_TOKEN` in the environment SAMSON starts
from). fairchem downloads `uma-s-1p2p1`, `uma-s-1p2p1-ft-d4rev` and
`uma-s-1p2p1-ft-r2scan` to the Hugging Face cache on first use. You can also put
a downloaded `.pt` in the checkpoint field. Weights and tokens are never
committed. The D3 table (1.8 MB) is generated once into
`~/.cache/nvalchemiops/dftd3_parameters.pt` from Grimme's official `dftd3.tgz`,
whose MD5 is checked.

### Nitro on Windows

Nitro's kernels import Triton, and its Blackwell path uses CuTe
(`nvidia-cutlass-dsl`, which fairchem installs only on Linux). Neither has an
official Windows build, so Nitro cannot run on this laptop. The worker therefore
selects fairchem's pure-PyTorch `umas_fast_pytorch` backend and turns compile
off, instead of letting fairchem crash on its automatic `umas_fast_gpu` choice.
Two ways to get real Nitro:

- **Linux** (LONI QB4 `gpu2`, WSL2 with CUDA): the same pins; there pip also
  installs `nvidia-cutlass-dsl` and `cuda-python`. The local `uma-s-1p2p1.pt`
  (md5 `3497615f…`) is on Nitro's checkpoint allowlist.
- The community `triton-windows` wheels may work. They are not an official
  dependency and were not installed or tested.

## Validation

```bash
"C:/Users/lguts/OneAngstrom/SAMSON-Application/11.0.1/Binaries/python.exe" scripts/uma_validate.py D:/MLIP_Work_Folder/envs/uma_nitro/Scripts/python.exe --device cuda --standard D:/MLIP_Foundational_Models/UMA/uma-s-1p2p1.pt --out D:/MLIP_Work_Folder/uma_nitro_validation
```

Regression tests: `tests/test_uma_backend.py`. The two real-model tests run with
`FAIRCHEM_PYTHON`, `UMA_CHECKPOINT` and `UMA_DEVICE=cuda` set. Results are in
[uma_benchmark.md](uma_benchmark.md).
