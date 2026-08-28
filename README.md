# SAMSON MLIP Visualizer

Run local [MACE](https://github.com/ACEsuit/mace) and
[DeepMD-kit](https://github.com/deepmodeling/deepmd-kit) models on structures in
**[SAMSON Connect](https://www.samson-connect.net/)** — the molecular modeling and
nanoscience platform by OneAngstrom — using ASE as the common calculator and
optimization layer.

Every reference to "SAMSON" in this repository means SAMSON Connect. It is
unrelated to any other product, company, or library that shares the name.

The first release provides:

- single-point energy and force evaluation;
- position-only FIRE relaxation with live geometry synchronization to SAMSON;
- periodic cell and per-axis PBC transfer from SAMSON to ASE;
- `FixAtoms` constraints derived from SAMSON fixed-atom flags;
- local MACE and DeepMD model files, with CPU or CUDA selection for MACE;
- no native SAMSON SDK build: the panel is an installable Python package.

> [!IMPORTANT]
> This is an alpha research tool. An ML potential is reliable only for elements,
> charge/spin states, structures, and thermodynamic conditions represented by its
> training data. The application does not make an incompatible model safe.

## Install in SAMSON

SAMSON includes a Python environment and PySide6. In **Python Console → Edit →
Manage packages**, install this repository as a local editable package, or run in
SAMSON's terminal:

```bash
python -m pip install -e C:\path\to\Samson_MLIP_Visualizer
```

Install the backend you intend to use in that same SAMSON Python environment:

```bash
# MACE
python -m pip install mace-torch

# DeepMD (choose the build/extras appropriate to your platform)
python -m pip install deepmd-kit
```

Do not install both GPU stacks merely because both backends are supported. Start
with a CPU build, verify a known structure, then follow the backend's current
CUDA installation guidance if acceleration is needed.

## Launch

Open `scripts/launch_in_samson.py` in SAMSON's Python code editor and run it, or
enter:

```python
from samson_mlip_visualizer.samson_app import show
show()
```

Then:

1. Open or construct a structure in SAMSON.
2. If the document contains multiple structural models, select exactly one
   complete model in Document View.
3. Mark immobile atoms with SAMSON's fixed-atom flag.
4. Choose MACE or DeepMD and select the trained model file.
5. Run **Single point** first. Check that the energy and forces are plausible.
6. Set the force threshold and maximum steps, then choose **Relax positions**.

The panel keeps the SAMSON interface responsive between optimization steps. Its
**Stop** button takes effect after the current energy/force call returns.

## Surface and passivant models

This workflow is compatible with passivated surface models when the potential is
compatible with the *entire* model:

- keep surface, adsorbate, passivants, and any counterions in one structural
  model;
- define the correct periodic cell and vacuum in SAMSON;
- set PBC only along genuinely periodic directions;
- fix bottom layers or passivants in SAMSON when they should not move;
- confirm that the model was trained for every chemical element and relevant
  environment in the structure;
- do not treat artificial H-like passivation, fractional nuclear charges, point
  charges, or implicit embedding as ordinary atoms unless the MLIP was explicitly
  trained with that representation.

The app deliberately refuses to evaluate a selected atom subset. A local MLIP
needs the whole atomic environment; evaluating only an adsorbate would produce a
number that is easy to misinterpret.

## Supported model interfaces

| Backend | ASE calculator | Typical local files | Device control |
|---|---|---|---|
| MACE | `mace.calculators.MACECalculator` | trained MACE checkpoint/model | `cpu` or `cuda` in the panel |
| DeepMD | `deepmd.calculator.DP` | `.pb`, `.pth`, `.json`, depending on backend | controlled by the installed DeepMD runtime |

The chemical species and cutoff compatibility are determined by the model, not
the file extension. Validate a new file against the code and structure used to
train or publish it.

## Developer setup

```bash
python -m pip install -e '.[test]'
pytest
ruff check .
```

The calculator and optimization layers are independent of SAMSON. Only
`samson_bridge.py` and `samson_app.py` touch its runtime API, which keeps most of
the project testable in a standard Python environment.

## Current scope

- One complete SAMSON structural model per run.
- Atomic energies and forces; no stress or cell optimization.
- FIRE geometry optimization; no molecular dynamics yet.
- No automatic model download or model-specific preprocessing.
- Geometry updates from a relaxation are grouped into one SAMSON undo
  transaction. Saving the source document before long runs is still recommended.

## License

MIT
