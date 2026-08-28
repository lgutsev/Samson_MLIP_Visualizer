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

Before it evaluates anything, the panel reads the element list the model reports
(MACE `z_table`, DeepMD `type_map`) and refuses structures containing an element
the model was not trained on. When the element list cannot be read it says so and
proceeds; that check is a convenience, not a guarantee.

## Check a model without SAMSON

The calculator and optimization layers do not need SAMSON. After installing the
package and a backend, a console script runs the same single-point and FIRE
relaxation on any ASE-readable structure file, which is the fastest way to
sanity-check a new model:

```bash
samson-mlip structure.cif model.model --backend mace
samson-mlip structure.xyz model.pb --backend deepmd --relax --fmax 0.03 -o relaxed.xyz
```

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
- Pseudo-atoms in the selected model are rejected, not silently evaluated.
- Model element coverage is checked when the backend exposes it.
- Atomic energies and forces; no stress or cell optimization.
- FIRE geometry optimization; no molecular dynamics yet.
- No automatic model download or model-specific preprocessing.
- Geometry updates from a relaxation are grouped into one SAMSON undo
  transaction. Saving the source document before long runs is still recommended.
- Every run logs its provenance (model SHA-256, device, dtype, package
  versions); the CLI also writes it into the output structure's metadata.

## Roadmap

This tool owns the optimization loop and uses SAMSON only as a structure source
and sink. It does **not** interoperate with SAMSON's own interactive simulation
(`Edit → Add simulator`, `Edit → Minimize`): that is SAMSON driving its own force
field frame by frame, and the two loops should not be run on one model at once.

Planned, roughly in priority order:

- Expose the MACE/DeepMD calculator as a native SAMSON interaction model so
  SAMSON's interactive simulator, minimizer, and atom dragging run on the MLIP.
  This is the real path to interactive use and needs a SAMSON SDK module rather
  than a Python package.
- Run energy/force calls off the UI thread. The calculator can move to a worker
  thread cleanly; the difficulty is that `sync_positions` and `SAMSON.holding`
  must stay on the main thread, so per-step live updates need a marshalling
  layer. Needs to be developed and tested inside SAMSON.
- Write results back as SAMSON data: total energy on the model, per-atom force
  vectors for arrow display. Blocked on confirming the property/visual API.
- Publish the relaxation as a SAMSON path (`node.type path`) so the trajectory
  can be scrubbed in the animation bar. Blocked on the path-creation API.
- Embed the run provenance in the SAMSON document itself, not just the log.

## License

MIT
