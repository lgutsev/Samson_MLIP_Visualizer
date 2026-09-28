"""Fine-tuning runs: on this desktop, or as a package for an HPC.

One :class:`TrainingSpec` describes a run; the same spec becomes a local run
(:func:`train_local`, seeds in parallel on the GPU) or a self-contained SLURM
package (:func:`write_training_package`) that the user copies to a cluster,
runs, and copies back. Nothing is ever submitted from here.

Three modes, two of them on top of a MACE foundation model:

- ``plain``: single-head fine-tuning on the reference labels. All foundation
  elements are kept (``--foundation_model_elements``); without it, mace-torch
  rebuilds the element table from the training data and the tuned model can no
  longer evaluate anything else (seen with HCN: H, C, N only).
- ``multihead``: the reference labels get their own head ("Default", the one
  the calculator uses) while a replay set keeps the foundation head
  ("pt_head") trained, which limits forgetting. mace-torch sets its own
  learning rate (1e-4) in this mode. The replay set is
  Materials Project data (``replay="mp"``: the MPtrj structures MACE-MP-0 was
  trained on, ~595 MB, downloaded by mace-torch into the MACE folder of
  :mod:`.paths` and randomly subsampled to ``replay_samples``) or a file of
  your own.
- ``scratch``: a new, small MACE (``architecture``) with fixed per-element
  energies (``e0s``) and no foundation model. For a Δ-learning correction on
  top of GFN-xTB (:mod:`.delta`): the residual is not an energy surface a
  foundation model has learned, so there is nothing to fine-tune from.

Every installed model gets a model card (:func:`install_models`), which the
tool records in the provenance of each run with it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MODES = ("plain", "multihead", "scratch")
# A small MACE for a Δ-learning correction: the residual is smooth and short-ranged.
SCRATCH_ARCHITECTURE = {"hidden_irreps": "32x0e+32x1o", "r_max": 5.0, "num_interactions": 2,
                        "correlation": 3, "max_ell": 3}
MP_REPLAY_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b/mp_traj_combined.xyz"
)
MP_REPLAY_CACHE_NAME = "mp_traj_combinedxyz"  # how mace-torch names it in its cache


@dataclass
class TrainingSpec:
    """One fine-tuning run (all seeds share it)."""

    name: str
    foundation: str  # a model file, or "small"/"medium"/"large"; "" for scratch
    train_file: str
    mode: str = "plain"
    replay: str | None = None  # multihead: "mp" or a path to an extxyz replay set
    replay_samples: int = 10000
    replay_weight: float = 1.0
    seeds: tuple[int, ...] = (1, 2, 3)
    epochs: int = 120
    lr: float = 0.005
    batch_size: int = 4
    energy_weight: float = 10.0
    forces_weight: float = 100.0
    stress_weight: float = 0.0
    valid_fraction: float = 0.1
    ema_decay: float = 0.99
    device: str = "cuda"
    dtype: str = "float64"
    keep_foundation_elements: bool = True
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"
    stress_key: str = "REF_stress"
    extra: tuple[str, ...] = ()  # more mace_run_train arguments, verbatim
    card: dict = field(default_factory=dict)  # what the data are (reference level, scope...)
    architecture: dict = field(default_factory=dict)  # scratch: overrides SCRATCH_ARCHITECTURE
    e0s: dict = field(default_factory=dict)  # scratch: eV per element, by atomic number

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if self.mode == "multihead" and not self.replay:
            raise ValueError("Multihead fine-tuning needs a replay set: 'mp' or a file")
        if self.mode != "multihead" and self.replay:
            raise ValueError("A replay set needs mode='multihead'")
        if self.mode == "scratch":
            if not self.e0s:
                raise ValueError("Training from scratch needs per-element energies (e0s)")
            self.e0s = {int(z): float(e) for z, e in self.e0s.items()}
            self.keep_foundation_elements = False
        elif self.architecture or self.e0s:
            raise ValueError("architecture and e0s are for mode='scratch'")
        self.seeds = tuple(int(seed) for seed in self.seeds)

    def model_name(self, seed: int) -> str:
        return f"{self.name}_seed{seed}"

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> TrainingSpec:
        data = dict(data)
        data["seeds"] = tuple(data["seeds"])
        data["extra"] = tuple(data.get("extra", ()))
        # JSON turns the atomic-number keys into strings
        data["e0s"] = {int(z): e for z, e in data.get("e0s", {}).items()}
        return cls(**data)


def train_arguments(spec: TrainingSpec, seed: int, workdir: str, *, train_file=None,
                    foundation=None) -> list[str]:
    """mace_run_train arguments for one seed (paths may be overridden for a package)."""
    if spec.mode == "scratch":
        from .delta import e0s_argument

        model = ["--model=MACE", f"--E0s={e0s_argument(spec.e0s)}"] + [
            f"--{key}={value}" for key, value in {**SCRATCH_ARCHITECTURE,
                                                  **spec.architecture}.items()]
    else:
        model = [f"--foundation_model={foundation or spec.foundation}", "--E0s=foundation"]
    arguments = [
        f"--name={spec.model_name(seed)}",
        *model,
        f"--train_file={train_file or spec.train_file}",
        f"--valid_fraction={spec.valid_fraction}",
        f"--energy_key={spec.energy_key}",
        f"--forces_key={spec.forces_key}",
        # "weighted" fits energies and forces only; stress needs its own loss
        f"--loss={'stress' if spec.stress_weight else 'weighted'}",
        f"--energy_weight={spec.energy_weight}",
        f"--forces_weight={spec.forces_weight}",
        f"--lr={spec.lr}",
        f"--batch_size={spec.batch_size}",
        f"--valid_batch_size={2 * spec.batch_size}",
        f"--max_num_epochs={spec.epochs}",
        "--ema",
        f"--ema_decay={spec.ema_decay}",
        f"--default_dtype={spec.dtype}",
        f"--device={spec.device}",
        f"--seed={seed}",
        f"--model_dir={workdir}",
        f"--work_dir={workdir}",
        f"--checkpoints_dir={workdir}/checkpoints",
        f"--results_dir={workdir}/results",
        f"--log_dir={workdir}/logs",
        "--save_cpu",
    ]
    if spec.stress_weight:
        arguments += [f"--stress_key={spec.stress_key}", f"--stress_weight={spec.stress_weight}",
                      "--compute_stress=True"]
    # Without it mace-torch rebuilds the element table from the data (plain: only
    # the training elements; multihead: those plus the replay sample's).
    if spec.keep_foundation_elements:
        arguments.append("--foundation_model_elements=True")
    if spec.mode == "plain":
        arguments.append("--multiheads_finetuning=False")
    elif spec.mode == "multihead":
        arguments += [
            "--multiheads_finetuning=True",
            f"--pt_train_file={spec.replay}",
            f"--num_samples_pt={spec.replay_samples}",
            f"--weight_pt_head={spec.replay_weight}",
        ]
    return arguments + list(spec.extra)


def mace_run_train() -> str:
    """The mace_run_train script of this Python (or on PATH)."""
    scripts = Path(sys.executable).parent / ("Scripts" if os.name == "nt" else "")
    local = scripts / ("mace_run_train.exe" if os.name == "nt" else "mace_run_train")
    return str(local) if local.exists() else (shutil.which("mace_run_train") or "mace_run_train")


@dataclass
class TrainingRun:
    seed: int
    model: Path | None
    seconds: float
    returncode: int
    log: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.model is not None and self.model.exists()


def train_local(spec: TrainingSpec, directory: str | Path, *, parallel: bool = True,
                cache_dir: str | Path | None = None) -> list[TrainingRun]:
    """Train every seed here (in parallel on one GPU: small models share it well).

    mace-torch reads and downloads (the ~595 MB Materials Project replay set)
    in :func:`.paths.downloads_dir` (the mirror folder when it is there);
    ``cache_dir`` overrides it with ``<cache_dir>/mace``.
    """
    from .paths import mace_environment

    environment = mace_environment()
    if cache_dir is not None:
        environment["XDG_CACHE_HOME"] = str(Path(cache_dir).resolve())
    directory = Path(directory).resolve()  # each seed runs inside its own folder
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "spec.json").write_text(json.dumps(spec.to_json(), indent=1), encoding="utf-8")
    # A fixed name: ASE guesses formats from file names (a name containing
    # "config" is read as DL_POLY), and the run should not depend on the source.
    train_file = directory / "train.extxyz"
    shutil.copyfile(spec.train_file, train_file)
    replay = spec.replay
    if replay not in (None, "mp"):
        replay = str(directory / "replay.extxyz")
        shutil.copyfile(spec.replay, replay)
    run_spec = TrainingSpec.from_json({**spec.to_json(), "train_file": str(train_file),
                                       "replay": replay})

    def run(seed: int) -> TrainingRun:
        workdir = directory / f"seed{seed}"
        workdir.mkdir(exist_ok=True)
        log = workdir / "train.log"
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as handle:
            code = subprocess.run(
                [mace_run_train(), *train_arguments(run_spec, seed, str(workdir))],
                stdout=handle, stderr=subprocess.STDOUT, cwd=workdir, env=environment,
            ).returncode
        model = workdir / f"{spec.model_name(seed)}.model"
        return TrainingRun(seed, model if model.exists() else None,
                           time.perf_counter() - start, code, log)

    workers = len(spec.seeds) if parallel else 1
    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(run, spec.seeds))


def final_errors(log_text: str) -> dict[str, dict[str, float]]:
    """The error table mace_run_train prints at the end: {set: {E, F}} in meV."""
    errors = {}
    for line in log_text.splitlines():
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) >= 3 and parts[0].startswith(("train", "valid", "test")):
            try:
                errors[parts[0]] = {"rmse_e_mev_per_atom": float(parts[1]),
                                    "rmse_f_mev_per_A": float(parts[2])}
            except ValueError:
                continue
    return errors


# --- HPC package ----------------------------------------------------------------------


def _slurm_script(spec: TrainingSpec, train_file: str, foundation: str, slurm) -> str:
    """Each seed runs inside runs/seed<N>/ (mace-torch writes its sampled replay set
    to the working directory), so package paths are given relative to it."""
    seeds = ",".join(str(seed) for seed in spec.seeds)

    def up(path: str) -> str:
        return path if path in ("mp", "small", "medium", "large") or Path(path).is_absolute() \
            else f"../../{path}"

    run_spec = TrainingSpec.from_json({**spec.to_json(), "replay": up(spec.replay)}) \
        if spec.replay else spec
    arguments = " \\\n    ".join(
        f'"{argument}"' for argument in train_arguments(
            run_spec, 0, ".", train_file=up(train_file), foundation=up(foundation)
        )
    ).replace("_seed0", "_seed${SEED}").replace("--seed=0", "--seed=${SEED}")
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name=ft-{spec.name}",
        f"#SBATCH --account={slurm.account}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --array={seeds}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={slurm.cpus}",
        f"#SBATCH --gres={slurm.gres}",
        f"#SBATCH --mem={slurm.memory_gb}G",
        f"#SBATCH --time={slurm.time}",
        "#SBATCH --output=logs/%x_%A_%a.out",
        "# Written by samson-mlip-visualizer. Replace every placeholder in angle brackets",
        "# (see README.md) before sbatch. One array task per seed.",
        # conda's activation scripts read unset variables: -u only after them
        "set -eo pipefail",
        # site default modules (QB4 loads the Intel compilers) can shadow the env's
        # libraries; the working QB4 launchers purge them first
        "module purge",
        *[f"module load {module}" for module in slurm.modules],
        slurm.activate,
        "set -u",
        'cd "$SLURM_SUBMIT_DIR"',
        "SEED=$SLURM_ARRAY_TASK_ID",
        'mkdir -p "runs/seed${SEED}"',
        'cd "runs/seed${SEED}"',
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        f"mace_run_train \\\n    {arguments}",
    ]) + "\n"


@dataclass
class GpuSlurmSettings:
    account: str = "<ACCOUNT>"
    partition: str = "<GPU_PARTITION>"
    modules: tuple[str, ...] = ("<CUDA_MODULE>",)
    activate: str = "<ACTIVATE_PYTHON_ENV_WITH_MACE>"  # e.g. source ~/venvs/mace/bin/activate
    gres: str = "gpu:1"
    cpus: int = 8
    memory_gb: int = 32
    time: str = "04:00:00"


def write_training_package(spec: TrainingSpec, directory: str | Path, *,
                           slurm: GpuSlurmSettings | None = None,
                           copy_foundation: bool = True) -> Path:
    """A folder to copy to a cluster: data, (the foundation model), a SLURM
    array script (one task per seed), a replay download script for multihead
    runs, and a README."""
    from .labeling import PLACEHOLDER

    slurm = slurm or GpuSlurmSettings()
    directory = Path(directory)
    (directory / "data").mkdir(parents=True, exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)
    (directory / "logs" / "README.txt").write_text("SLURM writes one log per seed here.\n")
    train_name = "data/train.extxyz"  # fixed: ASE guesses formats from file names
    shutil.copyfile(spec.train_file, directory / train_name)
    foundation = spec.foundation
    if copy_foundation and Path(spec.foundation).is_file():
        (directory / "foundation").mkdir(exist_ok=True)
        foundation = f"foundation/{Path(spec.foundation).name}"
        shutil.copyfile(spec.foundation, directory / foundation)
    replay = spec.replay
    if spec.mode == "multihead" and replay not in (None, "mp"):
        replay_name = "data/replay.extxyz"
        shutil.copyfile(replay, directory / replay_name)
        spec = TrainingSpec.from_json({**spec.to_json(), "replay": replay_name})
    script = _slurm_script(spec, train_name, foundation, slurm)
    (directory / "run_train.slurm").write_text(script, encoding="utf-8", newline="\n")
    if spec.replay == "mp":
        (directory / "download_mp_replay.sh").write_text("\n".join([
            "#!/bin/bash",
            "# Run on a LOGIN node (compute nodes often have no internet): puts the",
            "# Materials Project replay set where mace-torch looks for it.",
            "set -euo pipefail",
            'cache="${XDG_CACHE_HOME:-$HOME/.cache}/mace"',
            'mkdir -p "$cache"',
            f'target="$cache/{MP_REPLAY_CACHE_NAME}"',
            'if [ -s "$target" ]; then echo "already there: $target"; exit 0; fi',
            f'curl -L --fail -o "$target.part" "{MP_REPLAY_URL}"',
            'mv "$target.part" "$target"',
            'ls -lh "$target"',
        ]) + "\n", encoding="utf-8", newline="\n")
    placeholders = sorted(set(PLACEHOLDER.findall(script)))
    (directory / "spec.json").write_text(json.dumps({
        **spec.to_json(),
        "package_train_file": train_name,
        "package_foundation": foundation,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "placeholders_left": placeholders,
    }, indent=1), encoding="utf-8")
    todo = "\n".join(f"   - `{name}`" for name in placeholders) or "   - (none left)"
    replay_step = (
        "\n0. On a login node, run `bash download_mp_replay.sh` once (~595 MB; mace-torch\n"
        "   would otherwise try to download it on the compute node).\n"
        if spec.replay == "mp" else ""
    )
    (directory / "README.md").write_text(f"""# Fine-tuning package: {spec.name}

{spec.mode} fine-tuning of `{Path(spec.foundation).name}` on `{train_name}`,
{len(spec.seeds)} seed(s), {spec.epochs} epochs. Written by samson-mlip-visualizer;
nothing here was submitted.
{replay_step}
1. Copy this folder to the cluster. The Python environment needs mace-torch
   (this package was written for 0.3.16: `pip install mace-torch==0.3.16`).
2. In `run_train.slurm`, replace every placeholder:
{todo}
3. Submit: `sbatch run_train.slurm` (one array task per seed).
4. Copy back `runs/` (the `.model` files and logs) into this folder.
5. Install and label them, in SAMSON's Python on your desktop:

   ```python
   from samson_mlip_visualizer.training import install_models
   install_models("<this folder>")
   ```
""", encoding="utf-8")
    return directory


# --- installing -----------------------------------------------------------------------


def install_models(directory: str | Path, *, name: str | None = None,
                   destination: str | Path | None = None) -> list[Path]:
    """Copy the trained models of a local run or a returned package into
    ``<MACE folder>/finetuned/<name>/`` (:func:`.paths.finetuned_dir`), each
    with a model card, plus the spec and the training logs, and copy that folder
    to the mirror folder when one is set and reachable (not for ``destination``)."""
    from .paths import finetuned_dir, mirror

    directory = Path(directory)
    spec = TrainingSpec.from_json({
        key: value for key, value in json.loads((directory / "spec.json").read_text()).items()
        if key in TrainingSpec.__dataclass_fields__
    })
    target = Path(destination or finetuned_dir()) / (name or spec.name)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directory / "spec.json", target / "spec.json")
    installed = []
    for seed in spec.seeds:
        candidates = [
            *directory.glob(f"seed{seed}/{spec.model_name(seed)}.model"),
            *directory.glob(f"runs/seed{seed}/{spec.model_name(seed)}.model"),
        ]
        if not candidates:
            continue
        model = target / candidates[0].name
        shutil.copyfile(candidates[0], model)
        logs = sorted(candidates[0].parent.glob("*.log")) + sorted(
            candidates[0].parent.glob("logs/*.log"))
        errors = {}
        for log in logs:
            shutil.copyfile(log, target / f"{spec.model_name(seed)}_{log.name}")
            errors = final_errors(log.read_text(encoding="utf-8", errors="replace")) or errors
        if spec.mode == "scratch":
            origin = {"trained_from_scratch": {**SCRATCH_ARCHITECTURE, **spec.architecture},
                      "e0s_eV": spec.e0s}
        else:
            origin = {
                "fine_tuned_from": Path(spec.foundation).name,
                "fine_tuning": spec.mode + (f" with replay '{spec.replay}' "
                                            f"({spec.replay_samples} samples)"
                                            if spec.mode == "multihead" else ""),
                "keeps_foundation_elements": spec.keep_foundation_elements,
            }
        Path(str(model) + ".json").write_text(json.dumps({
            **origin,
            "heads": "Default (fine-tuned, used by default) + pt_head (replay)"
            if spec.mode == "multihead" else "Default",
            "trained_on": Path(spec.train_file).name,
            "seed": seed,
            "epochs": spec.epochs,
            "training_errors_meV": errors,
            **spec.card,
        }, indent=1), encoding="utf-8")
        installed.append(model)
    if destination is None and installed:
        mirror(target)
    return installed
