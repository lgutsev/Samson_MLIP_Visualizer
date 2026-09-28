"""Labeling packages to run on an HPC, and collectors for what comes back.

This tool never submits jobs or touches a cluster. :func:`write_label_package`
turns selected frames (``frames.extxyz`` + ``manifest.json`` from
:func:`.finetune.write_selection`) into a self-contained folder: one input per
frame, a SLURM array script with marked placeholders (account, partition,
modules), and a README with the exact steps and what to copy back. After the
jobs have run, :func:`collect_labels` parses the outputs, checks each one
(normal termination, SCF convergence, the geometry is the frame it was made
for), and writes ``labeled.extxyz`` for fine-tuning plus a report of what was
rejected and why.

Codes: Gaussian (``Force``) and ORCA (``EnGrad``), for molecules and clusters.
ORCA packages can also be energy-only (``job="energy"``), for methods without
analytic gradients such as DLPNO-CCSD(T): the labels then carry ``REF_energy``
and no forces.
Their energies are on another scale than the VASP-trained foundation models:
pass per-element offsets (:func:`.finetune.fit_element_offsets`) to the
collector to put ``REF_energy`` on the foundation scale; the raw energy is kept
as ``REF_energy_raw``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

from .finetune import ElementOffsets, Manifest, frame_checksum, labeled_structure
from .qm_export import gaussian_input, orca_input

CODES = ("gaussian", "orca")
# PBE by default: the level MACE-MP-0 was trained on, so corrections stay small.
DEFAULT_LEVEL = {"gaussian": "PBEPBE/def2TZVP", "orca": "PBE def2-TZVP def2/J"}
_SUFFIX = {"gaussian": ".gjf", "orca": ".inp"}
PLACEHOLDER = re.compile(r"<[A-Z_]+>")
PACKAGE_FILE = "package.json"
# Coordinates in Gaussian logs have 6 decimals; ORCA's .engrad has bohr to 1e-7.
_GEOMETRY_TOLERANCE = 1e-4  # Å


@dataclass
class SlurmSettings:
    """What goes into the array script. Placeholders (``<LIKE_THIS>``) are left
    for the user to fill in; the README lists every one that remains."""

    account: str = "<ACCOUNT>"
    partition: str = "<PARTITION>"
    modules: tuple[str, ...] = ("<MODULE>",)
    cpus: int = 8  # per frame
    memory_gb: int = 16  # per frame
    time: str = "02:00:00"
    max_parallel: int | None = 20  # array throttle (%N); None for no limit
    jobs_per_task: int = 1  # frames run side by side in one array task (whole-node clusters)
    setup: tuple[str, ...] = ()  # shell lines after the modules, e.g. a program's PATH


def frame_name(index: int) -> str:
    return f"frame_{index:04d}"


def _script(code: str, count: int, slurm: SlurmSettings, job: str = "force") -> str:
    """The array script. With ``jobs_per_task`` > 1, each array task runs that many
    frames side by side (``cpus`` cores and ``memory_gb`` each), for clusters that
    hand out whole nodes: e.g. four 16-core ORCA jobs on a 64-core node."""
    jobs = max(1, slurm.jobs_per_task)
    tasks = -(-count // jobs)
    throttle = f"%{slurm.max_parallel}" if slurm.max_parallel else ""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=label-{code}",
        f"#SBATCH --account={slurm.account}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --array=0-{tasks - 1}{throttle}",
        "#SBATCH --nodes=1",
        # ORCA runs its parallel steps through mpirun, which only starts as many
        # processes as SLURM allocated tasks; Gaussian runs threads in one task.
        *([f"#SBATCH --ntasks={slurm.cpus * jobs}", "#SBATCH --cpus-per-task=1"]
          if code == "orca" else
          ["#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={slurm.cpus * jobs}"]),
        f"#SBATCH --mem={slurm.memory_gb * jobs}G",
        f"#SBATCH --time={slurm.time}",
        "#SBATCH --output=logs/%x_%A_%a.out",
        "# Written by samson-mlip-visualizer. Replace every placeholder in angle brackets",
        "# (see README.md) before sbatch.",
        *([f"# Each array task runs {jobs} frames at once, {slurm.cpus} cores each."]
          if jobs > 1 else []),
        "set -euo pipefail",
        *[f"module load {module}" for module in slurm.modules],
        *slurm.setup,
        'cd "$SLURM_SUBMIT_DIR"',
        "mkdir -p outputs",
        "",
        "run_frame() {",
        '  local frame=$1',
    ]
    if code == "gaussian":
        lines += [
            '  local scratch="${TMPDIR:-/tmp}/g16_${SLURM_JOB_ID}_${frame}"',
            '  mkdir -p "$scratch"',
            '  GAUSS_SCRDIR="$scratch" g16 < "inputs/$frame.gjf" > "outputs/$frame.log"',
            '  rm -rf "$scratch"',
        ]
    else:
        lines += [
            "  # ORCA must be called with its full path for parallel runs.",
            '  local work="${TMPDIR:-/tmp}/orca_${SLURM_JOB_ID}_${frame}"',
            '  mkdir -p "$work"',
            '  cp "inputs/$frame.inp" "$work/"',
            '  (cd "$work" && "$ORCA_BIN" "$frame.inp" > "$frame.out")',
            '  cp "$work/$frame.out" outputs/',
            *(['  cp "$work/$frame.engrad" outputs/'] if job == "force" else []),
            '  rm -rf "$work"',
        ]
    lines += ["}", ""]
    if code == "orca":
        lines.append('ORCA_BIN=$(command -v orca)')
    if jobs == 1:
        lines.append('run_frame "$(printf "frame_%04d" "$SLURM_ARRAY_TASK_ID")"')
    else:
        lines += [
            f"for k in $(seq 0 {jobs - 1}); do",
            f"  index=$((SLURM_ARRAY_TASK_ID * {jobs} + k))",
            f'  if [ "$index" -lt {count} ]; then',
            '    run_frame "$(printf "frame_%04d" "$index")" &',
            "  fi",
            "done",
            "wait",
        ]
    return "\n".join(lines) + "\n"


def _readme(code: str, count: int, level: str, placeholders: list[str],
            job: str = "force") -> str:
    back = "`outputs/` (every `frame_NNNN.log`)" if code == "gaussian" else (
        "`outputs/` (every `frame_NNNN.out` and `frame_NNNN.engrad`)" if job == "force"
        else "`outputs/` (every `frame_NNNN.out`; energies only, no gradients)"
    )
    todo = (
        "\n".join(f"   - `{name}`" for name in placeholders)
        if placeholders
        else "   - (none left)"
    )
    frames = f"{count} frame" + ("" if count == 1 else "s")
    return f"""# Labeling package: {frames}, {code.capitalize()}, {level}

Written by samson-mlip-visualizer. Nothing here was submitted; run it yourself.

1. Copy this whole folder to the cluster.
2. In `run_{code}.slurm`, replace every placeholder:
{todo}
   and check the CPU count, memory, and time per frame. The inputs ask for
   the same number of cores and memory as the script.
3. Submit: `sbatch run_{code}.slurm` (one array task per frame).
4. When all tasks are done, copy back {back}
   into this folder on your desktop, keeping the folder layout.
5. Collect, in SAMSON's Python on your desktop:

   ```python
   from samson_mlip_visualizer.labeling import collect_labels
   collect_labels("<this folder>")
   ```

   It checks every output and writes `labeled.extxyz` and `collect_report.json`.

`frames.extxyz` and `manifest.json` are the frames the inputs were made from;
keep them with the outputs, the collector checks each geometry against them.
"""


def write_label_package(
    directory: str | Path,
    frames_path: str | Path,
    manifest_path: str | Path,
    *,
    code: str,
    level: str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    slurm: SlurmSettings | None = None,
    job: str = "force",
) -> Path:
    """A labeling package for ``code`` ('gaussian' or 'orca') in ``directory``.
    ``job="energy"`` (ORCA only) writes single points without gradients."""
    from ase.io import read

    if code not in CODES:
        raise ValueError(f"code must be one of {', '.join(CODES)}")
    if job not in ("force", "energy") or (job == "energy" and code != "orca"):
        raise ValueError("job must be 'force', or 'energy' with ORCA")
    slurm = slurm or SlurmSettings()
    frames = read(frames_path, ":")
    manifest = Manifest.load(manifest_path)
    changed = manifest.verify(frames)
    if changed:
        raise ValueError(f"frames.extxyz does not match manifest.json at frames {changed}")
    if any(frame.pbc.any() for frame in frames):
        raise ValueError(f"{code} labels molecules and clusters; periodic frames need VASP")
    directory = Path(directory)
    (directory / "inputs").mkdir(parents=True, exist_ok=True)
    # SLURM does not create the --output directory; a job without it fails at start.
    (directory / "logs").mkdir(exist_ok=True)
    (directory / "logs" / "README.txt").write_text("SLURM writes one log per frame here.\n")
    level = level or DEFAULT_LEVEL[code]
    options = {
        "job": job,
        "level": level,
        "charge": charge,
        "multiplicity": multiplicity,
        "nproc": slurm.cpus,
        "memory_gb": max(1, int(slurm.memory_gb * 0.85)),  # headroom for the program itself
    }
    for index, frame in enumerate(frames):
        name = frame_name(index)
        if code == "gaussian":
            text = gaussian_input([frame], title=f"{name} for fine-tuning", **options)
        else:
            text, _ = orca_input([frame], stem=name, **options)
        (directory / "inputs" / f"{name}{_SUFFIX[code]}").write_text(text, encoding="utf-8")
    shutil.copyfile(frames_path, directory / "frames.extxyz")
    shutil.copyfile(manifest_path, directory / "manifest.json")
    script = _script(code, len(frames), slurm, job)
    (directory / f"run_{code}.slurm").write_text(script, encoding="utf-8", newline="\n")
    placeholders = sorted(set(PLACEHOLDER.findall(script)))
    (directory / "README.md").write_text(
        _readme(code, len(frames), level, placeholders, job), encoding="utf-8")
    (directory / PACKAGE_FILE).write_text(json.dumps({
        "code": code,
        "job": job,
        "level": level,
        "charge": charge,
        "multiplicity": multiplicity,
        "frames": len(frames),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "placeholders_left": placeholders,
    }, indent=1), encoding="utf-8")
    return directory


# --- parsers --------------------------------------------------------------------------


class OutputError(ValueError):
    """An output that cannot be used as a label (with the reason)."""


def parse_gaussian_log(text: str) -> dict:
    """Energy (Hartree), forces (Hartree/bohr), the input-orientation geometry
    (Å), and the version from a Gaussian ``Force`` log. SCF (HF/DFT) energies
    only: post-HF total energies are not read."""
    if "Normal termination of Gaussian" not in text:
        raise OutputError("no 'Normal termination' line (the job failed or did not finish)")
    if "Convergence failure" in text:
        raise OutputError("SCF convergence failure")
    energies = re.findall(r"SCF Done:\s+E\(\S+\)\s+=\s+(-?\d+\.\d+)", text)
    if not energies:
        raise OutputError("no 'SCF Done' energy")
    blocks = text.split("Input orientation:")
    if len(blocks) < 2:
        raise OutputError("no 'Input orientation' geometry (run with NoSymm)")
    positions = []
    for line in blocks[-1].splitlines()[5:]:
        if line.strip().startswith("---"):
            break
        parts = line.split()
        positions.append([float(value) for value in parts[3:6]])
    match = re.search(
        r"Forces \(Hartrees/Bohr\)\s*\n.*\n\s*-+\s*\n(.*?)\n\s*-+", text, flags=re.DOTALL
    )
    if not match:
        raise OutputError("no 'Forces (Hartrees/Bohr)' block (run with Force)")
    forces = [[float(value) for value in line.split()[2:5]] for line in match.group(1).splitlines()]
    version = re.search(r"Gaussian (\d+):\s+(\S+)", text)
    return {
        "energy_hartree": float(energies[-1]),
        "forces_hartree_per_bohr": np.array(forces),
        "positions_angstrom": np.array(positions),
        "version": f"Gaussian {version.group(1)} {version.group(2)}" if version else "Gaussian",
    }


def parse_orca(out_text: str, engrad_text: str) -> dict:
    """Energy (Hartree), gradient (Hartree/bohr), geometry (Å), and version from
    an ORCA ``EnGrad`` run: its ``.out`` and ``.engrad`` files."""
    if "ORCA TERMINATED NORMALLY" not in out_text:
        raise OutputError("no 'ORCA TERMINATED NORMALLY' line (the job failed or did not finish)")
    if "SCF NOT CONVERGED" in out_text:
        raise OutputError("SCF not converged")
    values = [
        line.strip() for line in engrad_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    natoms = int(values[0])
    energy = float(values[1])
    gradient = np.array([float(v) for v in values[2 : 2 + 3 * natoms]]).reshape(natoms, 3)
    coordinates = np.array(
        [[float(v) for v in line.split()[1:4]] for line in values[2 + 3 * natoms : 2 + 4 * natoms]]
    )
    final = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out_text)
    if final and abs(float(final[-1]) - energy) > 1e-6:
        raise OutputError(".engrad and .out disagree on the energy (files from different runs?)")
    version = re.search(r"Program Version (\S+)", out_text)
    return {
        "energy_hartree": energy,
        "forces_hartree_per_bohr": -gradient,
        "positions_angstrom": coordinates * Bohr,
        "version": f"ORCA {version.group(1)}" if version else "ORCA",
    }


def parse_orca_energy(out_text: str) -> dict:
    """Energy (Hartree), geometry (Å), and version from an ORCA single point's
    ``.out`` (no gradient): the last FINAL SINGLE POINT ENERGY, which for a
    correlated method such as DLPNO-CCSD(T) is the correlated total energy."""
    if "ORCA TERMINATED NORMALLY" not in out_text:
        raise OutputError("no 'ORCA TERMINATED NORMALLY' line (the job failed or did not finish)")
    if "SCF NOT CONVERGED" in out_text:
        raise OutputError("SCF not converged")
    final = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out_text)
    if not final:
        raise OutputError("no 'FINAL SINGLE POINT ENERGY'")
    blocks = out_text.split("CARTESIAN COORDINATES (ANGSTROEM)")
    if len(blocks) < 2:
        raise OutputError("no 'CARTESIAN COORDINATES (ANGSTROEM)' block")
    positions = []
    for line in blocks[-1].splitlines()[2:]:
        parts = line.split()
        if len(parts) != 4:
            break
        positions.append([float(value) for value in parts[1:4]])
    version = re.search(r"Program Version (\S+)", out_text)
    return {
        "energy_hartree": float(final[-1]),
        "positions_angstrom": np.array(positions),
        "version": f"ORCA {version.group(1)}" if version else "ORCA",
    }


# --- collecting ---------------------------------------------------------------------


@dataclass
class CollectResult:
    labeled: list[Atoms]
    rejected: dict[int, str] = field(default_factory=dict)
    code: str = ""
    level: str = ""

    def report(self) -> dict:
        return {
            "code": self.code,
            "level": self.level,
            "labeled": len(self.labeled),
            "rejected": {frame_name(index): reason for index, reason in self.rejected.items()},
        }


def _parse(code: str, outputs: Path, name: str, job: str = "force") -> dict:
    if job == "energy":
        out = outputs / f"{name}.out"
        if not out.is_file():
            raise OutputError("missing .out")
        return parse_orca_energy(out.read_text(encoding="utf-8", errors="replace"))
    if code == "gaussian":
        log = outputs / f"{name}.log"
        if not log.is_file():
            raise OutputError("missing output")
        return parse_gaussian_log(log.read_text(encoding="utf-8", errors="replace"))
    out, engrad = outputs / f"{name}.out", outputs / f"{name}.engrad"
    if not (out.is_file() and engrad.is_file()):
        raise OutputError("missing .out or .engrad")
    return parse_orca(out.read_text(encoding="utf-8", errors="replace"),
                      engrad.read_text(encoding="utf-8", errors="replace"))


def collect_labels(
    directory: str | Path,
    *,
    offsets: ElementOffsets | None = None,
    write: bool = True,
) -> CollectResult:
    """Parse and check every output of a labeling package; write
    ``labeled.extxyz`` and ``collect_report.json`` (with ``write``)."""
    from ase.io import read
    from ase.io import write as write_frames

    directory = Path(directory)
    package = json.loads((directory / PACKAGE_FILE).read_text(encoding="utf-8"))
    code = package["code"]
    frames = read(directory / "frames.extxyz", ":")
    manifest = Manifest.load(directory / "manifest.json")
    changed = set(manifest.verify(frames))
    result = CollectResult(labeled=[], code=code, level=package["level"])
    for entry in manifest.frames:
        index, name = entry.index, frame_name(entry.index)
        if index in changed:
            result.rejected[index] = "frames.extxyz no longer matches the manifest"
            continue
        frame = frames[index]
        try:
            parsed = _parse(code, directory / "outputs", name, package.get("job", "force"))
            if parsed["positions_angstrom"].shape != frame.positions.shape:
                raise OutputError("the output has a different number of atoms")
            shift = np.abs(parsed["positions_angstrom"] - frame.positions).max()
            if shift > _GEOMETRY_TOLERANCE:
                raise OutputError(f"the output geometry differs from the frame by {shift:.2g} Å")
        except OutputError as exc:
            result.rejected[index] = str(exc)
            continue
        forces = parsed.get("forces_hartree_per_bohr")
        result.labeled.append(labeled_structure(
            frame,
            parsed["energy_hartree"] * Hartree,
            None if forces is None else forces * Hartree / Bohr,
            offsets=offsets,
            tag=entry.reason,
            meta={
                "code": parsed["version"],
                "level": package["level"],
                "charge": package["charge"],
                "multiplicity": package["multiplicity"],
                "frame": index,
                "source": entry.source,
                "checksum": frame_checksum(frame),
            },
        ))
    if write:
        if result.labeled:
            write_frames(directory / "labeled.extxyz", result.labeled)
        report = result.report()
        if offsets is not None:
            report["offsets"] = asdict(offsets)
        (directory / "collect_report.json").write_text(json.dumps(report, indent=1),
                                                       encoding="utf-8")
    return result
