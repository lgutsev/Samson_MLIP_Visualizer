"""VASP labeling packages for periodic frames: several levels on every frame.

The companion of :mod:`.labeling` (Gaussian, ORCA) for crystals. One package
labels every frame at each of ``levels``: by default PBE+U, the level
MACE-MP-0 was trained on (Materials Project settings), and HSE06 on the same
frame, restarted from the PBE+U wavefunction. A Δ-learning correction
(:mod:`.delta`) needs both numbers on exactly the same footing, so the levels
share one KPOINTS file, ENCUT, PREC, and LREAL, and symmetry is off in both.

- :func:`write_vasp_package`: ``frames.extxyz`` + ``manifest.json`` (from
  :func:`.finetune.write_selection`) -> a folder with, per frame, POSCAR,
  KPOINTS, one INCAR per level, and the POTCAR names to concatenate on the
  cluster (POTCARs are licensed, so none are copied); a SLURM array script with
  placeholders; and a README. Nothing is ever submitted from here.
- :func:`collect_vasp_labels`: checks every run (finished, electronic loop
  converged, geometry and cell of its frame) and writes ``labeled.extxyz``
  with ``<LEVEL>_energy`` / ``_forces`` / ``_stress`` for each level. A frame is
  kept only when every level of it is good.

POTCARs and U values follow pymatgen's ``MPRelaxSet.yaml`` (the legacy "PBE"
POTCAR set, U on the 3d/4d/5d metals listed there, for oxides and fluorides
only), which is what MACE-MP-0's training data used. Energies are the
force-consistent (free) energies, with Gaussian smearing of 0.05 eV.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms

from .finetune import Manifest, frame_checksum
from .labeling import PACKAGE_FILE, PLACEHOLDER, OutputError, frame_name

# pymatgen MPRelaxSet.yaml (POTCAR_FUNCTIONAL: PBE), retrieved 2026-09-27.
MP_POTCARS = {
    "Ac": "Ac", "Ag": "Ag", "Al": "Al", "Ar": "Ar", "As": "As", "Au": "Au", "B": "B",
    "Ba": "Ba_sv", "Be": "Be_sv", "Bi": "Bi", "Br": "Br", "C": "C", "Ca": "Ca_sv", "Cd": "Cd",
    "Ce": "Ce", "Cl": "Cl", "Co": "Co", "Cr": "Cr_pv", "Cs": "Cs_sv", "Cu": "Cu_pv",
    "Dy": "Dy_3", "Er": "Er_3", "Eu": "Eu", "F": "F", "Fe": "Fe_pv", "Ga": "Ga_d", "Gd": "Gd",
    "Ge": "Ge_d", "H": "H", "He": "He", "Hf": "Hf_pv", "Hg": "Hg", "Ho": "Ho_3", "I": "I",
    "In": "In_d", "Ir": "Ir", "K": "K_sv", "Kr": "Kr", "La": "La", "Li": "Li_sv", "Lu": "Lu_3",
    "Mg": "Mg_pv", "Mn": "Mn_pv", "Mo": "Mo_pv", "N": "N", "Na": "Na_pv", "Nb": "Nb_pv",
    "Nd": "Nd_3", "Ne": "Ne", "Ni": "Ni_pv", "Np": "Np", "O": "O", "Os": "Os_pv", "P": "P",
    "Pa": "Pa", "Pb": "Pb_d", "Pd": "Pd", "Pm": "Pm_3", "Pr": "Pr_3", "Pt": "Pt", "Pu": "Pu",
    "Rb": "Rb_sv", "Re": "Re_pv", "Rh": "Rh_pv", "Ru": "Ru_pv", "S": "S", "Sb": "Sb",
    "Sc": "Sc_sv", "Se": "Se", "Si": "Si", "Sm": "Sm_3", "Sn": "Sn_d", "Sr": "Sr_sv",
    "Ta": "Ta_pv", "Tb": "Tb_3", "Tc": "Tc_pv", "Te": "Te", "Th": "Th", "Ti": "Ti_pv",
    "Tl": "Tl_d", "Tm": "Tm_3", "U": "U", "V": "V_pv", "W": "W_pv", "Xe": "Xe", "Y": "Y_sv",
    "Yb": "Yb_2", "Zn": "Zn", "Zr": "Zr_sv",
}
# Materials Project U (eV) on d states, applied only when O or F is present.
MP_U = {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38, "Ni": 6.2, "V": 3.25,
        "W": 6.2}

# What every level shares: a single point with forces and stress on a fixed cell.
COMMON_INCAR = {
    "ENCUT": 520, "PREC": "Accurate", "LASPH": ".TRUE.", "LREAL": ".FALSE.",
    "EDIFF": "1E-6", "NELM": 200, "ISMEAR": 0, "SIGMA": 0.05,
    "NSW": 0, "IBRION": -1, "ISIF": 2, "ISYM": 0, "LCHARG": ".FALSE.", "LORBIT": 0,
}
LEVELS = {
    "pbe_u": {"GGA": "PE", "ALGO": "Normal", "LWAVE": ".TRUE."},  # its WAVECAR starts HSE06
    "hse06": {"GGA": "PE", "LHFCALC": ".TRUE.", "HFSCREEN": 0.2, "AEXX": 0.25,
              "PRECFOCK": "Normal", "ALGO": "Damped", "TIME": 0.4, "ISTART": 1,
              "LWAVE": ".FALSE."},
}
LEVEL_NAMES = {"pbe_u": "PBE+U (Materials Project U)", "hse06": "HSE06"}
LABEL_PREFIX = {"pbe_u": "PBEU", "hse06": "HSE06"}
_GEOMETRY_TOLERANCE = 1e-4  # Å; vasprun.xml keeps fractional coordinates to 8 decimals


def potcar_names(symbols) -> list[str]:
    """The Materials Project POTCAR for each species, in POSCAR order."""
    missing = [s for s in symbols if s not in MP_POTCARS]
    if missing:
        raise ValueError(f"No Materials Project POTCAR for {', '.join(missing)}")
    return [MP_POTCARS[s] for s in symbols]


def species_order(atoms: Atoms) -> list[str]:
    """Species in order of first appearance (the POSCAR species line)."""
    return list(dict.fromkeys(atoms.get_chemical_symbols()))


def kpoint_mesh(atoms: Atoms, spacing: float) -> tuple[int, int, int]:
    """A Γ-centered mesh with at most ``spacing`` (Å⁻¹, 2π included, as VASP's
    KSPACING) between k-points along each reciprocal vector."""
    lengths = 2 * np.pi * np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    return tuple(max(1, math.ceil(length / spacing - 1e-9)) for length in lengths)


def incar(level: str, atoms: Atoms, *, magmom: dict[str, float] | None = None,
          extra: dict | None = None) -> str:
    """The INCAR text of ``level`` for ``atoms``: shared settings, the level's own,
    Materials Project U (PBE+U only, when O or F is present), and spin polarization
    with the given initial moments (``magmom``: μB per element; none = non-magnetic)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}")
    tags = {**COMMON_INCAR, **LEVELS[level]}
    species = species_order(atoms)
    symbols = atoms.get_chemical_symbols()
    if level == "pbe_u" and {"O", "F"} & set(species) and set(MP_U) & set(species):
        tags.update(LDAU=".TRUE.", LDAUTYPE=2, LMAXMIX=4,
                    LDAUL=" ".join("2" if s in MP_U else "-1" for s in species),
                    LDAUU=" ".join(f"{MP_U.get(s, 0.0):g}" for s in species),
                    LDAUJ=" ".join("0" for _ in species))
    if magmom:
        tags.update(ISPIN=2, MAGMOM=" ".join(f"{magmom.get(s, 0.0):g}" for s in symbols))
    else:
        tags["ISPIN"] = 1
    tags.update(extra or {})
    lines = [f"SYSTEM = {atoms.get_chemical_formula()} {LEVEL_NAMES[level]} single point",
             "# Written by samson-mlip-visualizer; the other levels of this package share",
             "# ENCUT, PREC, LREAL, ISYM, and KPOINTS, so their differences are the method's."]
    lines += [f"{key} = {value}" for key, value in tags.items()]
    return "\n".join(lines) + "\n"


@dataclass
class VaspSlurmSettings:
    """What goes into the array script (one task per frame, all levels in it).
    Placeholders (``<LIKE_THIS>``) are left for the user; the README lists them."""

    account: str = "<ACCOUNT>"
    partition: str = "<PARTITION>"
    modules: tuple[str, ...] = ("<VASP_MODULE>",)
    run: str = "<VASP_COMMAND>"  # e.g. srun vasp_std, or mpirun -np $SLURM_NTASKS vasp_std
    potcar_dir: str = "<POTPAW_PBE_DIR>"  # the folder holding Ba_sv/POTCAR, O/POTCAR, ...
    nodes: int = 1
    tasks_per_node: int = 48
    time: str = "08:00:00"
    max_parallel: int | None = 10
    setup: tuple[str, ...] = ()  # shell lines after the modules (e.g. a container's env)
    # POTCARs: concatenated from ``potcar_dir`` by the names in POTCAR.names (the
    # Materials Project choices), or, with ``potcar_command``, made by that command in
    # the run folder (e.g. a site's POTCAR_gen). Either way the POTCARs used are
    # recorded; with ``potcar_command`` and ``potcar_strict`` a run whose POTCARs
    # differ from POTCAR.names stops instead of computing.
    potcar_command: str | None = None  # e.g. /home/<user>/bin/POTCAR_gen
    potcar_strict: bool = True


def _script(levels: tuple[str, ...], count: int, slurm: VaspSlurmSettings) -> str:
    throttle = f"%{slurm.max_parallel}" if slurm.max_parallel else ""
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=label-vasp",
        f"#SBATCH --account={slurm.account}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --array=0-{count - 1}{throttle}",
        f"#SBATCH --nodes={slurm.nodes}",
        f"#SBATCH --ntasks-per-node={slurm.tasks_per_node}",
        f"#SBATCH --time={slurm.time}",
        "#SBATCH --output=logs/%x_%A_%a.out",
        "# Written by samson-mlip-visualizer. Replace every placeholder in angle brackets",
        "# (see README.md) before sbatch. One array task per frame; its levels run in order,",
        f"# {' -> '.join(levels)}, each later one starting from the previous WAVECAR.",
        "set -euo pipefail",
        # a clean environment for the container module, as in the working launchers
        *(["module purge"] if slurm.modules else []),
        *[f"module load {module}" for module in slurm.modules],
        *slurm.setup,
        'cd "$SLURM_SUBMIT_DIR"',
        'frame=$(printf "frame_%04d" "$SLURM_ARRAY_TASK_ID")',
        'work="runs/$frame"',
        'mkdir -p "$work" "outputs/$frame"',
        'potcar="$work/POTCAR"',
    ]
    if slurm.potcar_command:
        lines += [
            f"# POTCAR from {slurm.potcar_command}, run in the frame's folder with its POSCAR",
            'cp "inputs/$frame/POSCAR" "$work/POSCAR"',
            f'(cd "$work" && {slurm.potcar_command})',
        ]
    else:
        lines += [
            f'POTCARS="{slurm.potcar_dir}"',
            ': > "$potcar"',
            'for name in $(cat "inputs/$frame/POTCAR.names"); do',
            '  cat "$POTCARS/$name/POTCAR" >> "$potcar"',
            "done",
        ]
    lines += [
        "# the POTCARs actually used, one per species (TITEL lines)",
        'grep "TITEL" "$potcar" | awk \'{print $4}\' > "outputs/$frame/POTCAR.used"',
    ]
    if slurm.potcar_command and slurm.potcar_strict:
        lines += [
            'if [ "$(tr -s \' \\n\' \' \' < "outputs/$frame/POTCAR.used" | xargs)" != \\',
            '     "$(xargs < "inputs/$frame/POTCAR.names")" ]; then',
            '  echo "POTCARs differ from POTCAR.names: $(xargs < "outputs/$frame/POTCAR.used")'
            ' vs $(xargs < "inputs/$frame/POTCAR.names"); stopping" >&2',
            "  exit 1",
            "fi",
        ]
    lines += ["previous="]
    for level in levels:
        lines += [
            f'mkdir -p "$work/{level}"',
            f'cp "inputs/$frame/POSCAR" "inputs/$frame/KPOINTS" "$potcar" "$work/{level}/"',
            f'cp "inputs/$frame/INCAR.{level}" "$work/{level}/INCAR"',
            'if [ -n "$previous" ] && [ -s "$work/$previous/WAVECAR" ]; then',
            f'  cp "$work/$previous/WAVECAR" "$work/{level}/"',
            "fi",
            f'(cd "$work/{level}" && {slurm.run} > vasp.out 2>&1) || true',
            f'mkdir -p "outputs/$frame/{level}"',
            "for f in vasprun.xml OUTCAR OSZICAR vasp.out; do",
            f'  if [ -f "$work/{level}/$f" ]; then',
            f'    cp "$work/{level}/$f" "outputs/$frame/{level}/"',
            "  fi",
            "done",
            f"previous={level}",
        ]
    lines += ['rm -f "$work"/*/WAVECAR "$work"/*/CHG "$work"/*/CHGCAR']
    return "\n".join(lines) + "\n"


def _readme(levels, count, mesh_note, placeholders) -> str:
    todo = "\n".join(f"   - `{name}`" for name in placeholders) or "   - (none left)"
    names = ", ".join(LEVEL_NAMES[level] for level in levels)
    return f"""# VASP labeling package: {count} frame{'s' if count != 1 else ''}, {names}

Written by samson-mlip-visualizer. Nothing here was submitted; run it yourself.
Each frame is computed at every level ({', '.join(levels)}) with the same
KPOINTS, ENCUT, PREC, LREAL, and ISYM, so the differences between the levels
are the method's alone. {mesh_note}

1. Copy this whole folder to the cluster.
2. In `run_vasp.slurm`, replace every placeholder:
{todo}
   `<POTPAW_PBE_DIR>` is the folder of the legacy PBE POTCARs (the set the
   Materials Project uses), holding `Ba_sv/POTCAR`, `O/POTCAR`, and so on;
   `inputs/frame_NNNN/POTCAR.names` lists which ones each frame needs.
   Check nodes, tasks, and time: hybrid functionals cost far more than PBE+U.
3. Submit: `sbatch run_vasp.slurm` (one array task per frame).
4. Copy back `outputs/` (vasprun.xml, OUTCAR, OSZICAR per frame and level)
   into this folder on your desktop, keeping the layout. `runs/` stays on the
   cluster (the WAVECARs are deleted at the end of each task).
5. Collect, in SAMSON's Python on your desktop:

   ```python
   from samson_mlip_visualizer.vasp_labeling import collect_vasp_labels
   collect_vasp_labels("<this folder>")
   ```

   It checks every run and writes `labeled.extxyz` (every level's energy,
   forces, and stress) and `collect_report.json`.
"""


def write_poscar(path: str | Path, atoms: Atoms) -> None:
    """A POSCAR with Unix line endings: VASP and the job scripts run on Linux, and a
    CRLF file written on Windows breaks both."""
    from io import StringIO

    from ase.io import write

    text = StringIO()
    write(text, atoms, format="vasp", direct=True, sort=False)
    Path(path).write_text(text.getvalue(), encoding="utf-8", newline="\n")


def write_vasp_package(
    directory: str | Path,
    frames_path: str | Path,
    manifest_path: str | Path,
    *,
    levels: tuple[str, ...] = ("pbe_u", "hse06"),
    kspacing: float = 0.25,
    magmom: dict[str, float] | None = None,
    incar_extra: dict[str, dict] | None = None,
    slurm: VaspSlurmSettings | None = None,
) -> Path:
    """A VASP labeling package: every frame at every level (see the module doc).

    ``kspacing`` (Å⁻¹) sets each frame's Γ-centered mesh; ``magmom`` (μB per
    element) makes every level spin-polarized; ``incar_extra`` adds or overrides
    tags per level, e.g. ``{"hse06": {"NCORE": 4, "KPAR": 2}}``.
    """
    from ase.io import read

    unknown = [level for level in levels if level not in LEVELS]
    if unknown or not levels:
        raise ValueError(f"levels must be some of {', '.join(LEVELS)}")
    slurm = slurm or VaspSlurmSettings()
    frames = read(frames_path, ":")
    manifest = Manifest.load(manifest_path)
    changed = manifest.verify(frames)
    if changed:
        raise ValueError(f"frames.extxyz does not match manifest.json at frames {changed}")
    if not all(frame.pbc.all() for frame in frames):
        raise ValueError("VASP packages are for periodic frames (pbc on all three axes)")
    directory = Path(directory)
    (directory / "inputs").mkdir(parents=True, exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)
    (directory / "logs" / "README.txt").write_text("SLURM writes one log per frame here.\n",
                                                    newline="\n")
    meshes = []
    for index, frame in enumerate(frames):
        folder = directory / "inputs" / frame_name(index)
        folder.mkdir(exist_ok=True)
        ordered = _sorted_by_species(frame)
        write_poscar(folder / "POSCAR", ordered)
        mesh = kpoint_mesh(ordered, kspacing)
        meshes.append(mesh)
        (folder / "KPOINTS").write_text(
            f"Gamma-centered, spacing {kspacing} 1/A\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n"
            "0 0 0\n", newline="\n")
        (folder / "POTCAR.names").write_text(" ".join(potcar_names(species_order(ordered)))
                                             + "\n", newline="\n")
        for level in levels:
            (folder / f"INCAR.{level}").write_text(
                incar(level, ordered, magmom=magmom, extra=(incar_extra or {}).get(level)),
                newline="\n")
    shutil.copyfile(frames_path, directory / "frames.extxyz")
    shutil.copyfile(manifest_path, directory / "manifest.json")
    script = _script(tuple(levels), len(frames), slurm)
    (directory / "run_vasp.slurm").write_text(script, encoding="utf-8", newline="\n")
    placeholders = sorted(set(PLACEHOLDER.findall(script)))
    distinct = sorted(set(meshes))
    mesh_note = "k-point mesh" + ("es " if len(distinct) > 1 else " ") + ", ".join(
        "×".join(map(str, m)) for m in distinct) + f" (spacing {kspacing} Å⁻¹)."
    (directory / "README.md").write_text(_readme(levels, len(frames), mesh_note, placeholders),
                                         encoding="utf-8")
    (directory / PACKAGE_FILE).write_text(json.dumps({
        "code": "vasp",
        "levels": list(levels),
        "level_names": {level: LEVEL_NAMES[level] for level in levels},
        "kspacing": kspacing,
        "magmom": magmom,
        "incar_extra": incar_extra or {},
        "frames": len(frames),
        "potcars": "Materials Project (pymatgen MPRelaxSet.yaml, legacy PBE set)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "placeholders_left": placeholders,
    }, indent=1), encoding="utf-8")
    return directory


def _sorted_by_species(atoms: Atoms) -> Atoms:
    """``atoms`` with its atoms grouped by species (VASP needs contiguous species),
    stably, so a frame that is already grouped keeps its order."""
    order = species_order(atoms)
    symbols = atoms.get_chemical_symbols()
    index = sorted(range(len(atoms)), key=lambda i: order.index(symbols[i]))
    if index != list(range(len(atoms))):
        raise ValueError("Group the atoms of each frame by species before packaging "
                         "(VASP needs contiguous species; the collector compares atom by atom)")
    grouped = atoms[index]
    grouped.calc = None
    return grouped


# --- parsing and collecting -------------------------------------------------------------


def _read_vasprun(path: Path) -> Atoms:
    from ase.io import read

    return read(path, index=-1, format="vasp-xml")


def parse_vasp_run(folder: str | Path) -> dict:
    """Energy (eV, force consistent), forces (eV/Å), stress (Voigt, eV/Å³, ASE
    sign), positions, cell, and version from one VASP single point: its
    ``vasprun.xml`` and ``OUTCAR``."""
    folder = Path(folder)
    outcar, vasprun = folder / "OUTCAR", folder / "vasprun.xml"
    if not (outcar.is_file() and vasprun.is_file()):
        raise OutputError("missing vasprun.xml or OUTCAR")
    text = outcar.read_text(encoding="utf-8", errors="replace")
    if "General timing and accounting" not in text:
        raise OutputError("OUTCAR has no timing summary (the run failed or did not finish)")
    if "aborting loop because EDIFF is reached" not in text:
        raise OutputError("the electronic loop did not reach EDIFF (NELM hit)")
    try:
        atoms = _read_vasprun(vasprun)
    except Exception as exc:  # a truncated or malformed vasprun.xml
        raise OutputError(f"vasprun.xml could not be read ({exc})") from exc
    try:
        stress = atoms.get_stress(voigt=True)
    except Exception as exc:
        raise OutputError("no stress in vasprun.xml (run with ISIF >= 2)") from exc
    version = next((line.split()[0] for line in text.splitlines()[:5] if line.startswith(" vasp.")),
                   "VASP")
    return {
        "energy": float(atoms.get_potential_energy(force_consistent=True)),
        "forces": atoms.get_forces(),
        "stress": np.asarray(stress, float),
        "positions": atoms.positions,
        "cell": np.asarray(atoms.cell),
        "symbols": atoms.get_chemical_symbols(),
        "version": version.strip() or "VASP",
    }


@dataclass
class VaspCollectResult:
    labeled: list[Atoms]
    rejected: dict[int, dict[str, str]] = field(default_factory=dict)
    levels: tuple[str, ...] = ()

    def report(self) -> dict:
        used = sorted({frame.info["potcars"] for frame in self.labeled if "potcars" in frame.info})
        return {"levels": list(self.levels), "labeled": len(self.labeled),
                "potcars_used": used,
                "rejected": {frame_name(i): reasons for i, reasons in self.rejected.items()}}


def _check_geometry(parsed: dict, frame: Atoms) -> None:
    if parsed["symbols"] != frame.get_chemical_symbols():
        raise OutputError("the run has other atoms, or another order, than the frame")
    if np.abs(parsed["cell"] - np.asarray(frame.cell)).max() > _GEOMETRY_TOLERANCE:
        raise OutputError("the run's cell differs from the frame's")
    from ase.geometry import find_mic

    shift, _ = find_mic(parsed["positions"] - frame.positions, frame.cell, frame.pbc)
    worst = np.linalg.norm(shift, axis=1).max()
    if worst > _GEOMETRY_TOLERANCE:
        raise OutputError(f"the run's geometry differs from the frame by {worst:.2g} Å")


def collect_vasp_labels(directory: str | Path, *, write: bool = True) -> VaspCollectResult:
    """Parse and check every run of a VASP package; write ``labeled.extxyz``
    (``<LEVEL>_energy``, ``<LEVEL>_forces``, ``<LEVEL>_stress`` per level, e.g.
    ``HSE06_energy``) and ``collect_report.json`` (with ``write``)."""
    from ase.io import read
    from ase.io import write as write_frames

    directory = Path(directory)
    package = json.loads((directory / PACKAGE_FILE).read_text(encoding="utf-8"))
    if package.get("code") != "vasp":
        raise ValueError("Not a VASP labeling package (use labeling.collect_labels)")
    levels = tuple(package["levels"])
    frames = read(directory / "frames.extxyz", ":")
    manifest = Manifest.load(directory / "manifest.json")
    changed = set(manifest.verify(frames))
    result = VaspCollectResult(labeled=[], levels=levels)
    for entry in manifest.frames:
        index = entry.index
        if index in changed:
            result.rejected[index] = {"frame": "frames.extxyz no longer matches the manifest"}
            continue
        frame = frames[index]
        parsed, reasons = {}, {}
        for level in levels:
            try:
                run = parse_vasp_run(directory / "outputs" / frame_name(index) / level)
                _check_geometry(run, frame)
                parsed[level] = run
            except OutputError as exc:
                reasons[level] = str(exc)
        if reasons:
            result.rejected[index] = reasons
            continue
        labeled = Atoms(frame.get_chemical_symbols(), positions=frame.positions,
                        cell=frame.cell, pbc=frame.pbc)
        for level, run in parsed.items():
            prefix = LABEL_PREFIX[level]
            labeled.info[f"{prefix}_energy"] = run["energy"]
            labeled.info[f"{prefix}_stress"] = run["stress"]
            labeled.arrays[f"{prefix}_forces"] = np.asarray(run["forces"], float)
        labeled.info.update(code=next(iter(parsed.values()))["version"], frame=index,
                            source=entry.source, tag=entry.reason,
                            checksum=frame_checksum(frame))
        used = directory / "outputs" / frame_name(index) / "POTCAR.used"
        if used.is_file():  # written by the run script: which POTCARs this frame used
            labeled.info["potcars"] = " ".join(used.read_text().split())
        result.labeled.append(labeled)
    if write:
        if result.labeled:
            write_frames(directory / "labeled.extxyz", result.labeled)
        (directory / "collect_report.json").write_text(json.dumps(result.report(), indent=1),
                                                       encoding="utf-8")
    return result
