"""Quantum-chemistry inputs (Gaussian, ORCA) that continue from an MLIP result.

For hard cases the MLIP does the fast exploration (TS guess, path, Hessian) and a
quantum-chemistry method gives the reference answer. These writers turn one,
two, or three structures into a ready-to-edit input:

======  ===============================  ==================================
job     Gaussian (.gjf / .com)           ORCA (.inp)
======  ===============================  ==================================
ts      Opt=(TS,CalcFC,NoEigenTest) Freq  OptTS Freq, exact initial Hessian
qst2    Opt=QST2 Freq (reactant, product) NEB-TS Freq (+ product.xyz)
qst3    Opt=QST3 Freq (+ TS guess)        NEB-TS Freq (+ product/guess .xyz)
irc     IRC=(CalcFC)                      IRC, exact initial Hessian
opt     Opt Freq                          Opt Freq
======  ===============================  ==================================

The level of theory, charge, and multiplicity are the user's responsibility;
the defaults are only a starting point. Periodic structures are refused: these
are molecular programs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ase import Atoms

JOBS = ("ts", "qst2", "qst3", "irc", "opt")
DEFAULT_LEVEL = {
    "gaussian": "B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ",
    "orca": "B3LYP D3BJ def2-SVP",
}
_STRUCTURES = {"ts": 1, "opt": 1, "irc": 1, "qst2": 2, "qst3": 3}
_GAUSSIAN_ROUTE = {
    "ts": "Opt=(TS,CalcFC,NoEigenTest) Freq",
    "qst2": "Opt=QST2 Freq",
    "qst3": "Opt=QST3 Freq",
    "irc": "IRC=(CalcFC,MaxPoints=30)",
    "opt": "Opt Freq",
}
_QST_TITLES = ("Reactant", "Product", "Transition-state guess")


def program_for(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in (".gjf", ".com"):
        return "gaussian"
    if suffix == ".inp":
        return "orca"
    raise ValueError("Use .gjf or .com for Gaussian, .inp for ORCA")


def _check(structures: Sequence[Atoms], job: str) -> None:
    if job not in JOBS:
        raise ValueError(f"Unknown job {job!r}; choose one of {', '.join(JOBS)}")
    if len(structures) != _STRUCTURES[job]:
        raise ValueError(f"A {job} input needs {_STRUCTURES[job]} structure(s)")
    for atoms in structures:
        if atoms.pbc.any():
            raise ValueError("Periodic structures cannot be exported to molecular QM codes")
    symbols = structures[0].get_chemical_symbols()
    if any(atoms.get_chemical_symbols() != symbols for atoms in structures[1:]):
        raise ValueError("QST structures must contain the same atoms in the same order")


def _coordinates(atoms: Atoms) -> str:
    return "\n".join(
        f"{symbol:<2s} {x:16.8f} {y:16.8f} {z:16.8f}"
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions, strict=True)
    )


def gaussian_input(
    structures: Sequence[Atoms],
    *,
    job: str = "ts",
    level: str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    nproc: int | None = None,
    memory_gb: int | None = None,
    title: str = "Continued from an MLIP result (SAMSON MLIP Visualizer)",
) -> str:
    """Gaussian input text; QST jobs get one titled geometry section per structure."""
    _check(structures, job)
    lines = []
    if nproc:
        lines.append(f"%nprocshared={nproc}")
    if memory_gb:
        lines.append(f"%mem={memory_gb}GB")
    lines += [f"#p {level or DEFAULT_LEVEL['gaussian']} {_GAUSSIAN_ROUTE[job]}", ""]
    titles = _QST_TITLES if job.startswith("qst") else (title,)
    for section, atoms in enumerate(structures):
        heading = f"{titles[section]} ({title})" if job.startswith("qst") else titles[section]
        lines += [heading, "", f"{charge} {multiplicity}", _coordinates(atoms), ""]
    return "\n".join(lines) + "\n"


def orca_input(
    structures: Sequence[Atoms],
    *,
    job: str = "ts",
    level: str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    nproc: int | None = None,
    memory_gb: int | None = None,
    stem: str = "structure",
) -> tuple[str, dict[str, str]]:
    """ORCA input text plus the extra .xyz files an NEB-TS job refers to."""
    _check(structures, job)
    keywords = {
        "ts": "OptTS Freq",
        "qst2": "NEB-TS Freq",
        "qst3": "NEB-TS Freq",
        "irc": "IRC",
        "opt": "Opt Freq",
    }[job]
    lines = [f"! {level or DEFAULT_LEVEL['orca']} {keywords}"]
    if nproc:
        lines += ["%pal", f"  nprocs {nproc}", "end"]
    if memory_gb:
        lines.append(f"%maxcore {int(memory_gb * 1000 / (nproc or 1))}")
    extra: dict[str, str] = {}
    if job == "ts":
        lines += ["%geom", "  Calc_Hess true", "end"]
    elif job == "irc":
        lines += ["%irc", "  InitHess calc_anfreq", "end"]
    elif job.startswith("qst"):
        product = f"{stem}_product.xyz"
        extra[product] = _xyz(structures[1], "product")
        lines += ["%neb", f'  NEB_End_XYZFile "{product}"']
        if job == "qst3":
            guess = f"{stem}_ts_guess.xyz"
            extra[guess] = _xyz(structures[2], "transition-state guess")
            lines.append(f'  NEB_TS_XYZFile "{guess}"')
        lines.append("end")
    lines += [f"* xyz {charge} {multiplicity}", _coordinates(structures[0]), "*"]
    return "\n".join(lines) + "\n", extra


def _xyz(atoms: Atoms, comment: str) -> str:
    return f"{len(atoms)}\n{comment}\n{_coordinates(atoms)}\n"


def write_qm_input(
    path: str | Path,
    structures: Sequence[Atoms],
    *,
    job: str = "ts",
    level: str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    nproc: int | None = None,
    memory_gb: int | None = None,
) -> list[Path]:
    """Write a Gaussian or ORCA input (chosen by the extension); return the files written."""
    path = Path(path)
    program = program_for(path)
    options = {
        "job": job,
        "level": level or None,
        "charge": charge,
        "multiplicity": multiplicity,
        "nproc": nproc,
        "memory_gb": memory_gb,
    }
    if program == "gaussian":
        path.write_text(gaussian_input(structures, **options), encoding="utf-8")
        return [path]
    text, extra = orca_input(structures, stem=path.stem, **options)
    path.write_text(text, encoding="utf-8")
    written = [path]
    for name, content in extra.items():
        target = path.with_name(name)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written
