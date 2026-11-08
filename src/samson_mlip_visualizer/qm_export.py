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
force   Force NoSymm SCF=Tight            EnGrad TightSCF
======  ===============================  ==================================

``force`` is the labeling job: one energy and gradient at the given geometry,
with NoSymm so Gaussian reports the forces in the input orientation.

The level of theory, charge, and multiplicity are the user's responsibility;
the defaults are only a starting point. Periodic structures are refused: these
are molecular programs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms

JOBS = ("ts", "qst2", "qst3", "irc", "opt", "force", "energy")
DEFAULT_LEVEL = {
    "gaussian": "B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ",
    "orca": "B3LYP D3BJ def2-SVP",
}
_STRUCTURES = {"ts": 1, "opt": 1, "irc": 1, "qst2": 2, "qst3": 3, "force": 1, "energy": 1}
_GAUSSIAN_ROUTE = {
    "ts": "Opt=(TS,CalcFC,NoEigenTest) Freq",
    "qst2": "Opt=QST2 Freq",
    "qst3": "Opt=QST3 Freq",
    "irc": "IRC=(CalcFC,MaxPoints=30)",
    "opt": "Opt Freq",
    "force": "Force NoSymm SCF=Tight",
    "energy": "SP NoSymm SCF=Tight",
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
        "force": "EnGrad TightSCF",
        "energy": "TightSCF",  # a single point, e.g. DLPNO-CCSD(T), which has no gradients
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


# --- Route-preserving Gaussian TS inputs (W2 saddle rescue) -----------------------
#
# A failed Gaussian saddle search is restarted from an MLIP-refined geometry with
# the campaign's own route section, not DEFAULT_LEVEL: only the Opt keyword is
# replaced (by TS_OPT) and Freq is added when missing, so the DFT Freq run decides
# whether the guess is a first-order saddle. Everything else (method/basis, IOps,
# SCF, Int, Pop, Link 0 lines) is copied verbatim. Geom=Checkpoint/AllCheck are
# removed because the geometry is written out explicitly: Gaussian dies in l101
# when both are given.

TS_OPT = "Opt=(TS,CalcFC,NoEigenTest)"
REQUIRED_IOP = "5/13=1"
_OPT = re.compile(r"^opt(imize)?(?=$|[=(])", re.IGNORECASE)
_FREQ = re.compile(r"^freq(uency)?(?=$|[=(])", re.IGNORECASE)
_GEOM = re.compile(r"^geom(etry)?(?=$|[=(])", re.IGNORECASE)
_IOP = re.compile(r"^iop\((?P<body>.*)\)$", re.IGNORECASE)
_GEOM_FROM_CHECKPOINT = {"check", "checkpoint", "allcheck", "allcheckpoint"}


@dataclass(frozen=True)
class GaussianTemplate:
    """The first job of a Gaussian input: Link 0 lines, route tokens, molecule spec."""

    link0: tuple[str, ...]
    route: tuple[str, ...]
    charge: int | None = None
    multiplicity: int | None = None
    extra_jobs: int = 0  # --Link1-- jobs after the first (ignored)


def route_tokens(route: str) -> list[str]:
    """Split a route section into keywords, keeping ``SCF=(A, B)`` in one token.

    Whitespace inside parentheses is dropped (Gaussian ignores it); everything
    else is kept exactly, including case and the leading ``#``/``#p``.
    """
    tokens, current, depth = [], "", 0
    for char in route:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced ')' in route: {route!r}")
        if char.isspace():
            if depth == 0 and current:
                tokens.append(current)
                current = ""
            continue
        current += char
    if depth:
        raise ValueError(f"Unbalanced '(' in route: {route!r}")
    if current:
        tokens.append(current)
    if not tokens or not tokens[0].startswith("#"):
        raise ValueError("A Gaussian route section starts with '#'")
    return tokens


def parse_gaussian_template(text: str) -> GaussianTemplate:
    """Link 0 lines, route tokens and (if present) charge/multiplicity of job 1."""
    jobs = re.split(r"^\s*--link1--\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    lines = jobs[0].splitlines()
    index, link0 = 0, []
    while index < len(lines) and not lines[index].lstrip().startswith("#"):
        stripped = lines[index].strip()
        if stripped.startswith("%"):
            link0.append(stripped)
        elif stripped and not stripped.startswith("!"):
            raise ValueError(f"Unexpected line before the route section: {stripped!r}")
        index += 1
    if index == len(lines):
        raise ValueError("No route section ('#' line) in the template")
    route_lines = []
    while index < len(lines) and lines[index].strip():
        route_lines.append(lines[index].strip())
        index += 1
    tokens = route_tokens(" ".join(route_lines))
    charge = multiplicity = None
    # then: blank, title (one or more lines), blank, "charge multiplicity"
    rest = lines[index:]
    blanks = [k for k, line in enumerate(rest) if not line.strip()]
    if len(blanks) >= 2 and blanks[1] + 1 < len(rest):
        fields = rest[blanks[1] + 1].split()
        if len(fields) >= 2 and all(re.fullmatch(r"[+-]?\d+", f) for f in fields[:2]):
            charge, multiplicity = int(fields[0]), int(fields[1])
    return GaussianTemplate(tuple(link0), tuple(tokens), charge, multiplicity, len(jobs) - 1)


def _iop_settings(token: str) -> list[str]:
    match = _IOP.match(token)
    return [item.strip() for item in match["body"].split(",")] if match else []


def _geom_options(token: str) -> list[str]:
    value = token.split("=", 1)[1] if "=" in token else token[token.find("(") :]
    return [item.strip().lower() for item in value.strip("()").split(",") if item.strip()]


def ts_route_from_template(tokens: Sequence[str]) -> tuple[list[str], list[str]]:
    """The TS route: ``tokens`` with Opt replaced by :data:`TS_OPT` and Freq ensured.

    Returns ``(new_tokens, changes)``. Geom options that read the geometry from the
    checkpoint are removed (other Geom options are kept). Raises ``ValueError`` if
    the route lacks ``IOP(5/13=1)`` (alone or inside a combined IOP list), which
    the W2 ladder needs.
    """
    if not any(REQUIRED_IOP in _iop_settings(token) for token in tokens):
        raise ValueError(
            f"The template route has no IOP({REQUIRED_IOP}); refusing to write a W2 input "
            "without it. Route: " + " ".join(tokens)
        )
    new, changes, opt_done = [], [], False
    for token in tokens:
        if _OPT.match(token):
            if not opt_done:
                new.append(TS_OPT)
                opt_done = True
            if token != TS_OPT:
                changes.append(f"replaced {token} with {TS_OPT}")
            continue
        if _GEOM.match(token):
            options = _geom_options(token)
            kept = [o for o in options if o not in _GEOM_FROM_CHECKPOINT]
            if len(kept) != len(options):
                changes.append(f"removed {token} (the geometry is written explicitly)")
                if kept:
                    new.append(f"Geom=({','.join(kept)})" if len(kept) > 1 else f"Geom={kept[0]}")
                continue
        new.append(token)
    if not opt_done:
        # after '#'/'#p' and the method/basis token
        at = 2 if len(new) > 1 and "/" in new[1] and "=" not in new[1] else 1
        new.insert(at, TS_OPT)
        changes.append(f"added {TS_OPT}")
    if not any(_FREQ.match(token) for token in new):
        new.insert(new.index(TS_OPT) + 1, "Freq")
        changes.append("added Freq")
    return new, changes


def gaussian_ts_input_from_template(
    template: str | GaussianTemplate,
    atoms: Atoms,
    *,
    charge: int,
    multiplicity: int,
    title: str,
    chk: str | None = None,
) -> tuple[str, list[str]]:
    """A Gaussian TS + Freq input for ``atoms`` with the template's route and Link 0.

    ``template`` is the text of a campaign .gjf (or its parsed form). Only the Opt
    keyword changes (see :func:`ts_route_from_template`); ``%chk`` is renamed to
    ``chk`` when given, so restarts do not share a checkpoint. The geometry is
    written as explicit Cartesians. Returns ``(text, changes)``.
    """
    _check([atoms], "ts")
    parsed = (template if isinstance(template, GaussianTemplate)
              else parse_gaussian_template(template))
    route, changes = ts_route_from_template(parsed.route)
    reads_guess = any(re.match(r"^guess=\(?[^)]*\bread\b", t, re.IGNORECASE) for t in route)
    has_oldchk = any(line.lower().startswith("%oldchk=") for line in parsed.link0)
    link0 = []
    for line in parsed.link0:
        if chk and line.lower().startswith("%chk="):
            link0.append(f"%chk={chk}")
            if line != f"%chk={chk}":
                changes.append(f"renamed {line} to %chk={chk}")
                if reads_guess and not has_oldchk:
                    # Guess=Read must still find the failed run's orbitals
                    link0.append(f"%oldchk={line.split('=', 1)[1]}")
                    changes.append(f"added %oldchk={line.split('=', 1)[1]} for Guess=Read")
            continue
        link0.append(line)
    if chk and not any(line.lower().startswith("%chk=") for line in link0):
        link0.insert(0, f"%chk={chk}")
        changes.append(f"added %chk={chk}")
    if parsed.extra_jobs:
        changes.append(f"ignored {parsed.extra_jobs} --Link1-- job(s) after the first")
    for token in route:  # the l101 trap, whatever the code above does
        if _GEOM.match(token) and set(_geom_options(token)) & _GEOM_FROM_CHECKPOINT:
            raise AssertionError("Geom=Checkpoint must never be written with explicit atoms")
    title = " ".join(title.split()) or "TS guess"
    lines = [*link0, " ".join(route), "", title, "", f"{charge} {multiplicity}",
             _coordinates(atoms), ""]
    return "\n".join(lines) + "\n", changes
