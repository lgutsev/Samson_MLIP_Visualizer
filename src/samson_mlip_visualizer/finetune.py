"""Building blocks for fine-tuning an MLIP on reference calculations.

The workflow (see ``docs/fine_tuning.md``): explore with the current model,
select frames worth labeling, label them with a reference code (locally, or on
an HPC from a package this tool writes), align the reference energies with the
foundation model's scale, fine-tune, evaluate, and repeat. This module holds the
pieces that do not depend on the reference code:

- :func:`structure_distance`: how far apart two frames of one system are, with
  rigid motion removed (rotation too, for molecules);
- :func:`select_for_labeling`: frames chosen by committee disagreement,
  geometric diversity (farthest-point sampling), and random spot checks, each
  kept a minimum distance from the others and from frames already labeled;
- :func:`fit_element_offsets`: per-element energy offsets between a reference
  code and the foundation model, fitted by least squares; a single constant
  shift only aligns one composition;
- :class:`Manifest`: which frames were selected, from where, and why, with a
  checksum per frame so returned labels can be matched to the right structure;
- :func:`labeled_structure`: a frame with ``REF_energy`` / ``REF_forces`` /
  ``REF_stress`` for mace-torch, keeping the raw reference energy.

Lessons from the HCN example shape the defaults: a committee of seeds from one
foundation model is overconfident, so disagreement never selects alone
(diversity and random spot checks sit beside it), and held-out errors are only
meaningful next to the distance from the training data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from .engine import _committee_spread
from .reaction_path import aligned_rmsd

MANIFEST_VERSION = 1


# --- distances --------------------------------------------------------------------


def structure_distance(a: Atoms, b: Atoms) -> float:
    """RMSD (Å) between two frames of the same system, rigid motion removed.

    Molecules (no periodic axis) are superimposed (Kabsch), so rotation and
    translation do not count. Periodic frames keep their orientation: the
    displacement is taken with the minimum-image convention in ``a``'s cell and
    its mean (a rigid translation) is removed. The frames must list the same
    elements in the same order.
    """
    if a.get_chemical_symbols() != b.get_chemical_symbols():
        raise ValueError("Frames must contain the same atoms in the same order")
    if not a.pbc.any():
        return aligned_rmsd(a.positions, b.positions)
    displacement, _ = find_mic(b.positions - a.positions, a.cell, a.pbc)
    displacement -= displacement.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))


def distances_to(
    frames: Sequence[Atoms],
    references: Sequence[Atoms],
    distance: Callable[[Atoms, Atoms], float] = structure_distance,
) -> np.ndarray:
    """For each frame, the distance to its nearest reference (inf without references)."""
    nearest = np.full(len(frames), np.inf)
    for reference in references:
        nearest = np.minimum(nearest, [distance(frame, reference) for frame in frames])
    return nearest


def committee_spread(frames: Sequence[Atoms], calculator) -> tuple[np.ndarray, np.ndarray]:
    """Energy std (eV) and largest per-atom force std (eV/Å) of a committee
    calculator (several MACE files) on each frame."""
    energy, force = [], []
    for frame in frames:
        atoms = frame.copy()
        atoms.calc = calculator
        atoms.get_forces()
        e_std, f_std = _committee_spread(calculator)
        if e_std is None or f_std is None:
            raise ValueError("The calculator is not a committee (no energy_comm/forces_comm)")
        energy.append(e_std)
        force.append(f_std)
    return np.array(energy), np.array(force)


# --- selection ----------------------------------------------------------------------


@dataclass
class Selection:
    """Chosen candidate indices, why each was chosen, and its score: the
    committee spread (``committee``), the distance to the nearest already chosen
    or labeled frame (``diversity``), or nothing (``spot-check``)."""

    indices: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    scores: list[float | None] = field(default_factory=list)
    nearest_distance: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.indices)

    def add(self, index: int, reason: str, score: float | None, nearest: float) -> None:
        self.indices.append(int(index))
        self.reasons.append(reason)
        self.scores.append(None if score is None else float(score))
        self.nearest_distance.append(float(nearest))


def select_for_labeling(
    candidates: Sequence[Atoms],
    *,
    spread: Sequence[float] | None = None,
    n_committee: int = 0,
    n_diverse: int = 0,
    n_random: int = 0,
    labeled: Sequence[Atoms] = (),
    min_distance: float = 0.05,
    seed: int = 0,
    distance: Callable[[Atoms, Atoms], float] = structure_distance,
) -> Selection:
    """Choose frames to label from ``candidates``, in three passes:

    1. ``n_committee`` frames with the largest committee ``spread`` (e.g. the
       force std from :func:`committee_spread`);
    2. ``n_diverse`` frames by farthest-point sampling: each time, the frame
       farthest from everything chosen or labeled so far;
    3. ``n_random`` random spot checks, which measure whether the committee's
       spread tracks the real error.

    Every chosen frame is at least ``min_distance`` (Å, see
    :func:`structure_distance`) from the others and from ``labeled``; a pass
    ends early when no candidate is that far from everything. Fewer frames than
    asked means the candidates add little new.
    """
    if n_committee and spread is None:
        raise ValueError("Committee selection needs the committee spread of each candidate")
    if spread is not None and len(spread) != len(candidates):
        raise ValueError("spread needs one value per candidate")
    nearest = distances_to(candidates, labeled, distance)
    chosen = Selection()
    available = np.ones(len(candidates), bool)

    def take(index: int, reason: str, score: float | None) -> None:
        nonlocal nearest
        chosen.add(index, reason, score, nearest[index])
        available[index] = False
        pick = candidates[index]
        nearest = np.minimum(nearest, [distance(frame, pick) for frame in candidates])

    if n_committee:
        for index in np.argsort(np.asarray(spread, float))[::-1]:
            if len(chosen) == n_committee:
                break
            if available[index] and nearest[index] >= min_distance:
                take(int(index), "committee", spread[index])
    for _ in range(n_diverse):
        pool = np.where(available)[0]
        if not len(pool):
            break
        index = int(pool[np.argmax(nearest[pool])])
        if nearest[index] < min_distance:
            break
        take(index, "diversity", None if np.isinf(nearest[index]) else nearest[index])
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        pool = np.where(available & (nearest >= min_distance))[0]
        if not len(pool):
            break
        take(int(rng.choice(pool)), "spot-check", None)
    return chosen


# --- energy scales ------------------------------------------------------------------


@dataclass
class ElementOffsets:
    """Per-element offsets δ_Z (eV per atom) with E_reference ≈ E_foundation + Σ n_Z δ_Z."""

    offsets: dict[str, float]
    rank: int
    residual_rms_ev: float
    residual_max_ev: float
    structures: int
    warnings: list[str] = field(default_factory=list)

    def correction(self, atoms: Atoms) -> float:
        symbols = atoms.get_chemical_symbols()
        missing = sorted(set(symbols) - set(self.offsets))
        if missing:
            raise ValueError(f"No fitted offset for {', '.join(missing)}")
        return float(sum(self.offsets[symbol] for symbol in symbols))

    def to_foundation_scale(self, energy: float, atoms: Atoms) -> float:
        """A reference energy moved onto the foundation model's scale."""
        return float(energy) - self.correction(atoms)

    def residuals(
        self, structures: Sequence[Atoms], reference: Sequence[float], foundation: Sequence[float]
    ) -> np.ndarray:
        """Remaining difference (eV) per structure; use on held-out structures."""
        return np.array([
            r - f - self.correction(atoms)
            for atoms, r, f in zip(structures, reference, foundation, strict=True)
        ])

    def to_json(self) -> dict:
        return asdict(self)


def fit_element_offsets(
    structures: Sequence[Atoms],
    reference_energies: Sequence[float],
    foundation_energies: Sequence[float],
) -> ElementOffsets:
    """Least-squares per-element offsets between a reference code and the
    foundation model, from structures labeled by both.

    Codes with other pseudopotentials, basis sets, or all-electron energies
    (Gaussian, ORCA, CP2K, Psi4) differ from VASP-trained foundation energies
    by roughly a constant per atom of each element. With one composition (or
    compositions that are multiples of each other) the offsets are not unique:
    the minimum-norm solution is returned with a warning, and it aligns only
    those compositions.
    """
    if not structures or not (
        len(structures) == len(reference_energies) == len(foundation_energies)
    ):
        raise ValueError("Give one reference and one foundation energy per structure")
    elements = sorted({symbol for atoms in structures for symbol in atoms.get_chemical_symbols()})
    counts = np.array([
        [atoms.get_chemical_symbols().count(element) for element in elements]
        for atoms in structures
    ], float)
    difference = np.asarray(reference_energies, float) - np.asarray(foundation_energies, float)
    solution, *_ = np.linalg.lstsq(counts, difference, rcond=None)
    rank = int(np.linalg.matrix_rank(counts))
    residual = difference - counts @ solution
    warnings = []
    if rank < len(elements):
        warnings.append(
            f"The {len(structures)} structures span only {rank} independent composition(s) for "
            f"{len(elements)} elements, so the offsets are not unique; they align only these "
            "compositions. Add structures with other compositions (e.g. the isolated "
            "molecules or elements) before labeling anything else."
        )
    if len(structures) <= rank:
        warnings.append("As many unknowns as structures: the residual says nothing; hold some out.")
    return ElementOffsets(
        offsets={element: float(value) for element, value in zip(elements, solution, strict=True)},
        rank=rank,
        residual_rms_ev=float(np.sqrt(np.mean(residual**2))),
        residual_max_ev=float(np.abs(residual).max()),
        structures=len(structures),
        warnings=warnings,
    )


# --- manifests and labeled frames -----------------------------------------------------


def frame_checksum(atoms: Atoms) -> str:
    """SHA-256 of the elements, positions (to 1e-6 Å), cell, and PBC of a frame."""
    digest = hashlib.sha256()
    digest.update(" ".join(atoms.get_chemical_symbols()).encode())
    digest.update(np.round(atoms.positions, 6).tobytes())
    digest.update(np.round(np.asarray(atoms.cell), 6).tobytes())
    digest.update(np.asarray(atoms.pbc, bool).tobytes())
    return digest.hexdigest()


@dataclass
class FrameEntry:
    index: int  # position in the frames file
    source: str  # the run the frame came from, e.g. "irc round 1"
    reason: str  # committee | diversity | spot-check | seed | ...
    checksum: str
    score: float | None = None
    nearest_labeled_distance: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Manifest:
    """What was selected for labeling, from where, and why."""

    frames: list[FrameEntry]
    model: str = ""  # the model that explored and scored the frames
    notes: dict = field(default_factory=dict)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    version: int = MANIFEST_VERSION

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != MANIFEST_VERSION:
            raise ValueError(f"Unsupported manifest version {data.get('version')!r}")
        data["frames"] = [FrameEntry(**entry) for entry in data["frames"]]
        return cls(**data)

    def verify(self, frames: Sequence[Atoms]) -> list[int]:
        """Indices of manifest entries whose frame is missing or has changed."""
        return [
            entry.index
            for entry in self.frames
            if entry.index >= len(frames) or frame_checksum(frames[entry.index]) != entry.checksum
        ]


def write_selection(
    directory: str | Path,
    candidates: Sequence[Atoms],
    selection: Selection,
    *,
    source: str,
    model: str = "",
    notes: dict | None = None,
) -> tuple[Path, Path]:
    """``frames.extxyz`` (the selected frames, in order) and ``manifest.json``.

    The checksums are those of the frames as stored: extxyz keeps 8 decimals, so
    a frame held in memory can round differently (at 1e-6 Å) from the same frame
    read back, and every consumer reads the file."""
    from ase.io import read, write

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames = [candidates[index].copy() for index in selection.indices]
    for frame in frames:
        frame.calc = None
    frames_path = directory / "frames.extxyz"
    write(frames_path, frames)
    if frames:  # ASE cannot read an empty file back
        frames = read(frames_path, ":")
    entries = [
        FrameEntry(
            index=position,
            source=source,
            reason=reason,
            checksum=frame_checksum(frame),
            score=score,
            nearest_labeled_distance=None if np.isinf(nearest) else nearest,
            extra={"candidate_index": candidate},
        )
        for position, (frame, candidate, reason, score, nearest) in enumerate(
            zip(frames, selection.indices, selection.reasons, selection.scores,
                selection.nearest_distance, strict=True)
        )
    ]
    manifest_path = Manifest(entries, model=model, notes=dict(notes or {})).save(
        directory / "manifest.json"
    )
    return frames_path, manifest_path


def labeled_structure(
    atoms: Atoms,
    energy: float,
    forces,
    *,
    stress=None,
    offsets: ElementOffsets | None = None,
    tag: str | None = None,
    meta: dict | None = None,
) -> Atoms:
    """A frame with reference labels under the keys mace-torch is told to read.

    ``REF_energy`` is on the foundation model's scale when ``offsets`` are given
    (and equal to the reference energy otherwise); ``REF_energy_raw`` keeps the
    code's own number. ``stress`` is 6 Voigt components or a 3×3 matrix, eV/Å³.
    """
    labeled = Atoms(
        atoms.get_chemical_symbols(), positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc
    )
    labeled.info["REF_energy_raw"] = float(energy)
    labeled.info["REF_energy"] = (
        float(energy) if offsets is None else offsets.to_foundation_scale(energy, atoms)
    )
    if stress is not None:
        stress = np.asarray(stress, float)
        if stress.shape == (3, 3):
            stress = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        labeled.info["REF_stress"] = stress
    if tag:
        labeled.info["tag"] = tag
    for key, value in (meta or {}).items():
        labeled.info[key] = value
    labeled.arrays["REF_forces"] = np.asarray(forces, float).reshape(len(atoms), 3)
    return labeled
