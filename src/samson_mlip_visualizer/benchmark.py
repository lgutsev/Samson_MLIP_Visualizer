"""Benchmark a model against a reference method along a reaction path.

A converged transition state from an MLIP is only as good as the MLIP in that
region, and foundation models see few transition states in training. This
module takes the frames of a path (an IRC, a scan, a QST band), evaluates a
model and a reference (PBE through Psi4, xTB, another MLIP) on the same
geometries, and reports where they disagree:

- energies relative to one frame (default: the first, e.g. the reactant), so
  each method is measured against its own reference state;
- per-atom force errors (mean and largest atom) along the path;
- when the model is a committee (several MACE checkpoints), its own spread
  along the path: the energy std and the largest per-atom force std;
- two figures: the energy profiles with their difference, and the mean force
  magnitudes with the mean and max force error.

The committee spread needs no reference calculation, so it is the cheap signal
for choosing which frames to label; the reference error is the ground truth it
should track. Together they drive active learning along an IRC.

Typical use: the model is right at the minima and wrong in the bent,
bond-shifting region between them, which is where fine-tuning data belongs.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms

from .engine import _committee_spread

# Chart colors: the first two categorical slots (model, reference), ink for text
# and single-series marks, and hairlines that recede.
_SURFACE, _INK, _INK2, _MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
_MODEL, _REFERENCE = "#2a78d6", "#eb6834"


@dataclass
class PathBenchmark:
    """A model and a reference evaluated on the same path frames."""

    names: tuple[str, str]  # (model, reference)
    coordinate: np.ndarray
    coordinate_label: str
    energies: dict[str, np.ndarray]  # absolute, eV
    forces: dict[str, np.ndarray]  # (frames, atoms, 3), eV/Å
    frame_indices: list[int]
    reference_frame: int = 0
    descriptor: tuple[str, np.ndarray] | None = None  # e.g. ("∠H–C–N", angles)
    stopped: bool = False
    # Committee spread of the model, when it is a committee: energy std (eV) and
    # the largest per-atom force std (eV/Å) per frame.
    committee_energy_std: np.ndarray | None = None
    committee_force_std: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.names[0]

    @property
    def reference(self) -> str:
        return self.names[1]

    def relative(self, name: str) -> np.ndarray:
        """Energies of ``name`` relative to its own value at the reference frame."""
        energies = self.energies[name]
        return energies - energies[self.reference_frame]

    @property
    def energy_error(self) -> np.ndarray:
        return self.relative(self.model) - self.relative(self.reference)

    def force_errors(self) -> np.ndarray:
        """Per-atom |F_model − F_reference| for every frame, (frames, atoms)."""
        return np.linalg.norm(self.forces[self.model] - self.forces[self.reference], axis=2)

    @property
    def force_error_mean(self) -> np.ndarray:
        return self.force_errors().mean(axis=1)

    @property
    def force_error_max(self) -> np.ndarray:
        return self.force_errors().max(axis=1)

    def mean_force(self, name: str) -> np.ndarray:
        """Mean per-atom force magnitude of ``name`` along the path."""
        return np.linalg.norm(self.forces[name], axis=2).mean(axis=1)

    def summary(self) -> dict[str, float]:
        error = self.energy_error
        worst = int(np.argmax(np.abs(error)))
        return {
            "frames": len(self.coordinate),
            "model_peak_ev": float(self.relative(self.model).max()),
            "reference_peak_ev": float(self.relative(self.reference).max()),
            "energy_error_max_ev": float(error[worst]),
            "energy_error_max_at": float(self.coordinate[worst]),
            "energy_error_mae_ev": float(np.abs(error).mean()),
            "energy_error_last_ev": float(error[-1]),
            "force_error_mean_ev_per_angstrom": float(self.force_error_mean.mean()),
            "force_error_max_ev_per_angstrom": float(self.force_error_max.max()),
            **(
                {}
                if self.committee_force_std is None
                else {
                    "committee_energy_std_max_ev": float(self.committee_energy_std.max()),
                    "committee_force_std_max_ev_per_angstrom": float(
                        self.committee_force_std.max()
                    ),
                }
            ),
        }

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        columns = {
            self.coordinate_label: self.coordinate,
            "frame": np.asarray(self.frame_indices),
            f"dE_{self.model}_eV": self.relative(self.model),
            f"dE_{self.reference}_eV": self.relative(self.reference),
            "energy_error_eV": self.energy_error,
            f"mean_F_{self.model}_eV_per_A": self.mean_force(self.model),
            f"mean_F_{self.reference}_eV_per_A": self.mean_force(self.reference),
            "force_error_mean_eV_per_A": self.force_error_mean,
            "force_error_max_eV_per_A": self.force_error_max,
        }
        if self.committee_force_std is not None:
            columns["committee_energy_std_eV"] = self.committee_energy_std
            columns["committee_force_std_max_eV_per_A"] = self.committee_force_std
        if self.descriptor is not None:
            columns[self.descriptor[0]] = self.descriptor[1]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in zip(*columns.values(), strict=True):
                writer.writerow([f"{value:.8g}" for value in row])
        return path


def select_frames(
    count: int,
    points: int | None,
    keep: Sequence[int] = (),
    coordinate: Sequence[float] | None = None,
) -> list[int]:
    """About ``points`` frame indices spread evenly along ``coordinate`` (default:
    the frame index), always including both ends and ``keep`` (e.g. the TS).
    IRC steps shorten near the minima, so spacing by index would crowd the ends."""
    if count < 1:
        raise ValueError("The path has no frames")
    if points is None or points >= count:
        return list(range(count))
    x = np.arange(count, dtype=float) if coordinate is None else np.asarray(coordinate, float)
    chosen = {int(np.argmin(np.abs(x - value))) for value in np.linspace(x[0], x[-1], points)}
    return sorted(chosen | {0, count - 1} | {int(k) for k in keep if 0 <= k < count})


def bond_angle(frames: Sequence[np.ndarray], triple: tuple[int, int, int]) -> np.ndarray:
    """The angle i–j–k (degrees, at j) in each frame, for a descriptor axis."""
    i, j, k = triple
    values = []
    for positions in frames:
        a, b = positions[i] - positions[j], positions[k] - positions[j]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        values.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.array(values)


def benchmark_path(
    frames: Sequence[Atoms],
    model,
    reference,
    *,
    names: tuple[str, str] = ("model", "reference"),
    coordinate: Sequence[float] | None = None,
    coordinate_label: str = "frame",
    points: int | None = None,
    keep: Sequence[int] = (),
    reference_frame: int = 0,
    descriptor: tuple[str, Sequence[float]] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> PathBenchmark:
    """Evaluate ``model`` and ``reference`` (ASE calculators) on the path frames.

    ``coordinate`` (default: the frame index) is the x axis, e.g. the IRC arc
    length; ``points`` subsamples long paths evenly, keeping both ends and
    ``keep``; ``descriptor`` is an optional per-frame geometric label (full
    path length) printed under the plots. ``reference_frame`` indexes the
    *selected* frames. ``on_progress(done, total)`` reports each frame.
    """
    if model is reference:
        raise ValueError("The model and the reference must be different calculators")
    if names[0] == names[1]:
        raise ValueError(f"The model and the reference need different names, not {names[0]!r}")
    indices = select_frames(len(frames), points, keep, coordinate)
    x = np.arange(len(frames), dtype=float) if coordinate is None else np.asarray(coordinate, float)
    energies = {name: [] for name in names}
    forces = {name: [] for name in names}
    spread: list[tuple[float | None, float | None]] = []
    stopped = False
    for done, index in enumerate(indices, 1):
        if should_stop and should_stop():
            stopped = True
            break
        for name, calculator in zip(names, (model, reference), strict=True):
            atoms = frames[index].copy()
            atoms.calc = calculator
            energies[name].append(float(atoms.get_potential_energy()))
            forces[name].append(atoms.get_forces().copy())
            if calculator is model:
                spread.append(_committee_spread(calculator))
        if on_progress:
            on_progress(done, len(indices))
    evaluated = indices[: len(energies[names[0]])]
    if not evaluated:
        raise ValueError("The benchmark was stopped before its first frame")
    return PathBenchmark(
        names=names,
        coordinate=x[evaluated],
        coordinate_label=coordinate_label,
        energies={name: np.array(values) for name, values in energies.items()},
        forces={name: np.array(values) for name, values in forces.items()},
        frame_indices=evaluated,
        reference_frame=min(reference_frame, len(evaluated) - 1),
        descriptor=None
        if descriptor is None
        else (descriptor[0], np.asarray(descriptor[1], float)[evaluated]),
        stopped=stopped,
        committee_energy_std=_column([energy for energy, _ in spread]),
        committee_force_std=_column([force for _, force in spread]),
    )


def _column(values: list[float | None]) -> np.ndarray | None:
    return None if not values or any(value is None for value in values) else np.array(values)


# --- figures ----------------------------------------------------------------------


def _axes(rows: int = 2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _INK2,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "text.color": _INK,
        }
    )
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(8, 6.2),
        sharex=True,
        dpi=150,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )
    figure.patch.set_facecolor(_SURFACE)
    for ax in axes:
        ax.set_facecolor(_SURFACE)
        ax.grid(axis="y", color=_GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    return plt, figure, axes


def _line(ax, x, y, color, label=None, **style):
    ax.plot(x, y, color=color, lw=2, marker="o", ms=4, mec=_SURFACE, mew=1, label=label, **style)


def _mark(axes, mark):
    if mark is None:
        return
    position, label = mark
    for ax in axes:
        ax.axvline(position, color=_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    axes[0].annotate(
        label, (position, 1.0), xycoords=("data", "axes fraction"), xytext=(-4, -4),
        textcoords="offset points", ha="right", va="top", color=_INK2,
    )


def _descriptor(ax, bench: PathBenchmark):
    """The descriptor at five points evenly spread along the x axis, the first
    one named (e.g. "∠H–C–N 180°")."""
    if bench.descriptor is None:
        return
    label, values = bench.descriptor
    x = bench.coordinate
    picks = sorted({int(np.argmin(np.abs(x - value))) for value in np.linspace(x[0], x[-1], 5)})
    for n, index in enumerate(picks):
        text = f"{values[index]:.0f}°" if "∠" in label else f"{values[index]:.2f}"
        ax.annotate(
            f"{label} {text}" if n == 0 else text,
            (x[index], 0.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
            textcoords="offset points", ha="left" if n == 0 else "center", va="bottom",
            color=_MUTED, fontsize=8,
        )


def _end_labels(ax, bench: PathBenchmark, end_labels):
    if not end_labels:
        return
    x = bench.coordinate
    reference = bench.relative(bench.reference)
    for position, value, text, align in (
        (x[0], reference[0], end_labels[0], "left"),
        (x[-1], reference[-1], end_labels[1], "right"),
    ):
        ax.annotate(
            text, (position, value), xytext=(0, -14), textcoords="offset points",
            ha=align, va="top", color=_INK2,
        )


def plot_energy(
    bench: PathBenchmark,
    path: str | Path,
    *,
    title: str | None = None,
    mark: tuple[float, str] | None = None,
    end_labels: tuple[str, str] | None = None,
) -> Path:
    """Energy profiles of both methods (top) and model − reference (bottom)."""
    plt, figure, (top, bottom) = _axes()
    x = bench.coordinate
    model, reference = bench.relative(bench.model), bench.relative(bench.reference)
    _line(top, x, model, _MODEL, bench.model)
    _line(top, x, reference, _REFERENCE, bench.reference)
    # Label each peak value above the higher curve and below the lower one.
    for values, name in ((model, bench.model), (reference, bench.reference)):
        k = int(np.argmax(values))
        higher = values[k] >= max(model.max(), reference.max()) - 1e-12
        top.annotate(
            f"{name}  {values[k]:.2f} eV", (x[k], values[k]), xytext=(10, 10 if higher else -16),
            textcoords="offset points", ha="left", va="center", color=_INK2,
        )
    top.set_ylabel(f"Energy relative to frame {bench.frame_indices[bench.reference_frame]} (eV)")
    top.legend(frameon=False, loc="upper left", labelcolor=_INK2)
    _end_labels(top, bench, end_labels)
    low, high = min(model.min(), reference.min()), max(model.max(), reference.max())
    top.set_ylim(low - 0.15 * (high - low), high + 0.15 * (high - low))

    error = bench.energy_error
    bottom.axhline(0, color=_MUTED, lw=0.8)
    _line(bottom, x, error, _INK2)
    bottom.fill_between(x, 0, error, color=_INK2, alpha=0.08, lw=0)
    k = int(np.argmax(np.abs(error)))
    bottom.annotate(
        f"{error[k]:+.2f} eV", (x[k], error[k]), xytext=(0, 9 if error[k] >= 0 else -9),
        textcoords="offset points", ha="center", va="bottom" if error[k] >= 0 else "top",
        color=_INK2,
    )
    # The error is zero at the reference frame by construction; label the far end.
    far = len(x) - 1 if bench.reference_frame == 0 else 0
    bottom.annotate(
        f"{error[far]:+.2f} eV", (x[far], error[far]), xytext=(0, 9), textcoords="offset points",
        ha="right" if far else "left", color=_INK2,
    )
    if bench.committee_energy_std is not None:
        bottom.plot(x, bench.committee_energy_std, color=_MUTED, lw=1.2, ls=(0, (4, 2)),
                    label="committee σ")
        bottom.legend(frameon=False, loc="upper right", labelcolor=_INK2)
    bottom.set_ylabel(f"{bench.model} − {bench.reference} (eV)")
    bottom.set_xlabel(bench.coordinate_label)
    low, high = min(error.min(), 0.0), max(error.max(), 0.0)
    if bench.committee_energy_std is not None:
        high = max(high, bench.committee_energy_std.max())
    pad = 0.25 * max(high - low, 0.05)
    bottom.set_ylim(low - pad, high + pad)
    _mark((top, bottom), mark)
    _descriptor(bottom, bench)
    top.set_title(title or f"{bench.model} vs {bench.reference}: energy along the path",
                  loc="left", color=_INK, fontsize=11)
    figure.savefig(path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(figure)
    return Path(path)


def plot_forces(
    bench: PathBenchmark,
    path: str | Path,
    *,
    title: str | None = None,
    mark: tuple[float, str] | None = None,
) -> Path:
    """Mean per-atom force magnitudes (top) and the force error (bottom): the
    mean over atoms, and the worst atom."""
    plt, figure, (top, bottom) = _axes()
    x = bench.coordinate
    _line(top, x, bench.mean_force(bench.model), _MODEL, bench.model)
    _line(top, x, bench.mean_force(bench.reference), _REFERENCE, bench.reference)
    top.set_ylabel("Mean |F| per atom (eV/Å)")
    top.set_ylim(bottom=0)
    top.legend(frameon=False, loc="best", labelcolor=_INK2)

    mean, worst = bench.force_error_mean, bench.force_error_max
    _line(bottom, x, mean, _INK2, "mean over atoms")
    bottom.fill_between(x, 0, mean, color=_INK2, alpha=0.08, lw=0)
    bottom.plot(x, worst, color=_MUTED, lw=1.2, ls=(0, (4, 2)), label="worst atom")
    if bench.committee_force_std is not None:
        bottom.plot(x, bench.committee_force_std, color=_MODEL, lw=1.2, ls=(0, (1, 1.5)),
                    label="committee σ, worst atom")
        worst = np.maximum(worst, bench.committee_force_std)
    k = int(np.argmax(mean))
    bottom.annotate(
        f"mean {mean[k]:.2f} eV/Å", (x[k], mean[k]), xytext=(-8, 8), textcoords="offset points",
        ha="right", va="bottom", color=_INK2,
    )
    bottom.set_ylabel("Force error (eV/Å)")
    bottom.set_xlabel(bench.coordinate_label)
    bottom.set_ylim(0, 1.2 * max(worst.max(), 1e-3))
    bottom.legend(frameon=False, loc="upper right", labelcolor=_INK2)
    _mark((top, bottom), mark)
    _descriptor(bottom, bench)
    top.set_title(title or f"{bench.model} vs {bench.reference}: forces along the path",
                  loc="left", color=_INK, fontsize=11)
    figure.savefig(path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(figure)
    return Path(path)


def write_report(
    bench: PathBenchmark,
    prefix: str | Path,
    *,
    title: str | None = None,
    mark: tuple[float, str] | None = None,
    end_labels: tuple[str, str] | None = None,
) -> dict[str, Path]:
    """``<prefix>.csv``, ``<prefix>_energy.png``, and ``<prefix>_forces.png``."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return {
        "csv": bench.write_csv(prefix.with_name(prefix.name + ".csv")),
        "energy": plot_energy(
            bench, prefix.with_name(prefix.name + "_energy.png"), title=title, mark=mark,
            end_labels=end_labels,
        ),
        "forces": plot_forces(
            bench, prefix.with_name(prefix.name + "_forces.png"), title=title, mark=mark
        ),
    }


# --- command line -------------------------------------------------------------------

_COORDINATES = {"irc_arc": "IRC coordinate (Å·amu½)", "arc": "IRC coordinate (Å·amu½)",
                "scan_distance": "scan distance (Å)"}


def main(argv: Sequence[str] | None = None) -> int:
    """``samson-mlip-benchmark``: a model vs a reference along a trajectory file."""
    import argparse

    from ase.io import read

    from .calculators import PROGRAMS, create_calculator
    from .cli import auto_program, backend_settings

    parser = argparse.ArgumentParser(
        prog="samson-mlip-benchmark",
        description="Compare a model with a reference method (PBE via Psi4, xTB, another "
        "MLIP) along a path: an IRC, scan, or band written by samson-mlip --trajectory.",
    )
    parser.add_argument("path", type=Path, help="Multi-frame file readable by ASE (extxyz, traj)")
    parser.add_argument("--model", nargs="+", required=True, help="Model file(s); several MACE "
                        "files form a committee whose spread is also reported")
    parser.add_argument("--backend", default="mace", choices=["mace", "deepmd", "xtb", "psi4"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--reference", default="psi4", choices=["psi4", "xtb", "mace", "deepmd"])
    parser.add_argument("--reference-model", default="auto",
                        help="Reference model file, xtb executable, or Psi4 python ('auto' finds "
                        "xtb or Psi4)")
    parser.add_argument("--points", type=int, default=30, help="Frames to evaluate (spread evenly)")
    parser.add_argument("--angle", default=None, metavar="I-J-K",
                        help="Print the I-J-K angle (at J) under the plots")
    parser.add_argument("--ends", default=None, metavar="A,B", help="Names of the two end points")
    parser.add_argument("--title", default=None)
    parser.add_argument("--labels", default=None, metavar="MODEL,REFERENCE",
                        help="Legend names (default: from the backends)")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="Output prefix (default: <path>_benchmark)")
    parser.add_argument("--xtb-method", default="gfn2", choices=["gfn2", "gfn1", "gfnff"])
    parser.add_argument("--psi4-method", default="pbe")
    parser.add_argument("--basis", default="def2-tzvp")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--solvent", default=None)
    args = parser.parse_args(argv)

    frames = read(args.path, index=":")
    model_files = args.model[0] if len(args.model) == 1 else args.model
    if args.backend in PROGRAMS and args.model == ["auto"]:
        model_files = auto_program(args.backend)
    model = create_calculator(args.backend, model_files, device=args.device, dtype=args.dtype,
                              options=backend_settings(args, args.backend))
    reference_model = args.reference_model
    if args.reference in PROGRAMS and reference_model == "auto":
        reference_model = auto_program(args.reference)
    reference = create_calculator(args.reference, reference_model, device=args.device,
                                  dtype=args.dtype, options=backend_settings(args, args.reference))
    names = (_label(args.backend, args), _label(args.reference, args))
    if args.labels:
        names = tuple(part.strip() for part in args.labels.split(",", 1))
    if names[0] == names[1]:
        names = (names[0], f"{names[1]} (reference)")

    key = next((key for key in _COORDINATES if key in frames[0].info), None)
    coordinate = None if key is None else [frame.info[key] for frame in frames]
    keep, mark = (), None
    if key in ("irc_arc", "arc"):
        ts = int(np.argmin(np.abs(coordinate)))
        keep, mark = (ts,), (coordinate[ts], "TS")
    descriptor = None
    if args.angle:
        triple = tuple(int(part) for part in args.angle.split("-"))
        symbols = frames[0].get_chemical_symbols()
        name = "∠" + "–".join(symbols[index] for index in triple)
        descriptor = (name, bond_angle([f.positions for f in frames], triple))

    bench = benchmark_path(
        frames, model, reference, names=names, coordinate=coordinate,
        coordinate_label=_COORDINATES.get(key, "frame"), points=args.points, keep=keep,
        descriptor=descriptor,
        on_progress=lambda done, total: print(f"frame {done}/{total}", flush=True),
    )
    prefix = args.out or args.path.with_name(args.path.stem + "_benchmark")
    ends = tuple(args.ends.split(",", 1)) if args.ends else None
    written = write_report(bench, prefix, title=args.title, mark=mark, end_labels=ends)
    for name, value in bench.summary().items():
        print(f"{name:40s} {value:.4f}" if isinstance(value, float) else f"{name:40s} {value}")
    for kind, path in written.items():
        print(f"wrote {kind:7s} {path}")
    return 0


def basis_name(basis: str) -> str:
    """Conventional spelling of a basis set name: def2-TZVP, cc-pVTZ, 6-31G*."""
    lower = basis.lower()
    if lower.startswith("def2-"):
        return "def2-" + lower[5:].upper()
    if lower.startswith(("cc-pv", "aug-cc-pv")):
        head, _, tail = lower.rpartition("pv")
        return f"{head}pV{tail.upper()}"
    return basis.upper()


def _label(backend: str, args) -> str:
    if backend == "psi4":
        return f"{args.psi4_method.upper()}/{basis_name(args.basis)}"
    if backend == "xtb":
        return f"{args.xtb_method.upper()}-xTB"
    return backend.upper() if backend == "mace" else "DeepMD"


if __name__ == "__main__":
    raise SystemExit(main())
