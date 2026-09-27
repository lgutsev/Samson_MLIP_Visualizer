"""A reusable, reference-checked active-learning loop for reaction fine-tuning.

Each round: **train** (or reuse) a committee -> **explore** (TS search,
frequencies, IRC, and an optional bond scan) -> **evaluate** against the
reference on frames the model was not trained on -> **stop**, or **select**
new frames -> **label** them with the reference -> add them to the training set.

Built from the existing pieces, with no selection logic of its own:
:func:`.finetune.select_for_labeling` (committee spread + farthest-point
diversity + random spot checks, never spread alone), :func:`.finetune.distances_to`
(how far each test frame is from the training data), :class:`.reference_cache.CachedReference`
(each reference calculation done once), and :func:`.benchmark.benchmark_path`.

The loop stops only on reference-checked criteria: the barrier (model vs the
reference on the same IRC frames), the largest energy error along the IRC, and
the largest along the scan, each within a tolerance. The committee spread is
logged next to the real error every round, and used to *select*, but it never
stops the loop: a committee that starts from one foundation model is often
overconfident (HCN, SN2).

Models plug in through :class:`ModelPlugin`: :class:`MacePlugin` (a committee
of seeds, :func:`.training.train_local`) and :class:`AIMNet2Plugin` (committee
members fine-tuned from different AIMNet2 ensemble members, trained in the
aimnet environment by :mod:`.aimnet2_train_worker`).

Everything a round produces is on disk (``round_NN/``: training set, models,
exploration, evaluation, selection manifest, labels) and summarized in
``rounds.json``; a restarted run skips every stage whose output exists.
``samson-mlip-finetune config.yaml`` runs it from one config file.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write

from .benchmark import _COORDINATES, benchmark_path, write_report
from .finetune import committee_spread, distances_to, select_for_labeling, write_selection

KCAL_PER_EV = 23.0605


# --- plug-ins -----------------------------------------------------------------------


@dataclass
class TrainedModel:
    """What a plug-in returns: the committee members' files and a training log."""

    files: list[str]
    info: dict = field(default_factory=dict)


class ModelPlugin(Protocol):
    name: str

    def train(self, data: Sequence[Atoms], directory: Path,
              previous: TrainedModel | None) -> TrainedModel:
        """Train on ``data`` (``REF_energy_raw``, ``REF_forces``, ``info["charge"]``)
        into ``directory``; return the existing model if it is already there."""
        ...

    def calculator(self, model: TrainedModel, charge: int) -> Calculator:
        """The committee as one calculator (mean energy and forces), with
        ``energy_comm`` / ``forces_comm`` in its results."""
        ...

    def uncertainty(self, model: TrainedModel, frames: Sequence[Atoms],
                    charge: int) -> tuple[np.ndarray, np.ndarray]:
        """Per frame: committee energy std (eV) and worst-atom force std (eV/Å)."""
        ...


class CommitteeCalculator(Calculator):
    """Several calculators as one: mean energy and forces, plus the members'
    values as ``energy_comm`` / ``forces_comm`` (read by the committee-spread code)."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, members: Sequence[Calculator], **kwargs):
        super().__init__(**kwargs)
        if len(members) < 2:
            raise ValueError("A committee needs at least two members")
        self.members = list(members)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energies, forces = [], []
        for member in self.members:
            probe = self.atoms.copy()
            probe.calc = member
            energies.append(float(probe.get_potential_energy()))
            forces.append(probe.get_forces().copy())
        energy = float(np.mean(energies))
        self.results = {"energy": energy, "free_energy": energy,
                        "forces": np.mean(forces, axis=0),
                        "energy_comm": np.array(energies), "forces_comm": np.array(forces)}


def element_offsets(data: Sequence[Atoms], predicted: Sequence[float]) -> dict[int, float]:
    """Per-element offsets δ_Z with E_reference ≈ E_model + Σ n_Z δ_Z (least squares)."""
    elements = sorted({int(z) for a in data for z in a.numbers})
    counts = np.array([[int((a.numbers == z).sum()) for z in elements] for a in data], float)
    residual = np.array([a.info["REF_energy_raw"] for a in data]) - np.asarray(predicted)
    solution = np.linalg.lstsq(counts, residual, rcond=None)[0]
    return dict(zip(elements, solution.tolist(), strict=True))


class MacePlugin:
    """A committee of MACE fine-tunes (seeds) from one foundation model."""

    name = "mace"

    def __init__(self, foundation: str, *, seeds: Sequence[int] = (1, 2, 3), epochs: int = 120,
                 device: str = "cuda", lr: float = 0.005, mode: str = "plain"):
        self.foundation, self.seeds, self.epochs = foundation, tuple(seeds), epochs
        self.device, self.lr, self.mode = device, lr, mode

    def _mace(self, files):
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=[str(f) for f in files] if len(files) > 1
                              else str(files[0]), device=self.device, default_dtype="float64")

    def train(self, data, directory, previous):
        from .training import TrainingSpec, train_local

        directory = Path(directory)
        existing = sorted(directory.glob("runs/seed*/*_seed?.model"))
        if len(existing) == len(self.seeds):
            return TrainedModel([str(p) for p in existing],
                                json.loads((directory / "offsets.json").read_text()))
        directory.mkdir(parents=True, exist_ok=True)
        # Labels onto the foundation model's energy scale, per element.
        foundation = self._mace([self.foundation])
        predicted = []
        for atoms in data:
            probe = atoms.copy()
            probe.calc = foundation
            predicted.append(probe.get_potential_energy())
        offsets = element_offsets(data, predicted)
        images = []
        for atoms in data:
            image = Atoms(atoms.numbers, atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
            image.info["REF_energy"] = atoms.info["REF_energy_raw"] - sum(
                offsets[int(z)] for z in atoms.numbers)
            image.arrays["REF_forces"] = atoms.arrays["REF_forces"]
            images.append(image)
        write(directory / "train_foundation_scale.extxyz", images)
        (directory / "offsets.json").write_text(json.dumps({"offsets_ev": offsets}))
        spec = TrainingSpec(name=f"al_{directory.parent.name}", foundation=self.foundation,
                            train_file=str(directory / "train_foundation_scale.extxyz"),
                            seeds=self.seeds, epochs=self.epochs, lr=self.lr,
                            device=self.device, mode=self.mode)
        runs = train_local(spec, directory / "runs")
        failed = [run for run in runs if not run.ok]
        if failed:
            raise RuntimeError(f"MACE training failed, see {failed[0].log}")
        return TrainedModel([str(run.model) for run in runs], {"offsets_ev": offsets})

    def calculator(self, model, charge):
        return self._mace(model.files)  # MACE has no charge input

    def uncertainty(self, model, frames, charge):
        return committee_spread(frames, self.calculator(model, charge))


class AIMNet2Plugin:
    """A committee of AIMNet2 fine-tunes, each started from a different AIMNet2
    ensemble member (full-batch training from one start would give identical
    seeds), trained in the aimnet environment in parallel subprocesses."""

    name = "aimnet2"
    WORKER = Path(__file__).with_name("aimnet2_train_worker.py")

    def __init__(self, python: str, members: Sequence[str], *, epochs: int = 800,
                 warm_start_epochs: int = 300, lr: float = 5e-5, e_scale: float = 0.010,
                 f_scale: float = 0.050):
        if len(members) < 2:
            raise ValueError("The AIMNet2 committee needs at least two starting members")
        self.python, self.members = str(python), [str(m) for m in members]
        self.epochs, self.warm_start_epochs = epochs, warm_start_epochs
        self.lr, self.e_scale, self.f_scale = lr, e_scale, f_scale

    def train(self, data, directory, previous):
        from .worker_process import worker_environment

        directory = Path(directory)
        files = [directory / f"member{k}.pt" for k in range(len(self.members))]
        if all(f.exists() for f in files):
            logs = [json.loads(f.with_suffix(".log.json").read_text()) for f in files]
            return TrainedModel([str(f) for f in files], {"logs": logs})
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / "train.extxyz", list(data))
        # The members train side by side: give each an equal share of the cores.
        environment = worker_environment()
        environment["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 2) // len(self.members)))
        processes = []
        for k, (member, out) in enumerate(zip(self.members, files, strict=True)):
            start = previous.files[k] if previous else member
            job = {"base_model": start, "out_model": str(out), "data": str(directory /
                   "train.extxyz"), "epochs": self.warm_start_epochs if previous else self.epochs,
                   "lr": self.lr, "e_scale": self.e_scale, "f_scale": self.f_scale, "seed": k,
                   "log": str(out.with_suffix(".log.json"))}
            job_file = directory / f"member{k}.job.json"
            job_file.write_text(json.dumps(job, indent=1))
            log = open(directory / f"member{k}.stdout", "w", encoding="utf-8")  # noqa: SIM115
            processes.append((subprocess.Popen(
                [self.python, "-u", str(self.WORKER), str(job_file)], stdout=log,
                stderr=subprocess.STDOUT, env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)), log, k))
        for process, log, k in processes:
            code = process.wait()
            log.close()
            if code != 0 or not files[k].exists():
                tail = (directory / f"member{k}.stdout").read_text(errors="replace")[-1500:]
                raise RuntimeError(f"AIMNet2 training of member {k} failed:\n{tail}")
        logs = [json.loads(f.with_suffix(".log.json").read_text()) for f in files]
        return TrainedModel([str(f) for f in files], {"logs": logs})

    def calculator(self, model, charge):
        from .aimnet2_backend import AIMNet2Calculator

        return CommitteeCalculator([AIMNet2Calculator(self.python, model=f, charge=charge)
                                    for f in model.files])

    def uncertainty(self, model, frames, charge):
        return committee_spread(frames, self.calculator(model, charge))


# --- configuration ------------------------------------------------------------------


@dataclass
class Config:
    output: str
    model: dict
    reference: dict
    seed_data: list[str]
    explore: dict
    system: dict = field(default_factory=lambda: {"charge": 0, "multiplicity": 1})
    tolerances: dict = field(default_factory=lambda: {
        "barrier_kcal": 0.5, "path_max_ev": 0.010, "scan_max_ev": 0.020})
    budget: dict = field(default_factory=lambda: {
        "committee": 4, "diverse": 3, "random": 2, "rattle_copies": 1, "rattle_sigma": 0.03,
        "min_distance": 0.05, "max_rounds": 4})

    @property
    def charge(self) -> int:
        return int(self.system.get("charge", 0))

    @classmethod
    def load(cls, path: str | Path) -> Config:
        text = Path(path).read_text(encoding="utf-8")
        if str(path).lower().endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        defaults = cls(output="", model={}, reference={}, seed_data=[], explore={})
        data["tolerances"] = {**defaults.tolerances, **data.get("tolerances", {})}
        data["budget"] = {**defaults.budget, **data.get("budget", {})}
        return cls(**data)


def make_plugin(model: dict) -> ModelPlugin:
    options = {k: v for k, v in model.items() if k != "plugin"}
    if model["plugin"] == "mace":
        return MacePlugin(**options)
    if model["plugin"] == "aimnet2":
        return AIMNet2Plugin(**options)
    raise ValueError(f"Unknown model plug-in {model['plugin']!r}; choose mace or aimnet2")


def make_reference(reference: dict, charge: int):
    from .reference_cache import CachedReference, psi4_factory

    if reference.get("program", "psi4") != "psi4":
        raise ValueError("Only program: psi4 is supported as a reference here")
    factory = psi4_factory(reference["method"], reference["basis"],
                           threads=int(reference.get("threads", 8)),
                           memory_mb=int(reference.get("memory_mb", 1900)),
                           python=reference.get("python"))
    return CachedReference(reference["cache"], factory, charge=charge)


def load_labeled(paths: Sequence[str | Path], charge: int) -> list[Atoms]:
    """Labeled structures with ``REF_energy_raw`` (also read from ``raw_energy``),
    ``REF_forces``, and ``info["charge"]`` (default: the system's)."""
    images = []
    for path in paths:
        for atoms in read(path, ":"):
            raw = atoms.info.get("REF_energy_raw", atoms.info.get("raw_energy"))
            if raw is None or "REF_forces" not in atoms.arrays:
                raise ValueError(f"{path}: every structure needs a raw reference energy "
                                 "(REF_energy_raw) and REF_forces")
            image = Atoms(atoms.numbers, atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
            image.info.update({"REF_energy_raw": float(raw),
                               "charge": int(atoms.info.get("charge", charge)),
                               "tag": atoms.info.get("tag", "seed")})
            if "weight" in atoms.info:
                image.info["weight"] = float(atoms.info["weight"])
            image.arrays["REF_forces"] = np.asarray(atoms.arrays["REF_forces"], float)
            images.append(image)
    return images


# --- exploration --------------------------------------------------------------------


@dataclass
class Exploration:
    ts_positions: np.ndarray
    ts: dict  # converged, imaginary mode, n_imaginary, energy
    irc: list[Atoms]  # info["irc_arc"]
    scan: list[Atoms] = field(default_factory=list)  # info["scan_distance"]
    scan_jumps: list[int] = field(default_factory=list)


def explore(calc, numbers, start_positions, options: dict) -> Exploration:
    """P-RFO (exact Hessian) from ``start_positions``, frequencies, IRC both ways,
    and (with ``options["scan"]``) a constrained bond scan from the IRC end whose
    pair distance is closest to the scan's start."""
    from .reaction_path import irc, scan_to_ts
    from .ts import prfo_search
    from .vibrations import harmonic_frequencies

    ts = Atoms(numbers, start_positions)
    ts.calc = calc
    search = prfo_search(ts, fmax=float(options.get("ts_fmax", 0.005)), exact_hessian=True)
    frequencies = harmonic_frequencies(ts)
    irc_options = options.get("irc", {})
    result = irc(ts, step=float(irc_options.get("step", 0.05)),
                 max_steps=int(irc_options.get("max_steps", 400)),
                 fmax=float(irc_options.get("fmax", 0.01)), relax_ends=True)
    frames = [Atoms(numbers, f.positions, info={"irc_arc": float(f.arc)})
              for f in result.frames(ts.get_positions())]
    found = Exploration(ts.get_positions().copy(), {
        "converged": bool(search.converged), "energy_ev": float(ts.get_potential_energy()),
        "imaginary_cm": float(frequencies.wavenumbers_cm[0]),
        "n_imaginary": int(frequencies.n_imaginary)}, frames)
    scan = options.get("scan")
    if scan:
        i, j = scan["pair"]
        ends = (frames[0], frames[-1])
        end = min(ends, key=lambda a: abs(a.get_distance(i, j) - float(scan["start"])))
        atoms = Atoms(numbers, end.positions)
        atoms.calc = calc
        scanned = scan_to_ts(atoms, (i, j), start=float(scan["start"]), stop=float(scan["stop"]),
                             points=int(scan.get("points", 15)),
                             relax_fmax=float(scan.get("relax_fmax", 0.01)), refine=False)
        found.scan = [Atoms(numbers, p, info={"scan_distance": float(d)})
                      for p, d in zip(scanned.frames, scanned.distances, strict=True)]
        found.scan_jumps = list(scanned.jumps)
    return found


# --- the loop -----------------------------------------------------------------------


def barrier_from_higher_end(relative: np.ndarray) -> float:
    """The highest point above the higher of the two ends (eV): the forward
    barrier of an exothermic path, whichever end comes first."""
    return float(relative.max() - max(relative[0], relative[-1]))


def stop_decision(evaluation: dict, tolerances: dict) -> tuple[bool, list[str]]:
    """Reference-checked only: every tolerance met. Returns (stop, reasons it did not)."""
    failed = []
    if abs(evaluation["barrier_error_kcal"]) > tolerances["barrier_kcal"]:
        failed.append(f"barrier off by {evaluation['barrier_error_kcal']:+.2f} kcal/mol "
                      f"(tolerance {tolerances['barrier_kcal']})")
    if evaluation["irc"]["energy_error_max_abs_ev"] > tolerances["path_max_ev"]:
        failed.append(f"IRC max error {1000 * evaluation['irc']['energy_error_max_abs_ev']:.1f}"
                      f" meV (tolerance {1000 * tolerances['path_max_ev']:.0f})")
    scan = evaluation.get("scan")
    if scan and scan["energy_error_max_abs_ev"] > tolerances["scan_max_ev"]:
        failed.append(f"scan max error {1000 * scan['energy_error_max_abs_ev']:.1f} meV "
                      f"(tolerance {1000 * tolerances['scan_max_ev']:.0f})")
    return not failed, failed


class ActiveLearning:
    """The round loop. ``explorer`` defaults to :func:`explore` (tests pass a stand-in)."""

    def __init__(self, config: Config, plugin: ModelPlugin, reference: Calculator, *,
                 explorer: Callable[..., Exploration] = explore, log=print):
        self.config, self.plugin, self.reference = config, plugin, reference
        self.explorer, self.log = explorer, log
        self.output = Path(config.output)
        self.rounds_file = self.output / "rounds.json"

    # -- bookkeeping --

    def rounds(self) -> list[dict]:
        return json.loads(self.rounds_file.read_text()) if self.rounds_file.exists() else []

    def _save_row(self, row: dict) -> None:
        rows = [r for r in self.rounds() if r["round"] != row["round"]] + [row]
        self.rounds_file.write_text(json.dumps(sorted(rows, key=lambda r: r["round"]), indent=1))

    def _seed(self) -> list[Atoms]:
        data = load_labeled(self.config.seed_data, self.config.charge)
        # The seed labels are reference results already: never compute them twice.
        add = getattr(self.reference, "add", None)
        if add is not None:
            for atoms in data:
                if atoms.info["charge"] == self.config.charge:
                    add(atoms.numbers, atoms.positions, atoms.info["REF_energy_raw"],
                        atoms.arrays["REF_forces"])
            self.reference.save()
        return data

    # -- one round --

    def run(self, max_rounds: int | None = None) -> list[dict]:
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "config.json").write_text(json.dumps(asdict(self.config), indent=1))
        max_rounds = int(max_rounds or self.config.budget["max_rounds"])
        previous_model, start = None, None
        for number in range(max_rounds):
            row = next((r for r in self.rounds() if r["round"] == number), None)
            directory = self.output / f"round_{number:02d}"
            if row and (row["status"] == "converged" or row["status"].startswith("stopped")):
                self.log(f"round {number}: {row['status']} earlier; nothing to do")
                return self.rounds()
            data = self._training_set(number)
            began = time.perf_counter()
            model = self.plugin.train(data, directory / "model", previous_model)
            calc = self.plugin.calculator(model, self.config.charge)
            exploration = self._explore(directory, calc, start)
            evaluation = self._evaluate(directory, calc, exploration, data)
            stop, failed = stop_decision(evaluation, self.config.tolerances)
            row = {"round": number, "training_structures": len(data),
                   "ts": exploration.ts, "scan_jumps": exploration.scan_jumps, **evaluation,
                   "failed": failed}
            if stop:
                row.update(status="converged", wall_s=round(time.perf_counter() - began, 1))
                self._save_row(row)
                self.log(f"round {number}: all tolerances met; stopping")
                return self.rounds()
            selection = self._select(directory, model, exploration, evaluation, data)
            if not selection["frames"]:
                # Every candidate is within min_distance of the data: more rounds of
                # the same exploration cannot add anything.
                row.update(status="stopped: nothing new to label", selected=selection["summary"],
                           labeled=0, wall_s=round(time.perf_counter() - began, 1))
                self._save_row(row)
                self.log(f"round {number}: {'; '.join(failed)}, but no candidate is new; stopping")
                return self.rounds()
            labeled = self._label(directory, number, selection)
            row.update(status="continued", selected=selection["summary"],
                       labeled=len(labeled), wall_s=round(time.perf_counter() - began, 1))
            self._save_row(row)
            self.log(f"round {number}: {'; '.join(failed)} -> labeled {len(labeled)} frames")
            previous_model, start = model, exploration.ts_positions
        rows = self.rounds()
        if rows and rows[-1]["status"] != "converged":
            rows[-1]["status"] = "not converged: max rounds reached"
            self._save_row(rows[-1])
        return self.rounds()

    def _training_set(self, number: int) -> list[Atoms]:
        path = self.output / f"round_{number:02d}" / "train.extxyz"
        if path.exists():
            return load_labeled([path], self.config.charge)
        if number == 0:
            data = self._seed()
        else:
            before = self.output / f"round_{number - 1:02d}"
            data = load_labeled([before / "train.extxyz", before / "labeled.extxyz"],
                                self.config.charge)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, data)
        return data

    def _explore(self, directory: Path, calc, start) -> Exploration:
        folder = directory / "explore"
        summary = folder / "explore.json"
        if summary.exists():
            info = json.loads(summary.read_text())
            scan = read(folder / "scan.extxyz", ":") if (folder / "scan.extxyz").exists() else []
            return Exploration(np.array(info["ts_positions"]), info["ts"],
                               read(folder / "irc.extxyz", ":"), scan, info["scan_jumps"])
        folder.mkdir(parents=True, exist_ok=True)
        if start is None:
            start_atoms = read(self.config.explore["ts_start"])
            numbers, start = start_atoms.numbers, start_atoms.positions
        else:
            numbers = read(self.config.explore["ts_start"]).numbers
        found = self.explorer(calc, numbers, start, self.config.explore)
        write(folder / "irc.extxyz", found.irc)
        if found.scan:
            write(folder / "scan.extxyz", found.scan)
        summary.write_text(json.dumps({"ts_positions": np.asarray(found.ts_positions).tolist(),
                                       "ts": found.ts, "scan_jumps": found.scan_jumps},
                                      indent=1))
        return found

    def _benchmark(self, name, frames, calc, key, data, points, folder):
        x = [f.info[key] for f in frames]
        keep = (int(np.argmin(np.abs(x))),) if key == "irc_arc" else ()
        bench = benchmark_path(frames, calc, self.reference, names=("model", "reference"),
                               coordinate=x, coordinate_label=_COORDINATES[key], points=points,
                               keep=keep)
        write_report(bench, folder / f"{name}_vs_reference",
                     mark=(0.0, "TS") if key == "irc_arc" else None,
                     title=f"Committee vs reference: {name}")
        summary = bench.summary()
        rmsd = distances_to([frames[k] for k in bench.frame_indices], data)
        error = np.abs(bench.energy_error)
        result = {
            "frames": summary["frames"], "frame_indices": list(bench.frame_indices),
            "energy_error_max_abs_ev": float(error.max()),
            "energy_rmse_ev": summary["energy_error_rmse_ev"],
            "force_rmse_ev_per_A": summary["force_rmse_ev_per_angstrom"],
            "worst_atom_force_error_ev_per_A": summary["force_error_max_ev_per_angstrom"],
            "rmsd_to_training_A": {"median": float(np.median(rmsd)), "max": float(np.max(rmsd))},
        }
        if bench.committee_energy_std is not None:
            spread = np.asarray(bench.committee_energy_std, float)
            result["committee_energy_std_max_ev"] = float(spread.max())
            result["error_over_spread_at_worst_frame"] = float(
                error.max() / max(spread[int(np.argmax(error))], 1e-9))
            if len(error) > 2 and np.std(spread) > 0 and np.std(error) > 0:
                result["error_spread_correlation"] = float(np.corrcoef(error, spread)[0, 1])
        return bench, result

    def _evaluate(self, directory, calc, exploration, data) -> dict:
        folder = directory / "evaluate"
        path = folder / "evaluation.json"
        if path.exists():
            return json.loads(path.read_text())
        folder.mkdir(parents=True, exist_ok=True)
        before = getattr(self.reference, "computed", 0)
        points = int(self.config.explore.get("held_out_points", 15))
        bench, irc_result = self._benchmark("irc", exploration.irc, calc, "irc_arc", data,
                                            points, folder)
        barrier = {n: barrier_from_higher_end(bench.relative(n)) * KCAL_PER_EV
                   for n in bench.names}
        evaluation = {"irc": irc_result, "barrier_model_kcal": barrier["model"],
                      "barrier_reference_kcal": barrier["reference"],
                      "barrier_error_kcal": barrier["model"] - barrier["reference"]}
        if exploration.scan:
            # About half the scan frames (evenly spread, both ends kept) are held out;
            # the rest stay candidates, so off-path frames can be selected.
            scan_points = int(self.config.explore.get(
                "held_out_scan_points", (len(exploration.scan) + 1) // 2))
            _, evaluation["scan"] = self._benchmark("scan", exploration.scan, calc,
                                                    "scan_distance", data, scan_points, folder)
        evaluation["new_reference_calculations"] = getattr(self.reference, "computed", 0) - before
        path.write_text(json.dumps(evaluation, indent=1))
        return evaluation

    def _select(self, directory, model, exploration, evaluation, data) -> dict:
        folder = directory / "select"
        manifest = folder / "manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            return {"frames": read(folder / "frames.extxyz", ":"),
                    "summary": saved["notes"]["summary"]}
        # Candidates: every explored frame except the held-out ones just evaluated.
        held = {("irc", k) for k in evaluation["irc"]["frame_indices"]}
        held |= {("scan", k) for k in evaluation.get("scan", {}).get("frame_indices", [])}
        candidates, sources = [], []
        for source, frames in (("irc", exploration.irc), ("scan", exploration.scan)):
            for k, frame in enumerate(frames):
                if (source, k) not in held:
                    candidates.append(frame)
                    sources.append(f"{source} frame {k}")
        budget = self.config.budget
        if not candidates:
            return {"frames": [], "summary": {"selected": 0, "note": "no candidates"}}
        _, force_std = self.plugin.uncertainty(model, candidates, self.config.charge)
        selection = select_for_labeling(
            candidates, spread=force_std, n_committee=int(budget["committee"]),
            n_diverse=int(budget["diverse"]), n_random=int(budget["random"]), labeled=data,
            min_distance=float(budget["min_distance"]), seed=int(directory.name[-2:]))
        summary = {"selected": len(selection), "candidates": len(candidates),
                   "by_reason": {r: selection.reasons.count(r) for r in set(selection.reasons)},
                   "frames": [{"source": sources[i], "reason": r, "score": s,
                               "nearest_training_A": d}
                              for i, r, s, d in zip(selection.indices, selection.reasons,
                                                    selection.scores,
                                                    selection.nearest_distance, strict=True)]}
        write_selection(folder, candidates, selection, source=directory.name,
                        model=self.plugin.name, notes={"summary": summary})
        return {"frames": [candidates[i] for i in selection.indices], "summary": summary}

    def _label(self, directory, number, selection) -> list[Atoms]:
        path = directory / "labeled.extxyz"
        if path.exists():
            return read(path, ":")
        budget = self.config.budget
        rng = np.random.default_rng(number)
        todo = []
        for frame, info in zip(selection["frames"], selection["summary"].get("frames", []),
                               strict=True):
            tag = f"round {number}: {info['reason']} ({info['source']})"
            todo.append((frame, tag))
            for _ in range(int(budget["rattle_copies"])):
                rattled = Atoms(frame.numbers, frame.positions + rng.normal(
                    0, float(budget["rattle_sigma"]), frame.positions.shape))
                todo.append((rattled, tag + ", rattled"))
        labeled = []
        for frame, tag in todo:
            probe = Atoms(frame.numbers, frame.positions)
            probe.calc = self.reference
            image = Atoms(frame.numbers, frame.positions)
            image.info.update({"REF_energy_raw": float(probe.get_potential_energy()),
                               "charge": self.config.charge, "tag": tag})
            image.arrays["REF_forces"] = probe.get_forces().copy()
            labeled.append(image)
        write(path, labeled)
        return labeled


# --- command line -------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``samson-mlip-finetune config.yaml``: run (or resume) the loop."""
    import argparse

    parser = argparse.ArgumentParser(prog="samson-mlip-finetune",
                                     description="Reference-checked active learning for a "
                                     "reaction (see samson_mlip_visualizer.active_learning).")
    parser.add_argument("config", type=Path, help="JSON or YAML config")
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    loop = ActiveLearning(config, make_plugin(config.model),
                          make_reference(config.reference, config.charge))
    rows = loop.run(args.max_rounds)
    print(round_table(rows))
    return 0


def round_table(rows: Sequence[dict[str, Any]]) -> str:
    """A Markdown table of the rounds."""
    lines = ["| Round | Training | Barrier model / ref (kcal/mol) | IRC max err (meV) | "
             "Scan max err (meV) | Error / spread at the worst frame | Labeled | Status |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        scan = r.get("scan")
        scan_cell = f"{1000 * scan['energy_error_max_abs_ev']:.1f}" if scan else "—"
        # The path with the larger error says more about the committee.
        worst = scan if scan else r["irc"]
        ratio = worst.get("error_over_spread_at_worst_frame")
        ratio_cell = f"{ratio:.1f}× ({'scan' if scan else 'IRC'})" if ratio else "—"
        cells = [str(r["round"]), str(r["training_structures"]),
                 f"{r['barrier_model_kcal']:.2f} / {r['barrier_reference_kcal']:.2f}",
                 f"{1000 * r['irc']['energy_error_max_abs_ev']:.1f}", scan_cell, ratio_cell,
                 str(r.get("labeled", 0)), r["status"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
