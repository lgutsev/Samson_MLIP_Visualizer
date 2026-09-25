"""MLIP jobs started through the bridge: single point, relax, MD, TS search, frequencies.

A job runs on SAMSON's main thread like a panel job, but it starts from a
scheduled callback after the request that created it has been answered. Its
progress callbacks pump SAMSON's event loop, so SAMSON keeps repainting and the
bridge keeps answering ``job.status``, ``job.stop``, and read requests.
"""

from __future__ import annotations

import contextlib
import itertools
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..calculators import create_calculator
from ..compat import assert_model_covers_structure
from ..engine import evaluate, relax
from ..md import ENSEMBLES, md_warnings, parse_pairs, run_md
from ..provenance import collect_provenance
from ..samson_bridge import choose_structural_models, extract_structure, sync_positions
from ..ts import dimer_search, prfo_search
from ..vibrations import harmonic_frequencies

KINDS = ("single_point", "relax", "md", "ts", "frequencies", "irc", "qst", "scan")
_MOVES_ATOMS = ("relax", "md", "ts", "irc", "scan")
_MISSING = object()


class JobSpecError(ValueError):
    """A job request with a missing or malformed parameter."""


class _Stopped(Exception):
    pass


def _get(params: dict[str, Any], name: str, kind: type | tuple, default: Any = _MISSING) -> Any:
    if name not in params or params[name] is None:
        if default is _MISSING:
            raise JobSpecError(f"Missing job parameter {name!r}")
        return default
    value = params[name]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if isinstance(value, bool) and bool not in kinds or not isinstance(value, kinds):
        names = " or ".join(k.__name__ for k in kinds)
        raise JobSpecError(f"Job parameter {name!r} must be {names}")
    return value


def _number(params: dict[str, Any], name: str, default: float) -> float:
    return float(_get(params, name, (int, float), default))


def _finite(value: float) -> float | None:
    """JSON has no NaN or infinity; report them as null."""
    return float(value) if np.isfinite(value) else None


def _publish_path(structure, frames, name: str) -> dict[str, Any]:
    from ..samson_modes import add_frames_path

    add_frames_path(structure, frames, name=name)
    return {"path": name, "path_frames": len(frames)}


@dataclass
class Job:
    id: int
    kind: str
    params: dict[str, Any]
    state: str = "queued"  # queued | running | finished | stopped | failed
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    step: int = 0
    log: deque = field(default_factory=lambda: deque(maxlen=500))
    result: dict[str, Any] | None = None
    error: str | None = None
    stop_requested: bool = False

    def to_dict(self, log_lines: int = 20) -> dict[str, Any]:
        end = self.finished or time.time()
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "step": self.step,
            "elapsed_s": round(end - self.started, 3) if self.started else 0.0,
            "log": list(self.log)[-log_lines:] if log_lines > 0 else [],
            "result": self.result,
            "error": self.error,
            "params": self.params,
        }


class JobManager:
    """Queue, run, and report MLIP jobs. One job runs at a time."""

    def __init__(
        self,
        get_samson: Callable[[], Any],
        *,
        schedule: Callable[[Callable[[], None]], None] | None = None,
        set_busy: Callable[[str | None], None] = lambda reason: None,
        defaults: Callable[[], dict[str, Any]] = dict,
    ):
        self._get_samson = get_samson
        self._schedule = schedule or (lambda function: function())
        self._set_busy = set_busy
        self._defaults = defaults
        self._jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._calculators: dict[tuple, Any] = {}

    # --- public API used by the dispatcher -------------------------------------------

    @property
    def active(self) -> Job | None:
        live = (job for job in self._jobs.values() if job.state in ("queued", "running"))
        return next(live, None)

    def start(self, kind: str, params: dict[str, Any]) -> Job:
        if kind not in KINDS:
            raise JobSpecError(f"Unknown job kind {kind!r}; choose one of {', '.join(KINDS)}")
        if self.active is not None:
            raise JobSpecError(f"Job {self.active.id} is still {self.active.state}")
        merged = {**self._defaults(), **params}
        _get(merged, "model", (str, list))
        job = Job(next(self._ids), kind, merged)
        self._jobs[job.id] = job
        self._set_busy(f"bridge job {job.id} ({kind}) is running")
        self._schedule(lambda: self._run(job))
        return job

    def get(self, job_id: int) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobSpecError(f"No job with id {job_id}") from exc

    def jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def stop(self, job_id: int) -> Job:
        job = self.get(job_id)
        if job.state in ("queued", "running"):
            job.stop_requested = True
            job.log.append("Stop requested; finishing the current MLIP evaluation…")
        return job

    # --- running ---------------------------------------------------------------------

    def _calculator(self, params: dict[str, Any]):
        model = params["model"]
        paths = [model] if isinstance(model, str) else list(model)
        backend = _get(params, "backend", str, "mace").lower()
        device = _get(params, "device", str, "cpu")
        dtype = _get(params, "dtype", str, "float64")
        key = (backend, tuple(paths), device, dtype)
        if key not in self._calculators:
            argument = paths[0] if len(paths) == 1 else paths
            # Keep one calculator: a second MACE model would double GPU memory.
            self._calculators = {
                key: create_calculator(backend, argument, device=device, dtype=dtype)
            }
        provenance = collect_provenance(
            backend=backend, model_path=paths[0], device=device, dtype=dtype
        ).as_dict()
        return self._calculators[key], provenance

    def _run(self, job: Job) -> None:
        samson = self._get_samson()
        job.state, job.started = "running", time.time()
        try:
            if job.stop_requested:
                raise _Stopped
            calculator, provenance = self._calculator(job.params)
            which = _get(job.params, "models", str, "auto")
            models = None
            if job.kind == "qst":
                # Reactant, (guess,) product: the selected models in document order.
                selected = choose_structural_models(samson)
                if len(selected) not in (2, 3):
                    raise JobSpecError(
                        "qst needs 2 (QST2) or 3 (QST3) selected structural models: reactant, "
                        f"(guess,) product in document order; {len(selected)} are selected"
                    )
                models = selected[:1]
                self._endpoints = [
                    extract_structure(samson, models=[model]).ase_atoms for model in selected[1:]
                ]
            elif which == "all":
                models = list(samson.getNodes("node.type structuralModel"))
            elif which != "auto":
                raise JobSpecError("models must be 'auto' or 'all'")
            structure = extract_structure(samson, models=models)
            atoms = structure.ase_atoms
            atoms.calc = calculator
            assert_model_covers_structure(calculator, atoms)
            job.log.append(f"{job.kind}: {len(atoms)} atoms from {len(structure.models)} model(s)")
            self._structure = structure
            self._publish: list[Callable[[], dict[str, Any]]] = []

            def pump() -> bool:
                with contextlib.suppress(Exception):
                    samson.processEvents()
                return job.stop_requested

            start = atoms.get_positions().copy()
            holding = getattr(samson, "holding", None)

            def show(positions) -> None:
                # Open the undo step on the first real move: SAMSON records nothing
                # for an empty one, and a later undo would revert an older change.
                if not undo_open and np.array_equal(positions, start):
                    return
                if not undo_open and callable(holding):
                    undo.enter_context(holding(f"Remote MLIP {job.kind}"))
                    undo_open.append(True)
                sync_positions(structure, positions, samson=samson, process_events=False)

            undo_open: list[bool] = []
            with contextlib.ExitStack() as undo:
                result = getattr(self, f"_{job.kind}")(job, atoms, pump, show)
                if job.kind in _MOVES_ATOMS:
                    show(atoms.get_positions())
            result["moved_atoms"] = not np.array_equal(atoms.get_positions(), start)
            # SAMSON nodes (paths, new models) are added after the job's undo step closes,
            # each as its own undo step; failures are reported without failing the job.
            for publish in self._publish:
                try:
                    result.update(publish())
                except Exception as exc:  # noqa: BLE001
                    result.setdefault("publish_errors", []).append(f"{type(exc).__name__}: {exc}")
            job.result = {**result, "provenance": provenance}
            job.state = "stopped" if result.get("stopped") else "finished"
        except _Stopped:
            job.state = "stopped"
        except Exception as exc:  # noqa: BLE001 - reported through job.status
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.extend(traceback.format_exc().strip().splitlines()[-3:])
        finally:
            job.finished = time.time()
            job.log.append(f"Job {job.state} after {job.finished - job.started:.1f} s")
            self._set_busy(None)

    def _single_point(self, job, atoms, pump, show) -> dict[str, Any]:
        result = evaluate(atoms)
        job.log.append(
            f"E = {result.energy_ev:.8f} eV, Fmax = {result.max_force_ev_per_angstrom:.6f} eV/Å"
        )
        return {
            "energy_ev": result.energy_ev,
            "max_force_ev_per_angstrom": result.max_force_ev_per_angstrom,
            "energy_std_ev": result.energy_std_ev,
            "max_force_std_ev_per_angstrom": result.max_force_std_ev_per_angstrom,
        }

    def _relax(self, job, atoms, pump, show) -> dict[str, Any]:
        params = job.params

        def progress(step, energy, max_force, positions):
            job.step = step
            job.log.append(f"step {step:5d}  E {energy:.8f} eV  Fmax {max_force:.6f} eV/Å")
            show(positions)

        result = relax(
            atoms,
            fmax=_number(params, "fmax", 0.05),
            max_steps=int(_number(params, "max_steps", 250)),
            optimizer=_get(params, "optimizer", str, "FIRE"),
            min_distance=_number(params, "min_distance", 0.5) or None,
            max_force_std=_number(params, "max_force_std", 0.0) or None,
            on_progress=progress,
            should_stop=pump,
        )
        return {
            "steps": result.steps,
            "converged": result.converged,
            "stopped": result.stopped,
            "energy_ev": result.evaluation.energy_ev,
            "max_force_ev_per_angstrom": result.evaluation.max_force_ev_per_angstrom,
        }

    def _md(self, job, atoms, pump, show) -> dict[str, Any]:
        params = job.params
        ensemble = _get(params, "ensemble", str, "Langevin")
        if ensemble not in ENSEMBLES:
            raise JobSpecError(f"ensemble must be one of {', '.join(ENSEMBLES)}")
        timestep = _number(params, "timestep_fs", 0.5)
        for message in md_warnings(
            atoms, timestep_fs=timestep, ensemble=ensemble, dtype=params.get("dtype")
        ):
            job.log.append(f"Warning: {message}")
        constraints = parse_pairs(_get(params, "fixed_distances", str, ""))

        def progress(frame):
            job.step = frame.step
            forces = "".join(f"  f={value:+.3f}" for value in frame.constraint_forces)
            job.log.append(
                f"step {frame.step:6d}  {frame.time_fs:9.1f} fs  Epot {frame.potential_ev:.5f}  "
                f"Etot {frame.total_ev:.5f} eV  T {frame.temperature_k:7.1f} K{forces}"
            )
            show(frame.positions)

        seed = _get(params, "seed", int, None)
        result = run_md(
            atoms,
            ensemble=ensemble,
            temperature_k=_number(params, "temperature_k", 300.0),
            timestep_fs=timestep,
            steps=int(_number(params, "steps", 1000)),
            friction_per_fs=_number(params, "friction_per_fs", 0.01),
            tdamp_fs=_number(params, "tdamp_fs", 100.0),
            seed=seed,
            distance_constraints=constraints,
            report_interval=int(_number(params, "report_interval", 10)),
            min_distance=_number(params, "min_distance", 0.5) or None,
            max_force_std=_number(params, "max_force_std", 0.0) or None,
            max_temperature_k=_number(params, "max_temperature_k", 0.0) or None,
            trajectory=_get(params, "trajectory", str, None),
            trajectory_interval=int(_number(params, "report_interval", 10)),
            on_progress=progress,
            should_stop=pump,
        )
        return {
            "steps": result.steps,
            "time_fs": result.time_fs,
            "stopped": result.stopped,
            "mean_temperature_k": result.mean_temperature_k,
            "energy_drift_mev_per_atom_ps": result.energy_drift_mev_per_atom_ps,
            "constraint_forces": [vars(summary) for summary in result.constraint_forces],
        }

    def _ts(self, job, atoms, pump, show) -> dict[str, Any]:
        params = job.params
        method = _get(params, "method", str, "prfo")
        if method not in ("prfo", "dimer"):
            raise JobSpecError("method must be 'prfo' or 'dimer'")
        if method == "prfo":
            return self._ts_summary(
                job,
                atoms,
                pump,
                show,
                prfo_search(
                    atoms,
                    fmax=_number(params, "fmax", 0.01),
                    max_steps=int(_number(params, "max_steps", 500)),
                    exact_hessian=_get(params, "exact_hessian", bool, False),
                    recompute_every=_get(params, "recompute_every", int, None),
                    min_distance=_number(params, "min_distance", 0.5) or None,
                    on_progress=self._ts_progress(job, show),
                    should_stop=pump,
                ),
            )
        pair_text = _get(params, "pair", str, "")
        start = _get(params, "start", str, "pair" if pair_text else "hessian")
        if start not in ("hessian", "pair", "random"):
            raise JobSpecError("start must be 'hessian', 'pair' or 'random'")
        pair = None
        if start == "pair":
            parsed = parse_pairs(pair_text)
            if len(parsed) != 1:
                raise JobSpecError("start='pair' needs pair='I-J'")
            pair = (parsed[0].i, parsed[0].j)

        result = dimer_search(
            atoms,
            fmax=_number(params, "fmax", 0.01),
            max_steps=int(_number(params, "max_steps", 500)),
            pair=pair,
            use_hessian=start == "hessian",
            displacement=_number(params, "displacement", 0.05),
            seed=_get(params, "seed", int, None),
            min_distance=_number(params, "min_distance", 0.5) or None,
            on_progress=self._ts_progress(job, show),
            should_stop=pump,
        )
        return self._ts_summary(job, atoms, pump, show, result)

    def _ts_progress(self, job, show):
        def progress(step, energy, max_force, curvature, positions):
            job.step = step
            job.log.append(
                f"step {step:4d}  E {energy:.8f} eV  Fmax {max_force:.6f}  "
                f"curvature {curvature:+.4f}"
            )
            show(positions)

        return progress

    def _ts_summary(self, job, atoms, pump, show, result) -> dict[str, Any]:
        summary = {
            "method": job.params.get("method", "prfo"),
            "steps": result.steps,
            "converged": result.converged,
            "stopped": result.stopped,
            "energy_ev": result.evaluation.energy_ev,
            "max_force_ev_per_angstrom": result.evaluation.max_force_ev_per_angstrom,
            "curvature_ev_per_angstrom2": _finite(result.curvature),
        }
        if result.converged and _get(job.params, "check_frequencies", bool, True):
            summary["frequencies"] = self._frequencies(job, atoms, pump, show)
        return summary

    def _irc(self, job, atoms, pump, show) -> dict[str, Any]:
        from ..reaction_path import irc

        params = job.params
        structure = self._structure

        def progress(direction, step, energy, max_force, positions):
            job.step = step
            if step % 10 == 0:
                job.log.append(
                    f"{direction:7s} step {step:4d}  E {energy:.8f} eV  Fmax {max_force:.5f}"
                )
            show(positions)

        job.log.append("Frequencies to find the imaginary mode…")
        result = irc(
            atoms,
            step=_number(params, "step", 0.1),
            max_steps=int(_number(params, "max_steps", 150)),
            fmax=_number(params, "fmax", 0.02),
            relax_ends=_get(params, "relax_ends", bool, True),
            on_progress=progress,
            should_stop=pump,
        )
        ts_positions = atoms.get_positions().copy()
        frames = result.frames(ts_positions)
        trajectory = _get(params, "trajectory", str, None)
        if trajectory:
            from ase.io import write

            images = []
            for frame in frames:
                image = atoms.copy()
                image.set_positions(frame.positions)
                image.info.update({"energy_ev": frame.energy_ev, "irc_arc": frame.arc})
                images.append(image)
            write(trajectory, images)
        self._publish.append(
            lambda: _publish_path(structure, [frame.positions for frame in frames], "IRC path")
        )
        summary = {
            "stopped": result.stopped,
            "frames": len(frames),
            "ts_frame": len(result.reverse),
            "ts_energy_ev": result.ts_energy_ev,
            "energies_ev": [frame.energy_ev for frame in frames],
            "arc": [frame.arc for frame in frames],
            "reverse_minimum_ev": result.reverse_minimum_ev,
            "forward_minimum_ev": result.forward_minimum_ev,
            "trajectory": trajectory,
        }
        if _get(params, "return_positions", bool, False):
            summary["positions"] = [frame.positions.tolist() for frame in frames]
        return summary

    def _scan(self, job, atoms, pump, show) -> dict[str, Any]:
        from ..reaction_path import scan_to_ts

        params = job.params
        parsed = parse_pairs(_get(params, "pair", str))
        if len(parsed) != 1:
            raise JobSpecError("scan needs pair='I-J'")
        structure = self._structure

        def progress(index, distance, energy, positions):
            job.step = index
            job.log.append(f"point {index:3d}  r {distance:.4f} Å  E {energy:.8f} eV")
            show(positions)
            pump()

        result = scan_to_ts(
            atoms,
            (parsed[0].i, parsed[0].j),
            stop=float(_get(params, "stop", (int, float))),
            start=_get(params, "start", (int, float), None),
            points=int(_number(params, "points", 11)),
            relax_fmax=_number(params, "relax_fmax", 0.05),
            refine=_get(params, "refine", bool, True),
            ts_fmax=_number(params, "fmax", 0.01),
            exact_hessian=_get(params, "exact_hessian", bool, False),
            on_progress=progress,
            should_stop=pump,
        )
        name = f"Bond scan {parsed[0].i}-{parsed[0].j}"
        frames = result.frames
        self._publish.append(lambda: _publish_path(structure, frames, name))
        summary = {
            "stopped": result.stopped,
            "distances": result.distances,
            "energies_ev": result.energies_ev,
            "highest": result.highest,
            "bracketed": result.bracketed,
        }
        if result.ts is not None:
            summary["ts"] = {
                "converged": result.ts.converged,
                "steps": result.ts.steps,
                "energy_ev": result.ts.evaluation.energy_ev,
                "max_force_ev_per_angstrom": result.ts.evaluation.max_force_ev_per_angstrom,
            }
            if result.ts.converged and _get(params, "check_frequencies", bool, True):
                summary["ts"]["frequencies"] = self._frequencies(job, atoms, pump, show)
        return summary

    def _qst(self, job, atoms, pump, show) -> dict[str, Any]:
        from ..reaction_path import qst
        from ..samson_bridge import add_structure_model

        params = job.params
        structure = self._structure
        guess = self._endpoints[0] if len(self._endpoints) == 2 else None
        product = self._endpoints[-1]
        label = "QST3" if guess is not None else "QST2"

        def progress(step, energies, max_force):
            job.step = step
            if step % 5 == 0:
                job.log.append(
                    f"band step {step:4d}  Fmax {max_force:.4f}  "
                    f"max ΔE {max(energies) - energies[0]:+.4f} eV"
                )
            pump()

        result = qst(
            atoms,
            product,
            atoms.calc,
            guess=guess,
            images=int(_number(params, "images", 7)),
            fmax=_number(params, "fmax", 0.05),
            max_steps=int(_number(params, "max_steps", 500)),
            refine=_get(params, "refine", bool, True),
            on_progress=progress,
            should_stop=pump,
        )
        self._publish.append(lambda: _publish_path(structure, result.images, f"{label} path"))
        summary = {
            "method": label,
            "stopped": result.stopped,
            "neb_converged": result.neb_converged,
            "neb_steps": result.neb_steps,
            "energies_ev": result.energies_ev,
            "highest_image": result.highest_image,
            "barrier_forward_ev": result.barrier_forward_ev,
            "barrier_reverse_ev": result.barrier_reverse_ev,
        }
        if result.ts is not None:
            summary["ts"] = {
                "converged": result.ts.converged,
                "steps": result.ts.steps,
                "energy_ev": result.ts.evaluation.energy_ev,
                "max_force_ev_per_angstrom": result.ts.evaluation.max_force_ev_per_angstrom,
            }
            if result.ts.converged and _get(params, "check_frequencies", bool, True):
                ts_atoms = atoms.copy()
                ts_atoms.calc = atoms.calc
                ts_atoms.set_positions(result.ts_positions)
                summary["ts"]["frequencies"] = self._frequencies(job, ts_atoms, pump, show)

            symbols = atoms.get_chemical_symbols()

            def add_ts_model() -> dict[str, Any]:
                model = add_structure_model(
                    f"Transition state ({label})", symbols, result.ts_positions
                )
                return {"ts_model": getattr(model, "name", f"Transition state ({label})")}

            self._publish.append(add_ts_model)
        return summary

    def _frequencies(self, job, atoms, pump, show) -> dict[str, Any]:
        def progress(done, total):
            job.step = done
            if pump():
                raise _Stopped

        job.log.append("Finite-difference Hessian…")
        result = harmonic_frequencies(atoms, on_progress=progress)
        job.log.append(f"Stationary point: {result.classification()}")
        return {
            "wavenumbers_cm": [round(float(value), 2) for value in result.wavenumbers_cm],
            "n_imaginary": result.n_imaginary,
            "classification": result.classification(),
            "hint": result.soft_mode_hint(),
            "rigid_body_modes_removed": result.rigid_body_modes_removed,
        }
