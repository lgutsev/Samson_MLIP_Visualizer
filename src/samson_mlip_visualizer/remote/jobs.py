"""MLIP jobs started through the bridge: single point, relax, MD, TS search, frequencies,
reaction paths, and free energies along a reaction coordinate.

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

from ..calculators import create_calculator, program_options
from ..compat import assert_model_covers_structure
from ..engine import evaluate, relax
from ..md import ENSEMBLES, md_warnings, parse_pairs, run_md
from ..provenance import collect_provenance
from ..samson_bridge import choose_structural_models, extract_structure, sync_positions
from ..ts import dimer_search, prfo_search
from ..vibrations import harmonic_frequencies

KINDS = (
    "single_point", "relax", "md", "ts", "frequencies", "irc", "qst", "scan",
    "slow_growth", "blue_moon", "metadynamics",
)
_MOVES_ATOMS = ("relax", "md", "ts", "irc", "scan", "slow_growth", "blue_moon", "metadynamics")
# job parameter naming each program backend's method (AIMNet2: which network)
_METHOD_KEYS = {"xtb": "xtb_method", "psi4": "psi4_method", "aimnet2": "aimnet_model"}
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


def _pumping(job, pump, interval_s: float = 0.1) -> Callable[[], bool]:
    """A per-step stop check that also turns SAMSON's event loop at least every
    ``interval_s``: pumping only at the report interval can leave SAMSON silent for
    seconds, and Windows then flags it as not responding."""
    last = [0.0]

    def check() -> bool:
        now = time.monotonic()
        if now - last[0] >= interval_s:
            last[0] = now
            pump()
        return job.stop_requested

    return check


def _thin(values, most: int = 201) -> list[float]:
    """At most ``most`` evenly spaced entries (first and last kept), as floats for JSON."""
    values = np.asarray(values, float)
    if len(values) > most:
        values = values[np.unique(np.linspace(0, len(values) - 1, most).round().astype(int))]
    return [round(float(value), 6) for value in values]


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
        options = program_options(
            backend,
            method=_get(params, _METHOD_KEYS.get(backend, "psi4_method"), str, None),
            basis=_get(params, "basis", str, None),
            charge=_get(params, "charge", int, 0),
            multiplicity=_get(params, "multiplicity", int, 1),
            solvent=_get(params, "solvent", str, None),
        )
        key = (backend, tuple(paths), device, dtype, tuple(sorted((options or {}).items())))
        if key not in self._calculators:
            argument = paths[0] if len(paths) == 1 else paths
            # Keep one calculator: a second MACE model would double GPU memory.
            self._calculators = {
                key: create_calculator(
                    backend, argument, device=device, dtype=dtype, options=options
                )
            }
        provenance = collect_provenance(
            backend=backend, model_path=paths[0], device=device, dtype=dtype, settings=options
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
            irc_check = self._check_irc(job, atoms, pump, show, summary["frequencies"])
            if irc_check is not None:
                summary["irc"] = irc_check
        return summary

    def _check_irc(
        self, job, atoms, pump, show, frequencies, *, references=None, pair=None
    ) -> dict[str, Any] | None:
        """With ``check_irc``, follow the IRC from a TS with one imaginary mode and
        report where both relaxed ends land: matched against ``references`` (name →
        positions) when given, and the ``pair`` distance at each end."""
        if not _get(job.params, "check_irc", bool, False):
            return None
        if frequencies["n_imaginary"] != 1:
            job.log.append("IRC check skipped: the TS needs exactly one imaginary mode")
            return {"skipped": "needs exactly one imaginary mode"}
        from ..reaction_path import irc, match_minimum

        structure = self._structure

        def progress(direction, step, energy, max_force, positions):
            job.step = step
            show(positions)

        job.log.append("IRC check: following the imaginary mode both ways…")
        result = irc(atoms, relax_ends=True, on_progress=progress, should_stop=pump)
        frames = [frame.positions for frame in result.frames(atoms.get_positions().copy())]
        self._publish.append(lambda: _publish_path(structure, frames, "IRC path"))
        summary: dict[str, Any] = {"stopped": result.stopped}
        for name in ("reverse", "forward"):
            positions = getattr(result, f"{name}_minimum_positions")
            end: dict[str, Any] = {"minimum_ev": getattr(result, f"{name}_minimum_ev")}
            if positions is not None:
                end["minimum_minus_ts_ev"] = end["minimum_ev"] - result.ts_energy_ev
                if pair is not None:
                    i, j = pair
                    end["pair_distance"] = float(np.linalg.norm(positions[i] - positions[j]))
                if references:
                    end["matches"] = match_minimum(positions, references)
                job.log.append(
                    f"IRC {name} end: E − E(TS) {end['minimum_minus_ts_ev']:+.4f} eV"
                    + (f", r(pair) {end['pair_distance']:.3f} Å" if "pair_distance" in end else "")
                    + (f", matches {end['matches'] or 'neither minimum'}" if references else "")
                )
            summary[name] = end
        if references and not result.stopped:
            landed = {summary[name].get("matches") for name in ("reverse", "forward")}
            summary["connects"] = landed == set(references)
            job.log.append(
                "IRC confirms the TS connects " + " and ".join(references)
                if summary["connects"]
                else "Warning: the IRC does not connect the intended minima"
            )
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
            "jumps": result.jumps,
        }
        if result.jumps:
            job.log.append(
                f"Warning: the geometry jumped at scan point(s) {result.jumps}: the constrained "
                "minimum switched branch, so this profile is not a reaction path and its "
                "highest point may be an artifact. Confirm the TS with an IRC."
            )
        if result.ts is not None:
            summary["ts"] = {
                "converged": result.ts.converged,
                "steps": result.ts.steps,
                "energy_ev": result.ts.evaluation.energy_ev,
                "max_force_ev_per_angstrom": result.ts.evaluation.max_force_ev_per_angstrom,
            }
            if result.ts.converged and _get(params, "check_frequencies", bool, True):
                frequencies = self._frequencies(job, atoms, pump, show)
                summary["ts"]["frequencies"] = frequencies
                irc_check = self._check_irc(
                    job, atoms, pump, show, frequencies, pair=(parsed[0].i, parsed[0].j)
                )
                if irc_check is not None:
                    summary["ts"]["irc"] = irc_check
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

        references = {"reactant": atoms.get_positions().copy(), "product": product.get_positions()}
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
                frequencies = self._frequencies(job, ts_atoms, pump, show)
                summary["ts"]["frequencies"] = frequencies
                # QST leaves the reactant model in place, so the IRC is not shown live.
                irc_check = self._check_irc(
                    job, ts_atoms, pump, lambda positions: None, frequencies, references=references
                )
                if irc_check is not None:
                    summary["ts"]["irc"] = irc_check

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

    # --- free energies along a reaction coordinate ------------------------------------

    def _free_energy_setup(self, job, atoms):
        """The coordinate, masses, and MD settings shared by the free-energy jobs."""
        from ..free_energy import parse_coordinate

        params = job.params
        try:
            coordinate = parse_coordinate(_get(params, "coordinate", str))
        except ValueError as exc:
            raise JobSpecError(str(exc)) from exc
        indices = [index for a, b, _ in coordinate.terms for index in (a, b)]
        if max(indices) >= len(atoms) or min(indices) < 0:
            raise JobSpecError(
                f"coordinate names atom {max(indices)}; the structure has {len(atoms)} atoms"
            )
        masses = atoms.get_masses().copy()
        hydrogens = np.array(atoms.get_chemical_symbols()) == "H"
        hydrogen_mass = _get(params, "hydrogen_mass", (int, float), None)
        if hydrogen_mass:
            masses[hydrogens] = float(hydrogen_mass)
        timestep = _number(params, "timestep_fs", 1.0)
        if hydrogens.any() and timestep > 1.0 and (hydrogen_mass or 1.0) < 2.5:
            job.log.append(
                f"Warning: a {timestep:g} fs step with light hydrogens; set hydrogen_mass=3 "
                "(tritium, as the VASP tutorial does) or use timestep_fs <= 1"
            )
        settings = {
            "temperature_k": _number(params, "temperature_k", 300.0),
            "timestep_fs": timestep,
            "andersen_probability": _number(params, "andersen_probability", 0.05),
            "masses": masses,
        }
        job.log.append(
            "ξ = " + " ".join(f"{c:+g}·d({a},{b})" for a, b, c in coordinate.terms)
            + f" = {coordinate.value(atoms.get_positions()):+.4f} Å now"
        )
        return coordinate, settings

    def _constrained_run(self, job, atoms, pump, show, coordinate, settings, label, **options):
        """``constrained_md`` with the job's progress log, live view, and stop button."""
        from ..free_energy import constrained_md

        every = max(1, int(_number(job.params, "report_interval", 10)))
        first = job.step

        def on_step(step, records):
            job.step = first + step + 1
            if (step + 1) % every == 0:
                job.log.append(
                    f"{label} step {step + 1:6d}  ξ {records['value'][-1]:+.4f} Å  "
                    f"λ {records['lam'][-1]:+.4f} eV/Å  T {records['temperature'][-1]:6.1f} K"
                )
                show(atoms.get_positions())

        return constrained_md(
            atoms, coordinate, **settings, on_step=on_step,
            should_stop=_pumping(job, pump), **options,
        )

    def _move_coordinate(self, job, atoms, pump, show, coordinate, settings, target, seed):
        """Drag ξ to ``target`` with a short slow-growth run (``increment`` Å per step)."""
        here = coordinate.value(atoms.get_positions())
        steps = int(np.ceil(abs(target - here) / self._increment(job)))
        if steps:
            self._constrained_run(
                job, atoms, pump, show, coordinate, settings, "move", steps=steps,
                target=here, increment=(target - here) / steps, seed=seed,
            )

    @staticmethod
    def _increment(job) -> float:
        rate = abs(_number(job.params, "increment", 1e-3))
        if rate <= 0:
            raise JobSpecError("increment must be positive")
        return rate

    @staticmethod
    def _save_output(job, arrays: dict[str, Any]) -> str | None:
        path = _get(job.params, "output", str, None)
        if path:
            np.savez(path, **{key: np.asarray(value) for key, value in arrays.items()})
        return path

    def _slow_growth(self, job, atoms, pump, show) -> dict[str, Any]:
        from ..free_energy import slow_growth_profile

        params = job.params
        coordinate, settings = self._free_energy_setup(job, atoms)
        here = coordinate.value(atoms.get_positions())
        start = float(_get(params, "start", (int, float), here))
        end = float(_get(params, "end", (int, float)))
        rate = self._increment(job)
        seed = _get(params, "seed", int, 0)
        self._move_coordinate(job, atoms, pump, show, coordinate, settings, start, seed)
        equilibration = int(_number(params, "equilibration_steps", 500))
        if equilibration and not job.stop_requested:
            self._constrained_run(job, atoms, pump, show, coordinate, settings, "hold",
                                  steps=equilibration, target=start, seed=seed + 1)
        if job.stop_requested:
            return {"stopped": True}
        steps = max(1, int(round(abs(end - start) / rate)))
        every = max(1, int(_number(params, "report_interval", 10)))
        job.log.append(f"slow growth: ξ {start:+.3f} → {end:+.3f} Å in {steps} steps")
        record = self._constrained_run(
            job, atoms, pump, show, coordinate, settings, "grow", steps=steps, target=start,
            increment=(end - start) / steps, seed=seed + 2, record_every=every,
        )
        xi, profile = slow_growth_profile(record)
        top = int(np.argmax(profile))
        job.log.append(
            f"highest ΔA {profile[top]:+.4f} eV at ξ {xi[top]:+.3f} Å; end {profile[-1]:+.4f} eV"
        )
        structure, frames = self._structure, record.frames
        name = f"Slow growth ξ {start:+.2f} → {end:+.2f}"
        self._publish.append(lambda: _publish_path(structure, frames, name))
        return {
            "stopped": len(profile) < steps,
            "steps": len(profile),
            "xi": _thin(xi),
            "free_energy_ev": _thin(profile),
            "max_ev": float(profile[top]),
            "xi_at_max": float(xi[top]),
            "end_ev": float(profile[-1]),
            "mean_temperature_k": float(record.temperature.mean()),
            "output": self._save_output(job, record.as_dict()),
            "note": "slow growth is irreversible work: run it both ways, the hysteresis is "
            "its error; blue_moon gives the converged profile",
        }

    def _blue_moon(self, job, atoms, pump, show) -> dict[str, Any]:
        from ..free_energy import (
            blue_moon_gradient,
            generalized_velocity,
            integrate_gradient,
            integration_error,
            zero_crossing,
        )

        params = job.params
        coordinate, settings = self._free_energy_setup(job, atoms)
        values = _get(params, "values", list, None)
        if values is None:
            values = np.linspace(
                float(_get(params, "start", (int, float))),
                float(_get(params, "stop", (int, float))),
                int(_number(params, "points", 11)),
            ).tolist()
        if len(values) < 2 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        ):
            raise JobSpecError("blue_moon needs two or more numeric values, or start/stop/points")
        steps = int(_number(params, "steps", 2000))
        skip = int(_number(params, "skip", steps // 5))
        if skip >= steps - 10:
            raise JobSpecError("skip must leave at least 10 steps of each window")
        every = max(1, int(_number(params, "report_interval", 10)))
        seed = _get(params, "seed", int, 0)
        temperature = settings["temperature_k"]
        windows, frames, records = [], [], []
        for k, target in enumerate(float(v) for v in values):
            self._move_coordinate(job, atoms, pump, show, coordinate, settings, target,
                                  seed + 1000 + k)
            if job.stop_requested:
                break
            record = self._constrained_run(
                job, atoms, pump, show, coordinate, settings, f"window {k + 1}",
                steps=steps, target=target, seed=seed + k, record_every=every,
            )
            if len(record.lam) < steps:
                break  # stopped inside the window: drop it
            gradient, error = blue_moon_gradient(record, temperature, skip)
            windows.append({"xi": target, "mean_force_ev_per_angstrom": gradient,
                            "error_ev_per_angstrom": error,
                            "mean_temperature_k": float(record.temperature[skip:].mean())})
            frames.append(atoms.get_positions().copy())
            records.append(record)
            job.log.append(
                f"window {k + 1}/{len(values)}: ξ {target:+.3f} Å  dA/dξ {gradient:+.4f} ± "
                f"{error:.4f} eV/Å"
            )
        summary: dict[str, Any] = {"stopped": len(windows) < len(values), "windows": windows}
        if len(windows) < 2:
            return summary
        # Integrate in increasing ξ, whatever order the windows ran in.
        order = np.argsort([w["xi"] for w in windows], kind="stable")
        records = [records[k] for k in order]
        xi = np.array([windows[k]["xi"] for k in order])
        gradient = np.array([windows[k]["mean_force_ev_per_angstrom"] for k in order])
        profile = integrate_gradient(xi, gradient)
        error = integration_error(xi, [windows[k]["error_ev_per_angstrom"] for k in order])
        xi_min = zero_crossing(xi, gradient, up=True)
        xi_star = zero_crossing(xi, gradient, up=False)
        summary.update({"xi": xi.tolist(), "free_energy_ev": profile.tolist(),
                        "free_energy_error_ev": error.tolist(),
                        "xi_min": xi_min, "xi_star": xi_star})
        if xi_star is not None:
            nearest = int(np.argmin(np.abs(xi - xi_star)))
            summary["ts_window"] = {
                "xi": float(xi[nearest]),
                "generalized_velocity_A_per_s": generalized_velocity(
                    records[nearest].z[skip:], temperature
                ),
            }
        if xi_star is not None:
            top = float(np.interp(xi_star, xi, profile))
            if xi_min is not None and xi_min < xi_star:
                barrier, low = top - float(np.interp(xi_min, xi, profile)), xi_min
            else:
                # The minimum lies before the first window: a lower bound.
                left = int(np.argmin(np.where(xi <= xi_star, profile, np.inf)))
                barrier, low = top - float(profile[left]), float(xi[left])
                summary["barrier_is_lower_bound"] = True
                job.log.append("No minimum inside the windows before ξ*: add windows at lower ξ")
            summary["barrier_ev"] = barrier
            job.log.append(f"barrier ΔA‡ {barrier:.4f} eV (ξ {low:+.3f} → {xi_star:+.3f} Å)")
        structure = self._structure
        self._publish.append(lambda: _publish_path(structure, frames, "Blue moon windows"))
        summary["output"] = self._save_output(job, {
            "xi": xi, "mean_force": gradient, "free_energy": profile, "error": error,
            **{f"lam_{k}": r.lam for k, r in enumerate(records)},
            **{f"z_{k}": r.z for k, r in enumerate(records)},
            **{f"g_{k}": r.g for k, r in enumerate(records)},
        })
        return summary

    def _metadynamics(self, job, atoms, pump, show) -> dict[str, Any]:
        from ..free_energy import MetadynamicsCalculator, fes_from_hills, metadynamics

        params = job.params
        coordinate, settings = self._free_energy_setup(job, atoms)
        atoms.set_masses(settings["masses"])
        wall_k = _number(params, "wall_k", 20.0)
        walls = []
        lower = _get(params, "lower_wall", (int, float), None)
        upper = _get(params, "upper_wall", (int, float), None)
        if lower is not None:
            walls.append(("xi_lower", None, float(lower), wall_k))
        if upper is not None:
            walls.append(("xi_upper", None, float(upper), wall_k))
        for pair in parse_pairs(_get(params, "max_distances", str, "")):
            if pair.target is None:
                raise JobSpecError("max_distances needs limits, like '0-5:5.0'")
            walls.append(("distance_upper", (pair.i, pair.j), pair.target, wall_k))
        sigma = _number(params, "sigma", 0.08)
        bias_factor = _number(params, "bias_factor", 10.0)
        if bias_factor <= 1:
            raise JobSpecError("bias_factor must be above 1")
        base = atoms.calc
        bias = MetadynamicsCalculator(
            base, coordinate, height=_number(params, "height", 0.02), sigma=sigma,
            temperature_k=settings["temperature_k"], bias_factor=bias_factor, walls=walls,
        )
        every = max(1, int(_number(params, "report_interval", 10)))

        def progress(step, xi):
            job.step = step + 1
            if step % (10 * every) == 0:
                last = bias.heights[-1] if bias.heights else 0.0
                job.log.append(
                    f"step {step:7d}  ξ {xi:+.4f} Å  hills {len(bias.centers)}  "
                    f"last height {last:.4f} eV"
                )
            show(atoms.get_positions())

        try:
            result = metadynamics(
                atoms, bias, steps=int(_number(params, "steps", 10000)),
                pace=int(_number(params, "pace", 50)), timestep_fs=settings["timestep_fs"],
                friction_per_fs=_number(params, "friction_per_fs", 0.01),
                seed=_get(params, "seed", int, 0), record_every=every, on_progress=progress,
                should_stop=_pumping(job, pump),
            )
        finally:
            atoms.calc = base
        trace = result["xi"]
        low = max(trace.min(), lower) if lower is not None else trace.min()
        high = min(trace.max(), upper) if upper is not None else trace.max()
        grid = np.linspace(low, high, 200)
        fes = fes_from_hills(grid, result["centers"], result["heights"], sigma, bias_factor)
        frames = result["frames"][:: max(1, len(result["frames"]) // 400)]
        structure = self._structure
        self._publish.append(lambda: _publish_path(structure, frames, "Metadynamics trajectory"))
        heights = result["heights"]
        return {
            "stopped": job.stop_requested,
            "steps": int(job.step),
            "hills": len(result["centers"]),
            "last_hill_height_ev": float(heights[-1]) if len(heights) else None,
            "xi": _thin(grid),
            "free_energy_ev": _thin(fes),
            "xi_at_min": float(grid[int(np.argmin(fes))]),
            "xi_visited": [float(trace.min()), float(trace.max())],
            "output": self._save_output(job, {
                "xi": trace, "centers": result["centers"], "heights": heights,
                "sigma": sigma, "bias_factor": bias_factor,
            }),
            "note": "the free energy is −γ/(γ−1)·V(ξ); trust it only after many crossings "
            "between the states, once the hills have become small",
        }
