"""UMA worker process for :mod:`.uma_backend`.

This file runs in the Python that has ``fairchem-core`` (its own environment), not
in SAMSON's, so it imports nothing from this package except :mod:`.uma_config`
(standard library only, loaded from this folder): NumPy, ASE, and fairchem.
Protocol as in :mod:`.worker_process`.

Request: ``{"config" (a :class:`~.uma_config.UMAConfig` as a dict), "device",
"atom_refs" (a references .yaml, or null to look beside the checkpoint), "numbers",
"positions" (Å), "cell" (or null), "pbc"}``.
Reply: ``{"energy" (eV), "forces" (eV/Å), "backend" (requested and effective
inference, fallback warnings), "events" (what changed since the last reply),
"timing"}`` or ``{"error"}``.

Predictors are kept between requests. UMA's fast paths (Nitro, and high-precision's
merged MoLE + compile) are specialised to one composition, task, charge and spin;
fairchem falls back permanently to a slower path when any of them changes. The
worker instead rebuilds the predictor for the new system and says so, so a later
structure runs on the fast path again. At most two predictors stay loaded.
"""

import json
import logging
import os
import sys
import time
import warnings

PREFIX = "@@SAMSON "
MAX_UNITS = 2


class _Collect(logging.Handler):
    """Warnings fairchem logs (fallbacks are logged, not raised)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _atom_refs(checkpoint, explicit):
    """Isolated-atom reference energies for a local checkpoint: ``explicit``, or
    ``iso_atom_elem_refs.yaml`` beside it or in a ``references`` folder beside it (the
    layout of the facebook/UMA repository). Without them single atoms (F⁻, Cl⁻) fail."""
    from pathlib import Path

    if not checkpoint.lower().endswith(".pt"):
        return None
    folder = Path(checkpoint).parent
    candidates = [Path(explicit)] if explicit else [
        folder / "iso_atom_elem_refs.yaml", folder / "references" / "iso_atom_elem_refs.yaml"]
    for path in candidates:
        if path.is_file():
            from omegaconf import OmegaConf

            return OmegaConf.load(path)
    return None


def _gpu():
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "torch": torch.__version__}
    index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return {"cuda": True, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "name": torch.cuda.get_device_name(index), "capability": f"{major}.{minor}"}


def _versions():
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in ("fairchem-core", "torch", "ase", "numpy", "nvalchemi-toolkit-ops",
                 "triton", "triton-windows", "nvidia-cutlass-dsl"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    return found


def main() -> None:
    protocol = sys.stdout
    sys.stdout = sys.stderr  # stray Python prints must not reach the protocol stream

    def reply(message):
        protocol.write(PREFIX + json.dumps(message) + "\n")
        protocol.flush()

    try:  # the OS certificate store for downloads (checkpoints, the D3 table), if installed
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    try:
        import numpy as np
        import torch
        from ase import Atoms

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import uma_config
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        reply({"error": f"Could not import fairchem: {type(exc).__name__}: {exc}"})
        return
    versions = _versions()
    reply({"ready": True, "version": versions.get("fairchem-core"), "versions": versions,
           "gpu": _gpu()})

    collect = _Collect()
    logging.getLogger().addHandler(collect)
    units = {}  # (model, device, inference, atom_refs) -> loaded predictor and its state
    calculators = {}  # (*unit key, task, dispersion) -> ASE calculator
    for line in sys.stdin:
        if not line.strip():
            continue
        events = []
        try:
            request = json.loads(line)
            config = uma_config.UMAConfig.from_dict(request["config"]).validated()
            device = request.get("device", "cpu")
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("device 'cuda' was requested but this environment's torch "
                                   "has no usable CUDA device")
            numbers = [int(z) for z in request["numbers"]]
            signature = (tuple(sorted(numbers)), config.task, config.charge,
                         config.multiplicity)
            unit_key = (config.model, device, config.inference, request.get("atom_refs"))
            entry = units.get(unit_key)
            if entry is not None and entry["signature"] not in (None, signature):
                fast = entry["info"]["nitro"] or entry["info"]["merge_mole"]
                if fast:
                    changed = [name for name, old, new in zip(
                        ("composition", "task", "charge", "spin"), entry["signature"],
                        signature, strict=True) if old != new]
                    events.append(f"Rebuilt the UMA predictor: {', '.join(changed)} changed "
                                  f"(its {entry['info']['effective']} fast path is "
                                  f"specialised to one system).")
                    _drop(units, calculators, unit_key)
                    entry = None
            load_s = 0.0
            if entry is None:
                while len(units) >= MAX_UNITS:
                    _drop(units, calculators, next(iter(units)))
                start = time.perf_counter()
                collect.messages.clear()
                unit, info = uma_config.load_predict_unit(
                    config, device=device,
                    atom_refs=_atom_refs(config.model, request.get("atom_refs")))
                info["warnings"] = info["warnings"] + collect.messages[:]
                load_s = time.perf_counter() - start
                entry = units[unit_key] = {"unit": unit, "info": info, "signature": None,
                                           "load_s": load_s, "announced": False}
            key = (*unit_key, config.task, config.dispersion)
            if key not in calculators:
                calculators[key], _ = uma_config.build_calculator(
                    config, device=device, predict_unit=entry["unit"])
            cell = request.get("cell")
            atoms = Atoms(numbers=numbers, positions=request["positions"],
                          cell=cell, pbc=request.get("pbc", False) if cell else False)
            atoms.info.update(charge=config.charge, spin=config.multiplicity)
            atoms.calc = calculators[key]
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            collect.messages.clear()
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            eval_s = time.perf_counter() - start
            entry["signature"] = signature
            new = [str(w.message) for w in caught] + collect.messages[:]
            new = [m for m in dict.fromkeys(new) if m not in entry["info"]["warnings"]]
            if new:
                entry["info"]["warnings"].extend(new)
                if entry["announced"]:  # the first evaluation's go into the load message
                    events.extend(f"fairchem: {m}" for m in new)
            # fairchem decides on Nitro at the first evaluation and can fall back at a
            # later one (a graph that outgrows Nitro's buffers); re-read what runs.
            now = uma_config.predict_unit_info(entry["unit"], config)
            if not entry["announced"]:
                entry["info"].update(now)
                entry["announced"] = True
                events.append(_describe(entry["info"], device))
            elif now["effective"] != entry["info"]["effective"]:
                events.append(f"UMA inference changed from {entry['info']['effective']} to "
                              f"{now['effective']}.")
                entry["info"].update(now)
            timing = {"load_s": load_s, "eval_s": eval_s}
            if device.startswith("cuda"):
                timing["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
            reply({"energy": energy, "forces": forces.tolist(), "backend": entry["info"],
                   "events": events, "timing": timing})
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            reply({"error": f"{type(exc).__name__}: {exc}", "events": events})


def _drop(units, calculators, unit_key):
    units.pop(unit_key, None)
    for key in [k for k in calculators if k[: len(unit_key)] == unit_key]:
        del calculators[key]
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - freeing memory is best effort
        pass


def _describe(info, device):
    text = (f"UMA loaded on {device}: requested {info['requested']}, effective "
            f"{info['effective']} (tf32 "
            f"{('on' if info['tf32'] else 'off') if device.startswith('cuda') else 'n/a'}, "
            f"backend "
            f"{info['execution_mode']}, compile {'on' if info['compile'] else 'off'})")
    if info["warnings"]:
        text += "; fairchem: " + " | ".join(info["warnings"])
    return text + "."


if __name__ == "__main__":
    main()
