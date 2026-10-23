"""UMA model presets and the one place a UMA calculator is configured and built.

This module imports only the standard library at import time, so three kinds of
callers share it: the bridge (SAMSON's Python, numpy 1.24, no fairchem), the UMA
worker process (:mod:`.uma_worker`, which loads it by file path), and other
projects that build the same calculator in a fairchem environment directly::

    from samson_mlip_visualizer.uma_config import UMAConfig, build_calculator

    config = UMAConfig.from_preset("r2scan").validated()
    calc, info = build_calculator(config, device="cuda")

Presets (fairchem-core >= 2.24, October 2026 checkpoints):

=============  ========================  ======  ======================================
preset         checkpoint                task    dispersion
=============  ========================  ======  ======================================
``standard``   ``uma-s-1p2p1``           any     ``none``, or ``pbe-d3bj`` on omat/oc22
``d4rev``      ``uma-s-1p2p1-ft-d4rev``  omol    included in the labels (ωB97M-D4rev)
``r2scan``     ``uma-s-1p2p1-ft-r2scan`` omat    ``r2scan-d3bj`` (fairchem DFTD3Calculator)
``custom``     any .pt or fairchem name  any     as ``standard``
=============  ========================  ======  ======================================

Inference modes: ``nitro`` (fairchem's ``default``: UMA Nitro on Ampere/Hopper/
Blackwell GPUs for the three checkpoints above, else a reduced-precision fallback
that fairchem names in a warning) and ``high-precision`` (FP32, TF32 off). The
removed ``turbo`` mode maps to ``nitro``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

TASKS = ("omol", "omat", "oc20", "odac", "omc", "oc25", "oc22")
# Level of theory of each task's labels and whether they already contain dispersion.
TASK_THEORY = {
    "omol": ("ωB97M-V/def2-TZVPD (VV10 non-local dispersion included)", True),
    "omat": ("PBE/PBE+U (VASP, Materials Project settings), no dispersion", False),
    "oc20": ("RPBE (VASP), no dispersion", False),
    "odac": ("PBE+D3 (VASP), dispersion included", True),
    "omc": ("PBE+D3 (VASP), dispersion included", True),
    "oc25": ("RPBE+D3 (VASP), dispersion included", True),
    "oc22": ("PBE+U (VASP), spin-polarized, no dispersion", False),
}
# Tasks whose labels are plain PBE(+U), the only ones PBE-D3(BJ) may be added to.
PBE_TASKS = ("omat", "oc22")

DISPERSIONS = ("none", "pbe-d3bj", "r2scan-d3bj")
INFERENCE_MODES = ("nitro", "high-precision")
# fairchem's name for each inference mode (``default`` is the Nitro preset).
FAIRCHEM_SETTINGS = {"nitro": "default", "high-precision": "high-precision"}
LEGACY_MODES = {"turbo": "nitro", "default": "nitro"}

# fairchem's DFTD3Calculator presets (docs/core/common_tasks/ase_calculator.md):
# Becke–Johnson damping, 15 Å cutoff, C5 smoothing over the outer 20 % of the cutoff.
D3_SETTINGS = {
    "pbe-d3bj": {"functional": "pbe", "damping": "BJ", "a1": 0.4289, "a2_bohr": 4.4407,
                 "s6": 1.0, "s8": 0.7875, "k1": 16.0, "k3": -4.0, "cutoff_angstrom": 15.0,
                 "smoothing": "C5 over the outer 20% of the cutoff"},
    "r2scan-d3bj": {"functional": "r2scan", "damping": "BJ", "a1": 0.49484001,
                    "a2_bohr": 5.73083694, "s6": 1.0, "s8": 0.78981345, "k1": 16.0,
                    "k3": -4.0, "cutoff_angstrom": 15.0,
                    "smoothing": "C5 over the outer 20% of the cutoff"},
}
D3_NOTE = (
    "DFT-D3(BJ) from fairchem's DFTD3Calculator (nvalchemiops kernels, zero three-body "
    "term). It is not equivalent to every VASP 'PBE+D3' setup (IVDW=11 zero damping, "
    "IVDW=12 BJ with VASP's own cutoffs and three-body option differ)."
)
NONMAGNETIC_WARNING = (
    "uma-s-1p2p1-ft-r2scan is not spin-polarized: it describes every system as "
    "nonmagnetic. Do not use it for magnetic systems (Fe clusters, antiferromagnetic "
    "NiO, spin ladders) or for cohesive energies, whose isolated atoms are often "
    "magnetic. Use the standard preset (uma-s-1p2p1, omat) for those."
)


class UMAConfigError(ValueError):
    """A UMA model/task/dispersion combination that must not be run."""


@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    checkpoint: str | None  # None: the user chooses (custom)
    tasks: tuple[str, ...]
    default_task: str
    dispersions: tuple[str, ...]
    default_dispersion: str
    dispersion_in_labels: bool  # the checkpoint's own labels include dispersion
    spin_polarized: bool
    level_of_theory: str
    notes: tuple[str, ...] = ()


PRESETS = {
    "standard": Preset(
        "standard", "Standard UMA (uma-s-1p2p1)", "uma-s-1p2p1", TASKS, "omol",
        DISPERSIONS[:2], "none", False, True,
        "per task (see TASK_THEORY)",
        ("PBE-D3(BJ) only on a PBE task (omat, oc22); omol, omc, odac and oc25 labels "
         "already include dispersion.",),
    ),
    "d4rev": Preset(
        "d4rev", "Molecular ωB97M-D4rev (uma-s-1p2p1-ft-d4rev)", "uma-s-1p2p1-ft-d4rev",
        ("omol",), "omol", ("none",), "none", True, True,
        "ωB97M-D4rev (OMol25-4M relabelled), dispersion included",
        ("Charge and spin multiplicity work as in uma-s-1p2p1's omol task.",
         "No isolated-atom references are shipped; single-atom energies may be unreliable.",
         "Fine-tuned on less data than uma-s-1p2p1; the latter is often closer to its own "
         "level of theory."),
    ),
    "r2scan": Preset(
        "r2scan", "Materials r²SCAN+D3 (uma-s-1p2p1-ft-r2scan)", "uma-s-1p2p1-ft-r2scan",
        ("omat",), "omat", ("r2scan-d3bj", "none"), "r2scan-d3bj", False, False,
        "r²SCAN (MAD-1.6), not spin-polarized, no dispersion in the labels",
        (NONMAGNETIC_WARNING,
         "No isolated-atom references are shipped; single-atom energies may be unreliable."),
    ),
    "custom": Preset(
        "custom", "Custom checkpoint", None, TASKS, "omol", DISPERSIONS[:2], "none",
        False, True, "per task (see TASK_THEORY)",
        ("Any UMA checkpoint (.pt path or fairchem name). Nitro serves only the "
         "uma-s-1p2p1 family; other checkpoints fall back.",),
    ),
}
FINE_TUNED = {"uma-s-1p2p1-ft-d4rev": "d4rev", "uma-s-1p2p1-ft-r2scan": "r2scan"}


def checkpoint_name(model: str) -> str:
    """``uma-s-1p2p1-ft-r2scan`` from a fairchem name or the path of its .pt file."""
    name = model.replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-3] if name.lower().endswith(".pt") else name


def normalize_mode(mode: str | None) -> tuple[str, str | None]:
    """``(mode, note)``: the supported inference mode for a requested one."""
    requested = (mode or "nitro").strip().lower()
    if requested in INFERENCE_MODES:
        return requested, None
    if requested in LEGACY_MODES:
        new = LEGACY_MODES[requested]
        if requested == "turbo":
            return new, ("fairchem >= 2.24 removed the 'turbo' inference mode; using "
                         "'nitro' (the default). turbo's settings are now 'multigpu', "
                         "for multi-worker inference only.")
        return new, None
    raise UMAConfigError(
        f"UMA inference mode must be one of {', '.join(INFERENCE_MODES)}, not {mode!r}"
    )


@dataclass(frozen=True)
class UMAConfig:
    """Everything that decides which UMA numbers come out, validated before a run.

    ``model`` is a fairchem checkpoint name or the path of a local .pt; it defaults
    to the preset's checkpoint. ``magnetic`` is an explicit request for a magnetic
    (spin-polarized) description: ``True`` is rejected by the nonmagnetic r²SCAN
    preset, ``None`` means not requested. It is never inferred from the multiplicity.
    """

    preset: str = "standard"
    model: str | None = None
    task: str | None = None
    dispersion: str | None = None
    inference: str = "nitro"
    charge: int = 0
    multiplicity: int = 1
    magnetic: bool | None = None
    notes: tuple[str, ...] = field(default=(), compare=False)

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> UMAConfig:
        return cls(preset=preset, **overrides)

    @classmethod
    def from_dict(cls, data: dict) -> UMAConfig:
        names = {f.name for f in dataclasses.fields(cls)} - {"notes"}
        return cls(**{k: v for k, v in data.items() if k in names})

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if f.name != "notes"}

    @property
    def checkpoint(self) -> str:
        return checkpoint_name(self.model or "")

    def validated(self) -> UMAConfig:
        """This configuration with defaults filled in, or :class:`UMAConfigError`."""
        name = (self.preset or "standard").strip().lower()
        if name not in PRESETS:
            raise UMAConfigError(
                f"UMA preset must be one of {', '.join(PRESETS)}, not {self.preset!r}"
            )
        preset = PRESETS[name]
        notes: list[str] = []
        model = (self.model or "").strip() or preset.checkpoint
        if not model:
            raise UMAConfigError(
                "Choose a UMA checkpoint for the custom preset: a .pt file or a fairchem "
                "name such as uma-s-1p2p1"
            )
        found = checkpoint_name(model)
        if preset.checkpoint is not None and found != preset.checkpoint:
            raise UMAConfigError(
                f"The {name} preset runs {preset.checkpoint}, not {found}; choose the "
                f"custom preset for other checkpoints"
            )
        if preset.checkpoint is None and found in FINE_TUNED:
            # a fine-tune loaded by path still gets its preset's rules
            return dataclasses.replace(self, preset=FINE_TUNED[found], model=model).validated()
        task = (self.task or preset.default_task).strip().lower()
        if task not in TASKS:
            raise UMAConfigError(f"UMA task must be one of {', '.join(TASKS)}, not {task!r}")
        if task not in preset.tasks:
            raise UMAConfigError(
                f"{preset.checkpoint} has only the {'/'.join(preset.tasks)} task, not {task}"
            )
        dispersion = (self.dispersion or preset.default_dispersion).strip().lower()
        if dispersion not in DISPERSIONS:
            raise UMAConfigError(
                f"UMA dispersion must be one of {', '.join(DISPERSIONS)}, not {dispersion!r}"
            )
        _check_dispersion(preset, task, dispersion)
        if dispersion != "none":
            notes.append(D3_NOTE)
        mode, note = normalize_mode(self.inference)
        if note:
            notes.append(note)
        if self.multiplicity < 1:
            raise UMAConfigError("The multiplicity must be at least 1")
        if task != "omol" and (self.charge != 0 or self.multiplicity != 1):
            if not preset.spin_polarized:
                raise UMAConfigError(
                    f"{preset.checkpoint} is not spin-polarized and takes no charge or "
                    f"multiplicity (got charge {self.charge}, multiplicity "
                    f"{self.multiplicity}). " + NONMAGNETIC_WARNING
                )
            raise UMAConfigError(
                f"Charge and multiplicity apply to the omol task, not {task}"
            )
        if self.magnetic and not preset.spin_polarized:
            raise UMAConfigError("A magnetic calculation was requested. " + NONMAGNETIC_WARNING)
        notes.extend(preset.notes)
        return dataclasses.replace(
            self, preset=name, model=model, task=task, dispersion=dispersion,
            inference=mode, charge=int(self.charge), multiplicity=int(self.multiplicity),
            notes=tuple(notes),
        )

    def describe(self) -> dict[str, str]:
        """What provenance records about the model (validated configurations only)."""
        preset = PRESETS[self.preset]
        theory = (TASK_THEORY[self.task][0] if preset.level_of_theory.startswith("per task")
                  else preset.level_of_theory)
        info = {
            "uma_preset": self.preset,
            "uma_checkpoint": self.checkpoint,
            "uma_task": self.task,
            "uma_level_of_theory": theory,
            "uma_dispersion": self.dispersion if self.dispersion != "none" else (
                "included in the labels" if (preset.dispersion_in_labels
                                             or TASK_THEORY[self.task][1]) else "none"),
            "uma_inference_requested": self.inference,
            "uma_spin_polarized": str(preset.spin_polarized).lower(),
        }
        if self.dispersion in D3_SETTINGS:
            settings = D3_SETTINGS[self.dispersion]
            info["uma_d3_settings"] = ", ".join(f"{k}={v}" for k, v in settings.items())
        return info


def _check_dispersion(preset: Preset, task: str, dispersion: str) -> None:
    if dispersion == "none":
        return
    if preset.dispersion_in_labels:
        raise UMAConfigError(
            f"{preset.checkpoint} already includes dispersion in its labels "
            f"({preset.level_of_theory}); adding {dispersion} would count it twice"
        )
    if dispersion not in preset.dispersions:
        allowed = ", ".join(d for d in preset.dispersions if d != "none") or "none"
        raise UMAConfigError(
            f"{preset.checkpoint or 'This checkpoint'} takes dispersion {allowed}, "
            f"not {dispersion}"
        )
    theory, included = TASK_THEORY[task]
    if dispersion == "pbe-d3bj":
        if included:
            raise UMAConfigError(
                f"The {task} task's labels already include dispersion ({theory}); "
                f"adding PBE-D3(BJ) would count it twice"
            )
        if task not in PBE_TASKS:
            raise UMAConfigError(
                f"PBE-D3(BJ) parameters fit PBE, but the {task} task is {theory}; use "
                f"it with {' or '.join(PBE_TASKS)}"
            )


MAGNETIC_ELEMENTS = frozenset(
    {24, 25, 26, 27, 28, 44, 45, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 92, 93, 94}
)


def structure_warnings(config: UMAConfig, numbers, initial_magmoms=None) -> list[str]:
    """Reasons to doubt a validated configuration for one structure.

    Raises :class:`UMAConfigError` when the structure carries initial magnetic moments
    (an explicit magnetic request) for the nonmagnetic r²SCAN preset. Elements that
    are often magnetic only warn: the multiplicity or composition alone cannot decide
    whether a system is magnetic.
    """
    preset = PRESETS[config.preset]
    numbers = [int(z) for z in numbers]
    found: list[str] = []
    if not preset.spin_polarized:
        if initial_magmoms is not None and any(abs(float(m)) > 1e-8 for m in initial_magmoms):
            raise UMAConfigError(
                "The structure carries initial magnetic moments (a magnetic calculation). "
                + NONMAGNETIC_WARNING
            )
        if MAGNETIC_ELEMENTS.intersection(numbers):
            found.append(
                "This structure contains elements that are often magnetic (Cr, Mn, Fe, Co, "
                "Ni, ...); the r²SCAN preset treats it as nonmagnetic."
            )
        if len(numbers) == 1:
            found.append("A single atom: r²SCAN-preset energies of isolated atoms are "
                         "nonmagnetic and unreferenced; not for cohesive energies.")
    elif config.checkpoint in FINE_TUNED and len(numbers) == 1:
        found.append("No isolated-atom references ship with this fine-tune; single-atom "
                     "energies may be unreliable.")
    return found


# --- construction (runs only where fairchem is installed) --------------------------------


def build_calculator(config: UMAConfig, *, device: str = "cuda", atom_refs=None,
                     predict_unit=None):
    """An ASE calculator for a validated ``config``, and what was actually built.

    Returns ``(calculator, info)``: ``info`` has the effective inference mode
    (``nitro`` when fairchem reports UMA Nitro running), the fallback warnings
    fairchem raised while loading, and the D3 settings. ``predict_unit`` reuses an
    already-loaded predictor (from :func:`load_predict_unit`).
    """
    from fairchem.core import FAIRChemCalculator

    if predict_unit is None:
        predict_unit, info = load_predict_unit(config, device=device, atom_refs=atom_refs)
    else:
        info = predict_unit_info(predict_unit, config)
    calculator = FAIRChemCalculator(predict_unit, task_name=config.task)
    if config.dispersion in D3_SETTINGS:
        from fairchem.core import DFTD3Calculator

        calculator = DFTD3Calculator(
            calculator, functional=D3_SETTINGS[config.dispersion]["functional"],
            device=device,
        )
    return calculator, info


def triton_available() -> bool:
    """Whether Triton imports: UMA Nitro, fairchem's umas_fast_gpu backend and
    torch.compile on CUDA all need it, and it has no official Windows build."""
    import importlib.util

    return importlib.util.find_spec("triton") is not None


def inference_settings(config: UMAConfig, device: str):
    """``(settings, reasons)``: fairchem's preset for the config's mode, adjusted for
    what this platform can run, with the reason for every adjustment."""
    import dataclasses as dc

    from fairchem.core.units.mlip_unit.api.inference import NAME_TO_INFERENCE_SETTING

    settings = dc.replace(NAME_TO_INFERENCE_SETTING[FAIRCHEM_SETTINGS[config.inference]])
    reasons = []
    if not triton_available():
        if device.startswith("cuda"):
            if settings.nitro:
                reasons.append("UMA Nitro's kernels need Triton, which has no official "
                               "build for this platform (Windows): Nitro cannot run")
            reasons.append("fairchem's umas_fast_gpu backend and torch.compile need Triton: "
                           "running the umas_fast_pytorch backend without compile")
            settings = dc.replace(settings, nitro=False, compile=False,
                                  execution_mode="umas_fast_pytorch")
        elif settings.compile or settings.nitro:
            if settings.nitro:
                reasons.append("UMA Nitro needs a CUDA GPU")
            reasons.append("torch.compile is off without Triton on the CPU")
            settings = dc.replace(settings, nitro=False, compile=False)
    return settings, reasons


def load_predict_unit(config: UMAConfig, *, device: str = "cuda", atom_refs=None):
    """Load the checkpoint with the config's inference settings; ``(unit, info)``.

    fairchem decides whether Nitro runs at the first evaluation, so read
    :func:`predict_unit_info` again after it.
    """
    import warnings

    from fairchem.core import pretrained_mlip

    settings, reasons = inference_settings(config, device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if config.model.lower().endswith(".pt"):
            unit = pretrained_mlip.load_predict_unit(
                config.model, inference_settings=settings, device=device,
                atom_refs=atom_refs)
        else:
            unit = pretrained_mlip.get_predict_unit(
                config.model, inference_settings=settings, device=device)
    info = predict_unit_info(unit, config)
    info["warnings"] = reasons + [str(w.message) for w in caught]
    return unit, info


def predict_unit_info(unit, config: UMAConfig) -> dict:
    """The inference mode that runs, named like fairchem's presets: ``nitro``;
    ``multigpu`` (merged MoLE, TF32: Nitro's own fallback); ``high-precision``
    (merged MoLE, FP32); ``general`` (unmerged, after a fixed-system fast path gave up)."""
    settings = getattr(unit, "inference_settings", None)
    nitro = bool(getattr(settings, "nitro", False))
    tf32 = bool(getattr(settings, "tf32", False))
    merged = bool(getattr(settings, "merge_mole", False))
    if nitro:
        effective = "nitro"
    elif merged:
        effective = "multigpu" if tf32 else "high-precision"
    else:
        effective = "general"
    return {
        "requested": config.inference,
        "effective": effective,
        "nitro": nitro,
        "tf32": tf32,
        "merge_mole": merged,
        "compile": bool(getattr(settings, "compile", False)),
        "execution_mode": str(getattr(settings, "execution_mode", None)),
        "dispersion": config.dispersion,
    }
