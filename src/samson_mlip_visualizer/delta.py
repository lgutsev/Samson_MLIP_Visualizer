"""Δ-learning: a tight-binding baseline plus a machine-learned correction.

Instead of learning the whole potential energy surface, a small MLIP learns the
difference between a reference method and a cheap quantum-mechanical baseline
(GFN-xTB here):

    E(x) = E_xTB(x) + ΔE_ML(x),    ΔE_ML ≈ E_reference − E_xTB

The baseline brings the physics a local MLIP lacks (charges, polarization, a
qualitatively right bond breaking); the correction only has to learn a smoother,
smaller residual, so it needs fewer reference labels. The cost of a step is
the xTB calculation plus one MLIP evaluation. AIQM2 (GFN2-xTB + ANI) and QDπ
(DFTB3 + DeepPot-SE) are published models of this form.

- :func:`delta_labels`: reference-labeled frames -> frames labeled with the
  residual (``DELTA_energy`` / ``DELTA_forces``), plus the baseline's numbers;
- :func:`delta_e0s`: per-element energies for the correction model (the
  residual's per-element offsets), so it only learns what varies;
- :class:`DeltaCalculator`: baseline + correction as one ASE calculator; a
  correction committee keeps its spread (``energy_comm`` / ``forces_comm``);
- :func:`delta_baseline`: what a correction model's card says its baseline is.
  :func:`~.calculators.create_calculator` reads it and wraps a correction model
  in its baseline automatically: on its own, a correction is not a potential.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers

from .compat import supported_species
from .finetune import fit_element_offsets

BASELINE_KEY = "delta_baseline"  # in a model card: the baseline the correction needs
ENERGY_KEY = "DELTA_energy"
FORCES_KEY = "DELTA_forces"


class DeltaCalculator(Calculator):
    """``baseline`` + ``correction``: energies and forces add.

    ``results`` also carries ``baseline_energy`` and ``correction_energy``; a
    correction committee's ``energy_comm`` / ``forces_comm`` are shifted by the
    baseline, so their spread (the extrapolation signal) is unchanged.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, baseline, correction, **kwargs):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.correction = correction
        elements = [supported_species(calc) for calc in (baseline, correction)]
        known = [set(found) for found in elements if found is not None]
        if known:
            self.supported_elements = frozenset(set.intersection(*known))

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        parts = []
        for calc in (self.baseline, self.correction):
            image = self.atoms.copy()
            image.calc = calc
            parts.append((image.get_potential_energy(), image.get_forces(), calc.results))
        (e_base, f_base, _), (e_corr, f_corr, corr) = parts
        self.results = {
            "energy": e_base + e_corr,
            "free_energy": e_base + e_corr,
            "forces": f_base + f_corr,
            "baseline_energy": e_base,
            "correction_energy": e_corr,
        }
        if corr.get("energy_comm") is not None:
            self.results["energy_comm"] = np.asarray(corr["energy_comm"], float) + e_base
        if corr.get("forces_comm") is not None:
            self.results["forces_comm"] = np.asarray(corr["forces_comm"], float) + f_base


def delta_labels(
    frames: Sequence[Atoms],
    baseline,
    *,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
) -> list[Atoms]:
    """Copies of reference-labeled ``frames`` with the residual as the target.

    Each copy has ``DELTA_energy`` = E_ref − E_base and ``DELTA_forces`` =
    F_ref − F_base (the keys mace-torch is told to read), and keeps the
    reference (``REF_energy`` / ``REF_forces``) and the baseline
    (``BASE_energy`` / ``BASE_forces``). A frame the baseline cannot evaluate
    raises: a residual with a failed baseline is meaningless.
    """
    labeled = []
    for frame in frames:
        image = Atoms(frame.get_chemical_symbols(), positions=frame.positions,
                      cell=frame.cell, pbc=frame.pbc)
        image.calc = baseline
        e_base, f_base = image.get_potential_energy(), image.get_forces()
        image.calc = None
        e_ref = float(frame.info[energy_key])
        f_ref = np.asarray(frame.arrays[forces_key], float)
        image.info.update(frame.info)
        image.info["REF_energy"] = e_ref
        image.info["BASE_energy"] = float(e_base)
        image.info[ENERGY_KEY] = e_ref - float(e_base)
        image.arrays["REF_forces"] = f_ref
        image.arrays["BASE_forces"] = np.asarray(f_base, float)
        image.arrays[FORCES_KEY] = f_ref - f_base
        labeled.append(image)
    return labeled


def delta_e0s(frames: Sequence[Atoms]) -> dict[int, float]:
    """Per-element energies (eV, by atomic number) for the correction model.

    Least-squares offsets of the residual ``DELTA_energy`` per element (see
    :func:`~.finetune.fit_element_offsets`). With one composition they are not
    unique (the minimum-norm solution is used); any solution fits the
    compositions in the data, which are the only ones the correction can be
    trusted for anyway.
    """
    offsets = fit_element_offsets(
        frames,
        [frame.info[ENERGY_KEY] for frame in frames],
        [0.0] * len(frames),
    )
    return {atomic_numbers[symbol]: value for symbol, value in offsets.offsets.items()}


def e0s_argument(e0s: Mapping[int, float]) -> str:
    """mace-torch's ``--E0s`` value for fixed per-element energies."""
    return "{" + ",".join(f"{int(z)}:{float(e):.10f}" for z, e in sorted(e0s.items())) + "}"


def xtb_baseline_card(method: str = "gfn2", *, charge: int = 0, multiplicity: int = 1,
                      solvent: str | None = None, executable: str | Path | None = None) -> dict:
    """The ``delta_baseline`` entry for a model card: what the correction was
    trained on top of (the xtb version too, when the executable is given)."""
    from .xtb_backend import normalize_method, xtb_version

    card: dict[str, Any] = {"program": "xtb", "method": normalize_method(method),
                            "charge": int(charge), "multiplicity": int(multiplicity),
                            "solvent": solvent or None}
    if executable is not None:
        card["xtb_version"] = xtb_version(executable)
    return card


def delta_baseline(model_path: str | Path) -> dict | None:
    """The baseline a correction model needs (its card's ``delta_baseline``), or
    ``None`` for an ordinary model."""
    import json

    card = Path(str(Path(model_path).expanduser()) + ".json")
    try:
        data = json.loads(card.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    baseline = data.get(BASELINE_KEY) if isinstance(data, dict) else None
    return baseline if isinstance(baseline, dict) else None


def baseline_calculator(baseline: Mapping[str, Any], executable: str | Path | None = None):
    """The calculator a ``delta_baseline`` card entry describes."""
    from .xtb_backend import XTBCalculator, find_xtb

    if baseline.get("program") != "xtb":
        raise ValueError(f"Unsupported Δ-learning baseline: {baseline.get('program')!r}")
    executable = executable or find_xtb()
    if executable is None:
        raise FileNotFoundError(
            "This model is a Δ-learning correction on top of GFN-xTB, and no xtb "
            "executable was found (set XTB_EXE, or `micromamba create -n xtb -c "
            "conda-forge xtb`)."
        )
    return XTBCalculator(
        executable,
        method=baseline.get("method", "gfn2"),
        charge=int(baseline.get("charge", 0)),
        multiplicity=int(baseline.get("multiplicity", 1)),
        solvent=baseline.get("solvent"),
    )
