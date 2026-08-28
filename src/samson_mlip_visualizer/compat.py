"""Best-effort check that a model's training elements cover the structure.

A local MLIP is only meaningful for the chemical elements it was trained on.
Neither MACE nor DeepMD expose this through a single stable attribute, so the
introspection here is deliberately defensive: when the supported set cannot be
determined the check yields ``None`` and the caller proceeds unguarded rather
than blocking a valid run.
"""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.data import chemical_symbols


class ModelCompatibilityError(RuntimeError):
    """Raised when a structure contains elements a model was not trained on."""


def _symbols_from_atomic_numbers(numbers: Any) -> set[str]:
    out: set[str] = set()
    for value in numbers:
        z = int(value)
        if 0 < z < len(chemical_symbols):
            out.add(chemical_symbols[z])
    return out


def supported_species(calculator: Any) -> set[str] | None:
    """Return the element symbols a calculator's model supports, or ``None``.

    ``None`` means "could not determine" and must not be treated as "supports
    nothing".
    """
    # MACE: AtomicNumberTable on the calculator, or an atomic_numbers buffer on
    # the wrapped model(s).
    z_table = getattr(calculator, "z_table", None)
    zs = getattr(z_table, "zs", None)
    if zs:
        return _symbols_from_atomic_numbers(zs)

    models = getattr(calculator, "models", None)
    if models:
        numbers = getattr(models[0], "atomic_numbers", None)
        if numbers is not None and len(numbers):
            return _symbols_from_atomic_numbers(numbers)

    # DeepMD: a type map of element symbols, on the DP wrapper or its backend.
    for holder in (calculator, getattr(calculator, "dp", None)):
        if holder is None:
            continue
        getter = getattr(holder, "get_type_map", None)
        type_map = getter() if callable(getter) else getattr(holder, "type_map", None)
        if type_map:
            return {str(symbol) for symbol in type_map}

    return None


def incompatible_species(atoms: Atoms, supported: set[str]) -> list[str]:
    """Element symbols present in ``atoms`` but absent from ``supported``."""
    present = set(atoms.get_chemical_symbols())
    return sorted(present - set(supported))


def assert_model_covers_structure(calculator: Any, atoms: Atoms) -> list[str] | None:
    """Refuse the run if the model demonstrably lacks a structure element.

    Returns the sorted supported-element list when it is known (useful for
    logging), or ``None`` when the model could not be introspected.
    """
    supported = supported_species(calculator)
    if supported is None:
        return None
    missing = incompatible_species(atoms, supported)
    if missing:
        raise ModelCompatibilityError(
            "The selected model was not trained for: "
            f"{', '.join(missing)}. Supported elements: {', '.join(sorted(supported))}. "
            "Evaluating a structure outside a model's training elements produces a "
            "number that is easy to misinterpret, so this is refused."
        )
    return sorted(supported)
