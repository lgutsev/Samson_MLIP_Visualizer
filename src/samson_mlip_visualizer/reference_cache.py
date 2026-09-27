"""A reference calculator that computes each structure once.

Reference calculations (DFT) dominate the cost of benchmarking and labeling.
:class:`CachedReference` wraps any calculator factory and keeps every result in a
JSON file keyed by the geometry, the atoms, and the charge, so the same frame
evaluated for several models, labeled after being benchmarked, or reached again
by an interrupted run started over costs nothing the second time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def geometry_key(numbers, positions, charge: int) -> str:
    """Identifies a structure to 1e-5 Å (rounding noise in files stays below that)."""
    rounded = np.round(np.asarray(positions, float), 5) + 0.0  # + 0.0 turns -0.0 into 0.0
    data = f"{int(charge)}|{list(map(int, numbers))}|".encode() + rounded.tobytes()
    return hashlib.sha1(data).hexdigest()


def psi4_factory(method: str, basis: str, *, threads: int = 8, memory_mb: int = 1900,
                 python: str | Path | None = None) -> Callable[[int], Calculator]:
    """A factory ``charge -> Psi4Calculator``. Keep ``memory_mb`` below 2048 on
    Windows: conda-forge Psi4 1.11 coupled-cluster codes fail above 2 GiB."""

    def factory(charge: int) -> Calculator:
        from .psi4_backend import Psi4Calculator, find_psi4

        return Psi4Calculator(python or find_psi4(), method=method, basis=basis,
                              charge=charge, threads=threads, memory_mb=memory_mb)

    return factory


class CachedReference(Calculator):
    """Energies and forces from ``factory(charge)``, each structure computed once.

    Results live in ``cache_file`` (JSON), written after every new calculation;
    ``save`` merges with what is on disk first, so several scripts can share one
    cache. ``computed`` counts the calculations this instance actually ran.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, cache_file: str | Path, factory: Callable[[int], Calculator],
                 charge: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.cache_file = Path(cache_file)
        self.factory, self.charge = factory, int(charge)
        self.cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        self.computed = 0
        self._inner = None

    def add(self, numbers, positions, energy: float, forces) -> None:
        """Record a result computed elsewhere (e.g. existing labels)."""
        key = geometry_key(numbers, positions, self.charge)
        self.cache[key] = {"energy_ev": float(energy), "forces": np.asarray(forces).tolist()}

    def has(self, atoms) -> bool:
        return geometry_key(atoms.numbers, atoms.positions, self.charge) in self.cache

    def save(self) -> None:
        if self.cache_file.exists():
            self.cache = {**json.loads(self.cache_file.read_text()), **self.cache}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.cache))
        temporary.replace(self.cache_file)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        key = geometry_key(self.atoms.numbers, self.atoms.positions, self.charge)
        if key not in self.cache:
            if self._inner is None:
                self._inner = self.factory(self.charge)
            probe = self.atoms.copy()
            probe.calc = self._inner
            self.add(probe.numbers, probe.positions, probe.get_potential_energy(),
                     probe.get_forces())
            self.computed += 1
            self.save()
        entry = self.cache[key]
        energy = entry["energy_ev"]
        self.results = {"energy": energy, "free_energy": energy,
                        "forces": np.array(entry["forces"])}
