"""Record exactly which model and software produced a result.

An MLIP number is only interpretable next to the model file and runtime that
produced it. These helpers are pure so they can be logged in the panel, written
into an output structure's metadata by the CLI, and unit tested without SAMSON.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_BACKEND_DISTRIBUTIONS = {
    "mace": "mace-torch",
    "deepmd": "deepmd-kit",
}


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def model_digest(model_path: str | Path, *, chunk_size: int = 1 << 20) -> tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` for a model file."""
    path = Path(model_path).expanduser()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


@dataclass(frozen=True)
class Provenance:
    backend: str
    model_path: str
    model_sha256: str
    model_size_bytes: int
    device: str
    dtype: str
    created_utc: str
    versions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        flat = {
            "mlip_backend": self.backend,
            "mlip_model_path": self.model_path,
            "mlip_model_sha256": self.model_sha256,
            "mlip_model_size_bytes": str(self.model_size_bytes),
            "mlip_device": self.device,
            "mlip_dtype": self.dtype,
            "mlip_created_utc": self.created_utc,
        }
        for name, value in self.versions.items():
            flat[f"mlip_version_{name}"] = value
        return flat

    def as_text(self) -> str:
        lines = [
            f"backend         {self.backend}",
            f"model           {self.model_path}",
            f"model sha256    {self.model_sha256}",
            f"model size      {self.model_size_bytes} bytes",
            f"device          {self.device}",
            f"dtype           {self.dtype}",
            f"created (UTC)   {self.created_utc}",
        ]
        for name, value in self.versions.items():
            lines.append(f"{name:<15} {value}")
        return "\n".join(lines)


def collect_provenance(
    *,
    backend: str,
    model_path: str | Path,
    device: str,
    dtype: str,
) -> Provenance:
    sha256, size = model_digest(model_path)
    versions: dict[str, str] = {}
    for dist in ("samson-mlip-visualizer", "ase", "numpy"):
        found = _distribution_version(dist)
        if found:
            versions[dist] = found
    backend_dist = _BACKEND_DISTRIBUTIONS.get(backend)
    if backend_dist:
        found = _distribution_version(backend_dist)
        if found:
            versions[backend_dist] = found
    return Provenance(
        backend=backend,
        model_path=str(Path(model_path).expanduser()),
        model_sha256=sha256,
        model_size_bytes=size,
        device=device,
        dtype=dtype,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        versions=versions,
    )
