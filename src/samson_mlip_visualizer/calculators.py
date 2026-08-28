"""Lazy ASE calculator construction for supported MLIP backends."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Backend = Literal["mace", "deepmd"]


class CalculatorLoadError(RuntimeError):
    """Raised when an MLIP calculator cannot be imported or initialized."""


def _assert_cuda_available() -> None:
    """Fail early with a clear message when 'cuda' is picked without a GPU."""
    try:
        import torch
    except ImportError:
        return  # A missing torch is reported by the backend import path.
    if not torch.cuda.is_available():
        raise CalculatorLoadError(
            "device='cuda' was requested but torch reports no usable CUDA device "
            "(torch.cuda.is_available() is False). Select 'cpu', or repair the "
            "CUDA-enabled PyTorch installation in SAMSON's Python environment."
        )


def create_calculator(
    backend: Backend,
    model_path: str | Path,
    *,
    device: str = "cpu",
    dtype: str = "float64",
):
    """Create an ASE calculator without importing unused ML frameworks.

    Parameters are deliberately small and stable so the SAMSON UI does not
    depend on backend-specific implementation details.
    """
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise CalculatorLoadError(f"Model file does not exist: {path}")

    try:
        if backend == "mace":
            from mace.calculators import MACECalculator

            if str(device).startswith("cuda"):
                _assert_cuda_available()
            return MACECalculator(
                model_paths=str(path),
                device=device,
                default_dtype=dtype,
            )
        if backend == "deepmd":
            from deepmd.calculator import DP

            return DP(model=str(path))
    except CalculatorLoadError:
        raise
    except ImportError as exc:
        package = "mace-torch" if backend == "mace" else "deepmd-kit"
        raise CalculatorLoadError(
            f"The {backend.upper()} backend is not installed. Install '{package}' "
            "in SAMSON's Python package manager."
        ) from exc
    except Exception as exc:
        message = f"Could not load {backend.upper()} model '{path}': {exc}"
        raise CalculatorLoadError(message) from exc

    raise ValueError(f"Unsupported backend: {backend!r}")
