"""Lazy ASE calculator construction for supported MLIP backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

Backend = Literal["mace", "deepmd", "xtb"]

ModelPaths = str | Path | Sequence[str | Path]


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


def _resolve_model_paths(model_path: ModelPaths) -> list[Path]:
    entries = [model_path] if isinstance(model_path, (str, Path)) else list(model_path)
    if not entries:
        raise CalculatorLoadError("No model file was provided")
    resolved: list[Path] = []
    for entry in entries:
        path = Path(entry).expanduser()
        if not path.is_file():
            raise CalculatorLoadError(f"Model file does not exist: {path}")
        resolved.append(path)
    return resolved


def create_calculator(
    backend: Backend,
    model_path: ModelPaths,
    *,
    device: str = "cpu",
    dtype: str = "float64",
    xtb: Mapping[str, Any] | None = None,
):
    """Create an ASE calculator without importing unused ML frameworks.

    ``model_path`` may be a single file or several. Passing several MACE
    checkpoints builds a committee: ``atoms.calc.results`` then carries
    ``energy_comm`` / ``forces_comm``, whose spread is an extrapolation signal.
    For ``backend="xtb"`` the "model" is the xtb executable, and ``xtb`` holds
    the :class:`~.xtb_backend.XTBCalculator` options (method, charge,
    multiplicity, solvent); device and dtype do not apply.
    """
    paths = _resolve_model_paths(model_path)

    if backend == "xtb":
        from .xtb_backend import XTBCalculator, is_xtb_executable

        if len(paths) > 1 or not is_xtb_executable(paths[0]):
            raise CalculatorLoadError(
                "For the xTB backend, choose the xtb executable (xtb.exe) as the model "
                "file, e.g. from `micromamba create -n xtb -c conda-forge xtb`."
            )
        try:
            return XTBCalculator(paths[0], **dict(xtb or {}))
        except (TypeError, ValueError) as exc:
            raise CalculatorLoadError(f"Invalid xTB settings: {exc}") from exc

    try:
        if backend == "mace":
            from mace.calculators import MACECalculator

            if str(device).startswith("cuda"):
                _assert_cuda_available()
            model_arg = str(paths[0]) if len(paths) == 1 else [str(p) for p in paths]
            return MACECalculator(
                model_paths=model_arg,
                device=device,
                default_dtype=dtype,
            )
        if backend == "deepmd":
            from deepmd.calculator import DP

            if len(paths) > 1:
                raise CalculatorLoadError(
                    "DeepMD committee evaluation is not supported yet; select one model file."
                )
            return DP(model=str(paths[0]))
    except CalculatorLoadError:
        raise
    except ImportError as exc:
        package = "mace-torch" if backend == "mace" else "deepmd-kit"
        raise CalculatorLoadError(
            f"The {backend.upper()} backend could not be imported ({exc}). If "
            f"'{package}' is not installed, install it in SAMSON's Python package manager."
        ) from exc
    except Exception as exc:
        message = f"Could not load {backend.upper()} model '{paths[0]}': {exc}"
        raise CalculatorLoadError(message) from exc

    raise ValueError(f"Unsupported backend: {backend!r}")
