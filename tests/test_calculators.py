from pathlib import Path
from types import ModuleType

import pytest

from samson_mlip_visualizer.calculators import CalculatorLoadError, create_calculator


def test_missing_model_is_clear(tmp_path):
    with pytest.raises(CalculatorLoadError, match="does not exist"):
        create_calculator("mace", tmp_path / "missing.model")


def test_unknown_backend(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"test")
    with pytest.raises(ValueError, match="Unsupported backend"):
        create_calculator("unknown", model)


def test_mace_factory_is_lazy(monkeypatch, tmp_path):
    model = tmp_path / "model.model"
    model.write_bytes(b"test")
    captured = {}

    class FakeMACECalculator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = FakeMACECalculator
    monkeypatch.setitem(__import__("sys").modules, "mace", package)
    monkeypatch.setitem(__import__("sys").modules, "mace.calculators", calculators)

    calculator = create_calculator("mace", model, device="cuda", dtype="float32")

    assert isinstance(calculator, FakeMACECalculator)
    assert captured == {
        "model_paths": str(Path(model)),
        "device": "cuda",
        "default_dtype": "float32",
    }


def test_deepmd_factory_is_lazy(monkeypatch, tmp_path):
    model = tmp_path / "model.pb"
    model.write_bytes(b"test")
    captured = {}

    class FakeDP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    package = ModuleType("deepmd")
    calculator_module = ModuleType("deepmd.calculator")
    calculator_module.DP = FakeDP
    monkeypatch.setitem(__import__("sys").modules, "deepmd", package)
    monkeypatch.setitem(__import__("sys").modules, "deepmd.calculator", calculator_module)

    calculator = create_calculator("deepmd", model)

    assert isinstance(calculator, FakeDP)
    assert captured == {"model": str(Path(model))}
