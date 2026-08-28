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


def _install_fake_mace(monkeypatch, calculator_cls):
    package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = calculator_cls
    monkeypatch.setitem(__import__("sys").modules, "mace", package)
    monkeypatch.setitem(__import__("sys").modules, "mace.calculators", calculators)


def _install_fake_torch(monkeypatch, *, cuda_available):
    torch = ModuleType("torch")
    torch.cuda = ModuleType("torch.cuda")
    torch.cuda.is_available = lambda: cuda_available
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)


def test_mace_factory_is_lazy(monkeypatch, tmp_path):
    model = tmp_path / "model.model"
    model.write_bytes(b"test")
    captured = {}

    class FakeMACECalculator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    _install_fake_mace(monkeypatch, FakeMACECalculator)
    _install_fake_torch(monkeypatch, cuda_available=True)

    calculator = create_calculator("mace", model, device="cuda", dtype="float32")

    assert isinstance(calculator, FakeMACECalculator)
    assert captured == {
        "model_paths": str(Path(model)),
        "device": "cuda",
        "default_dtype": "float32",
    }


def test_mace_cuda_without_device_is_rejected(monkeypatch, tmp_path):
    model = tmp_path / "model.model"
    model.write_bytes(b"test")

    class FakeMACECalculator:
        def __init__(self, **kwargs):
            raise AssertionError("should not be constructed without a CUDA device")

    _install_fake_mace(monkeypatch, FakeMACECalculator)
    _install_fake_torch(monkeypatch, cuda_available=False)

    with pytest.raises(CalculatorLoadError, match="no usable CUDA device"):
        create_calculator("mace", model, device="cuda")


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


def test_mace_committee_passes_a_list_of_paths(monkeypatch, tmp_path):
    models = [tmp_path / "a.model", tmp_path / "b.model"]
    for model in models:
        model.write_bytes(b"test")
    captured = {}

    class FakeMACECalculator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    _install_fake_mace(monkeypatch, FakeMACECalculator)

    create_calculator("mace", models, device="cpu")

    assert captured["model_paths"] == [str(m) for m in models]


def test_deepmd_committee_is_rejected(monkeypatch, tmp_path):
    models = [tmp_path / "a.pb", tmp_path / "b.pb"]
    for model in models:
        model.write_bytes(b"test")

    package = ModuleType("deepmd")
    calculator_module = ModuleType("deepmd.calculator")
    calculator_module.DP = object
    monkeypatch.setitem(__import__("sys").modules, "deepmd", package)
    monkeypatch.setitem(__import__("sys").modules, "deepmd.calculator", calculator_module)

    with pytest.raises(CalculatorLoadError, match="not supported yet"):
        create_calculator("deepmd", models)


def test_missing_model_in_committee_is_reported(monkeypatch, tmp_path):
    good = tmp_path / "a.model"
    good.write_bytes(b"test")
    with pytest.raises(CalculatorLoadError, match="does not exist"):
        create_calculator("mace", [good, tmp_path / "missing.model"])
