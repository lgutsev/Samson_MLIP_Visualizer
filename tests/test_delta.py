"""Δ-learning: residual labels, per-element energies, baseline + correction, card wrapping."""

import json
import sys
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from samson_mlip_visualizer import delta
from samson_mlip_visualizer.calculators import CalculatorLoadError, create_calculator
from samson_mlip_visualizer.delta import (
    DeltaCalculator,
    delta_baseline,
    delta_e0s,
    delta_labels,
    e0s_argument,
)


class Harmonic(Calculator):
    """E = offset·N + k Σ |r_i − r_0|², a stand-in for any method; ``members``
    fakes a committee (``energy_comm`` / ``forces_comm``)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k, offset=0.0, members=(), elements=None):
        super().__init__()
        self.k, self.offset, self.members = k, offset, members
        if elements:
            self.supported_elements = frozenset(elements)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        d = self.atoms.positions - self.atoms.positions[0]
        self.results = {"energy": self.offset * len(self.atoms) + self.k * float((d**2).sum()),
                        "forces": -2 * self.k * d}
        if self.members:
            self.results["energy_comm"] = [self.results["energy"] + m for m in self.members]
            self.results["forces_comm"] = [self.results["forces"] for _ in self.members]


def frames(n=5):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        atoms = Atoms("CNH", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.07]])
        atoms.positions += rng.normal(0, 0.05, (3, 3))
        out.append(atoms)
    return out


def reference_labeled(reference):
    labeled = []
    for atoms in frames():
        atoms.calc = reference
        atoms.info["REF_energy"] = atoms.get_potential_energy()
        atoms.arrays["REF_forces"] = atoms.get_forces()
        atoms.calc = None
        labeled.append(atoms)
    return labeled


def test_residual_labels_and_their_sum_give_back_the_reference():
    reference, baseline = Harmonic(3.0, offset=-10.0), Harmonic(2.0, offset=-4.0)
    data = delta_labels(reference_labeled(reference), baseline)
    for atoms in data:
        assert atoms.info["DELTA_energy"] == pytest.approx(
            atoms.info["REF_energy"] - atoms.info["BASE_energy"])
        np.testing.assert_allclose(atoms.arrays["DELTA_forces"],
                                   atoms.arrays["REF_forces"] - atoms.arrays["BASE_forces"])
        assert atoms.calc is None
    # One composition: the per-element energies add up to the mean residual.
    e0s = delta_e0s(data)
    assert e0s.keys() == {1, 6, 7}
    assert sum(e0s.values()) == pytest.approx(np.mean([a.info["DELTA_energy"] for a in data]))
    combined = DeltaCalculator(baseline, Harmonic(1.0, offset=-6.0))
    for atoms in data:
        atoms.calc = combined
        assert atoms.get_potential_energy() == pytest.approx(atoms.info["REF_energy"])
        np.testing.assert_allclose(atoms.get_forces(), atoms.arrays["REF_forces"], atol=1e-12)
        assert combined.results["baseline_energy"] == pytest.approx(atoms.info["BASE_energy"])


def test_committee_spread_survives_the_baseline():
    combined = DeltaCalculator(Harmonic(2.0, elements="HCNO"),
                               Harmonic(1.0, members=(-0.1, 0.0, 0.1), elements="HCN"))
    atoms = frames(1)[0]
    atoms.calc = combined
    energy = atoms.get_potential_energy()
    comm = combined.results["energy_comm"]
    assert np.mean(comm) == pytest.approx(energy)
    assert np.std(comm) == pytest.approx(np.std([-0.1, 0.0, 0.1]))
    assert combined.results["forces_comm"].shape == (3, 3, 3)
    assert combined.supported_elements == frozenset("HCN")  # both must know an element


def test_e0s_argument():
    assert e0s_argument({7: -1.5, 1: 0.25}) == "{1:0.2500000000,7:-1.5000000000}"


def _fake_mace(monkeypatch):
    class FakeMACE(Harmonic):
        def __init__(self, **kwargs):
            super().__init__(1.0)
            self.kwargs = kwargs

    package, calculators = ModuleType("mace"), ModuleType("mace.calculators")
    calculators.MACECalculator = FakeMACE
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    return FakeMACE


def _model(tmp_path, name, card=None):
    path = tmp_path / name
    path.write_bytes(b"weights")
    if card is not None:
        (tmp_path / (name + ".json")).write_text(json.dumps(card))
    return path


def test_a_correction_model_is_wrapped_in_its_baseline(monkeypatch, tmp_path):
    fake = _fake_mace(monkeypatch)
    xtb = tmp_path / "xtb.exe"
    xtb.write_bytes(b"")
    monkeypatch.setenv("XTB_EXE", str(xtb))
    baseline = {"program": "xtb", "method": "gfn1", "charge": 0, "multiplicity": 1}
    model = _model(tmp_path, "delta.model", {"delta_baseline": baseline, "scope": "HCN"})
    assert delta_baseline(model) == baseline
    calc = create_calculator("mace", model, device="cpu")
    assert isinstance(calc, DeltaCalculator) and isinstance(calc.correction, fake)
    assert calc.baseline.method == "gfn1" and calc.baseline.executable == xtb
    # An ordinary model stays as it is.
    plain = _model(tmp_path, "plain.model", {"scope": "anything"})
    assert delta_baseline(plain) is None
    assert isinstance(create_calculator("mace", plain, device="cpu"), fake)
    # A committee mixing both is refused rather than silently half-corrected.
    with pytest.raises(CalculatorLoadError, match="baselines"):
        create_calculator("mace", [model, plain], device="cpu")


def test_missing_xtb_for_a_correction_model_is_clear(monkeypatch, tmp_path):
    _fake_mace(monkeypatch)
    monkeypatch.setattr("samson_mlip_visualizer.xtb_backend.find_xtb", lambda: None)
    model = _model(tmp_path, "delta.model", {"delta_baseline": {"program": "xtb"}})
    with pytest.raises(CalculatorLoadError, match="Δ-learning correction"):
        create_calculator("mace", model, device="cpu")


def test_baseline_card_records_the_method():
    card = delta.xtb_baseline_card("GFN1-xTB", charge=-1)
    assert card == {"program": "xtb", "method": "gfn1", "charge": -1, "multiplicity": 1,
                    "solvent": None}
