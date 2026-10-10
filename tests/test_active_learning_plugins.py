"""Characterization tests for CommitteeCalculator, element_offsets and plugin cache/validation.

No MACE, AIMNet2 or GPU is needed: cached-model paths return before any model is loaded.
"""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from samson_mlip_visualizer import active_learning as al


class Constant(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy, force):
        super().__init__()
        self._energy, self._force = energy, force

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": self._energy, "forces": np.full((len(atoms), 3), self._force)}


def dimer():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])


def test_committee_calculator_means_and_keeps_member_values():
    atoms = dimer()
    atoms.calc = al.CommitteeCalculator([Constant(1.0, 0.1), Constant(3.0, 0.3)])
    assert atoms.get_potential_energy() == pytest.approx(2.0)
    assert atoms.get_forces() == pytest.approx(np.full((2, 3), 0.2))
    results = atoms.calc.results
    assert results["energy_comm"].tolist() == [1.0, 3.0]
    assert results["forces_comm"].shape == (2, 2, 3)
    assert results["free_energy"] == results["energy"]


def test_element_offsets_recovers_exact_per_element_shifts():
    # E_ref = E_model + sum(n_Z * delta_Z) with delta_H = -0.5, delta_O = 2.0
    structures = [Atoms("H2"), Atoms("O2"), Atoms("H2O"), Atoms("H4O")]
    for atoms in structures:
        atoms.info["REF_energy_raw"] = 0.0
    delta = {1: -0.5, 8: 2.0}
    predicted = [-1.0, 4.0, 0.3, 1.1]
    for atoms, model_energy in zip(structures, predicted, strict=True):
        shift = sum(delta[int(z)] for z in atoms.numbers)
        atoms.info["REF_energy_raw"] = model_energy + shift
    offsets = al.element_offsets(structures, predicted)
    assert sorted(offsets) == [1, 8]
    assert offsets[1] == pytest.approx(-0.5) and offsets[8] == pytest.approx(2.0)


def test_element_offsets_underdetermined_gives_minimum_norm_solution():
    # One element, one structure: the least-squares solution splits the residual evenly per atom.
    atoms = Atoms("H2")
    atoms.info["REF_energy_raw"] = 1.0
    offsets = al.element_offsets([atoms], [0.0])
    assert offsets == {1: pytest.approx(0.5)}


def test_aimnet2_plugin_requires_two_members():
    with pytest.raises(ValueError, match="at least two"):
        al.AIMNet2Plugin("python", ["only.pt"])
    plugin = al.AIMNet2Plugin("python", ["a.pt", "b.pt"])
    assert plugin.members == ["a.pt", "b.pt"] and plugin.name == "aimnet2"


def test_aimnet2_train_reuses_existing_members(tmp_path):
    plugin = al.AIMNet2Plugin("python", ["a.pt", "b.pt"])
    for k in range(2):
        (tmp_path / f"member{k}.pt").write_bytes(b"x")
        (tmp_path / f"member{k}.log.json").write_text(json.dumps({"k": k}))
    model = plugin.train([dimer()], tmp_path, None)
    assert [p.rsplit("/", 1)[-1] for p in model.files] == ["member0.pt", "member1.pt"]
    assert model.info == {"logs": [{"k": 0}, {"k": 1}]}


def test_mace_train_reuses_existing_seed_models(tmp_path):
    plugin = al.MacePlugin("foundation.model", seeds=(1, 2))
    for seed in (1, 2):
        run = tmp_path / "runs" / f"seed{seed}"
        run.mkdir(parents=True)
        (run / f"al_seed{seed}.model").write_bytes(b"x")
    (tmp_path / "offsets.json").write_text(json.dumps({"offsets_ev": {"1": -0.1}}))
    model = plugin.train([dimer()], tmp_path, None)
    assert len(model.files) == 2 and model.info == {"offsets_ev": {"1": -0.1}}


def test_mace_plugin_defaults():
    plugin = al.MacePlugin("f.model")
    defaults = (plugin.seeds, plugin.epochs, plugin.device, plugin.mode)
    assert defaults == ((1, 2, 3), 120, "cuda", "plain")


def test_committee_needs_two_members():
    with pytest.raises(ValueError, match="at least two"):
        al.CommitteeCalculator([Constant(1.0, 0.0)])
