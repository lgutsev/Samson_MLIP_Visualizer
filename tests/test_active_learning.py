"""The active-learning loop with toy models: stopping, selection, held-out frames, resuming."""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from test_benchmark import Spring

from samson_mlip_visualizer import active_learning as al
from samson_mlip_visualizer.finetune import structure_distance
from samson_mlip_visualizer.reference_cache import CachedReference


def h2(r, **info):
    return Atoms("H2", positions=[[0, 0, 0], [r, 0, 0]], info=info)


class ToyPlugin:
    """A two-member committee of springs around the reference (k = 1) whose
    error halves once any labels have been added (seed frames excluded)."""

    name = "toy"

    def __init__(self):
        self.trainings = 0

    def train(self, data, directory, previous):
        directory = Path(directory)
        path = directory / "model.json"
        if not path.exists():
            self.trainings += 1
            labels = sum(1 for a in data if str(a.info.get("tag", "")).startswith("round"))
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"error": 0.5 if labels == 0 else 0.25}))
        return al.TrainedModel([str(path)], json.loads(path.read_text()))

    def calculator(self, model, charge):
        error = model.info["error"]
        return al.CommitteeCalculator([Spring(k=1 + 0.9 * error), Spring(k=1 + 1.1 * error)])

    def uncertainty(self, model, frames, charge):
        return al.committee_spread(frames, self.calculator(model, charge))


def toy_explorer(calls):
    def explorer(calc, numbers, start, options):
        calls.append(1)
        irc = [h2(r, irc_arc=r - 1.1) for r in np.linspace(0.6, 1.6, 21)]
        scan = [h2(r, scan_distance=r) for r in np.linspace(1.0, 2.0, 11)]
        return al.Exploration(np.array(start), {"converged": True, "n_imaginary": 1,
                                                "imaginary_cm": -500.0, "energy_ev": 0.0},
                              irc, scan, [])

    return explorer


@pytest.fixture
def setup(tmp_path):
    reference = CachedReference(tmp_path / "cache.json", lambda charge: Spring(), charge=0)
    seed = []
    for r in (0.9, 1.0, 1.1):
        atoms = h2(r)
        atoms.calc = Spring()
        labeled = h2(r, raw_energy=atoms.get_potential_energy(), tag="seed")
        labeled.arrays["REF_forces"] = atoms.get_forces()
        seed.append(labeled)
    write(tmp_path / "seed.extxyz", seed)
    write(tmp_path / "ts.xyz", h2(1.3))
    config = al.Config(
        output=str(tmp_path / "run"), model={"plugin": "toy"}, reference={},
        seed_data=[str(tmp_path / "seed.extxyz")],
        explore={"ts_start": str(tmp_path / "ts.xyz"), "held_out_points": 6},
        tolerances={"barrier_kcal": 0.5, "path_max_ev": 0.03, "scan_max_ev": 0.2},
        budget={"committee": 2, "diverse": 2, "random": 1, "rattle_copies": 1,
                "rattle_sigma": 0.02, "min_distance": 0.05, "max_rounds": 3},
    )
    return config, reference


def test_stop_decision_is_reference_checked():
    good = {"barrier_error_kcal": 0.2, "irc": {"energy_error_max_abs_ev": 0.005},
            "scan": {"energy_error_max_abs_ev": 0.01}}
    tolerances = {"barrier_kcal": 0.5, "path_max_ev": 0.01, "scan_max_ev": 0.02}
    assert al.stop_decision(good, tolerances) == (True, [])
    bad = {**good, "scan": {"energy_error_max_abs_ev": 0.05}, "barrier_error_kcal": -0.8}
    stop, failed = al.stop_decision(bad, tolerances)
    assert not stop and len(failed) == 2 and "barrier" in failed[0] and "scan" in failed[1]
    # No scan explored: only the barrier and the IRC count.
    assert al.stop_decision({k: v for k, v in good.items() if k != "scan"}, tolerances)[0]


def test_loop_selects_then_stops_on_reference_errors(setup):
    config, reference = setup
    calls = []
    loop = al.ActiveLearning(config, ToyPlugin(), reference, explorer=toy_explorer(calls),
                             log=lambda *_: None)
    rows = loop.run()
    assert [r["status"] for r in rows] == ["continued", "converged"]
    first = rows[0]
    assert first["failed"] and first["selected"]["selected"] > 0
    assert first["labeled"] == 2 * first["selected"]["selected"]  # one rattled copy each
    # Committee spread is recorded next to the real error, not used to stop.
    assert "committee_energy_std_max_ev" in first["irc"]
    # The selection never takes a held-out (evaluated) frame...
    held = {f"irc frame {k}" for k in first["irc"]["frame_indices"]}
    held |= {f"scan frame {k}" for k in first["scan"]["frame_indices"]}
    sources = [f["source"] for f in first["selected"]["frames"]]
    assert not held & set(sources)
    assert {f["reason"] for f in first["selected"]["frames"]} <= {"committee", "diversity",
                                                                  "spot-check"}
    # ...and keeps min_distance from each other and from the training data.
    selected = read(Path(config.output) / "round_00" / "select" / "frames.extxyz", ":")
    seed = read(config.seed_data[0], ":")
    for i, a in enumerate(selected):
        assert min(structure_distance(a, s) for s in seed) >= 0.05
        for b in selected[i + 1:]:
            assert structure_distance(a, b) >= 0.05
    # Round 1 trained on the seed plus the new labels.
    assert rows[1]["training_structures"] == len(seed) + first["labeled"]


def test_resume_skips_finished_stages(setup):
    config, reference = setup
    calls, plugin = [], ToyPlugin()
    loop = al.ActiveLearning(config, plugin, reference, explorer=toy_explorer(calls),
                             log=lambda *_: None)
    rows = loop.run(max_rounds=1)
    assert [r["status"] for r in rows] == ["not converged: max rounds reached"]
    assert plugin.trainings == 1 and len(calls) == 1
    computed = reference.computed
    # A new process: nothing of round 0 is recomputed, round 1 runs.
    again = ToyPlugin()
    reference2 = CachedReference(reference.cache_file, lambda charge: Spring(), charge=0)
    rows = al.ActiveLearning(config, again, reference2, explorer=toy_explorer(calls),
                             log=lambda *_: None).run(max_rounds=3)
    assert [r["round"] for r in rows] == [0, 1] and rows[-1]["status"] == "converged"
    assert again.trainings == 1  # round 1 only
    assert len(calls) == 2  # round 1's exploration only
    assert computed > 0 and reference2.computed < computed  # the cache carried over


def test_pieces():
    committee = al.CommitteeCalculator([Spring(k=1.0), Spring(k=3.0)])
    atoms = h2(1.5)
    atoms.calc = committee
    assert atoms.get_potential_energy() == pytest.approx(0.5 * 2.0 * 0.25)
    assert committee.results["energy_comm"] == pytest.approx([0.125, 0.375])
    with pytest.raises(ValueError):
        al.CommitteeCalculator([Spring()])
    assert al.barrier_from_higher_end(np.array([0.1, 0.5, -1.0])) == pytest.approx(0.4)
    assert al.barrier_from_higher_end(np.array([-1.0, 0.5, 0.1])) == pytest.approx(0.4)


def test_load_labeled_and_config(tmp_path):
    frame = h2(1.0, raw_energy=-1.5, charge=-1)
    frame.arrays["REF_forces"] = np.zeros((2, 3))
    write(tmp_path / "a.extxyz", frame)
    (loaded,) = al.load_labeled([tmp_path / "a.extxyz"], charge=0)
    assert loaded.info["REF_energy_raw"] == -1.5 and loaded.info["charge"] == -1
    bare = h2(1.0)
    write(tmp_path / "b.extxyz", bare)
    with pytest.raises(ValueError, match="raw reference energy"):
        al.load_labeled([tmp_path / "b.extxyz"], charge=0)
    (tmp_path / "c.json").write_text(json.dumps({
        "output": "x", "model": {"plugin": "mace", "foundation": "f"}, "reference": {},
        "seed_data": [], "explore": {}, "tolerances": {"scan_max_ev": 0.05}}))
    config = al.Config.load(tmp_path / "c.json")
    assert config.tolerances == {"barrier_kcal": 0.5, "path_max_ev": 0.010, "scan_max_ev": 0.05}
    assert config.budget["max_rounds"] == 4
    assert isinstance(al.make_plugin(config.model), al.MacePlugin)
    with pytest.raises(ValueError, match="Unknown model"):
        al.make_plugin({"plugin": "nope"})
