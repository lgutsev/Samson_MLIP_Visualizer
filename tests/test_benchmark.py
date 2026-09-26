"""Benchmarking a model against a reference along a path."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from samson_mlip_visualizer import benchmark
from samson_mlip_visualizer.benchmark import (
    basis_name,
    benchmark_path,
    bond_angle,
    select_frames,
    write_report,
)


class Spring(Calculator):
    """A harmonic bond, optionally reporting a fake committee."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k=1.0, r0=1.0, committee=False):
        super().__init__()
        self.k, self.r0, self.committee = k, r0, committee

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        vector = atoms.positions[1] - atoms.positions[0]
        r = np.linalg.norm(vector)
        force = self.k * (r - self.r0) * vector / r
        forces = np.array([force, -force])
        self.results = {"energy": 0.5 * self.k * (r - self.r0) ** 2, "forces": forces}
        if self.committee:
            self.results["energy_comm"] = [self.results["energy"], self.results["energy"] + r]
            self.results["forces_comm"] = [forces, forces * 1.1]


def path(count=11):
    return [Atoms("H2", positions=[[0, 0, 0], [r, 0, 0]]) for r in np.linspace(1.0, 2.0, count)]


def test_select_frames_spreads_along_the_coordinate():
    assert select_frames(5, None) == [0, 1, 2, 3, 4]
    # Frames crowded near the end: spacing by coordinate, not index.
    coordinate = [0.0, 1.0, 1.8, 2.9, 2.95, 3.0]
    assert select_frames(6, 3, coordinate=coordinate) == [0, 2, 5]
    assert select_frames(6, 3, keep=[3], coordinate=coordinate) == [0, 2, 3, 5]


def test_benchmark_path_errors_are_relative_to_the_first_frame():
    frames = path()
    bench = benchmark_path(frames, Spring(k=2.0), Spring(k=1.0), names=("fast", "slow"))
    r = np.linspace(1.0, 2.0, 11)
    assert bench.relative("fast") == pytest.approx((r - 1.0) ** 2)
    assert bench.energy_error == pytest.approx(0.5 * (r - 1.0) ** 2)
    # Each atom's force differs by k·Δr: mean and max over the two atoms agree.
    assert bench.force_error_mean == pytest.approx(r - 1.0)
    assert bench.force_error_max == pytest.approx(r - 1.0)
    summary = bench.summary()
    assert summary["energy_error_max_ev"] == pytest.approx(0.5)
    assert summary["energy_error_max_at"] == 10
    assert bench.committee_force_std is None and "committee_energy_std_max_ev" not in summary


def test_benchmark_reports_committee_spread_and_can_stop():
    bench = benchmark_path(path(4), Spring(committee=True), Spring(k=1.5), points=4)
    assert bench.committee_energy_std == pytest.approx(np.linspace(1.0, 2.0, 4) / 2)
    assert bench.summary()["committee_force_std_max_ev_per_angstrom"] > 0
    calls = iter([False, True])
    stopped = benchmark_path(path(4), Spring(), Spring(k=3.0), should_stop=lambda: next(calls))
    assert stopped.stopped and len(stopped.coordinate) == 1
    with pytest.raises(ValueError):
        calculator = Spring()
        benchmark_path(path(3), calculator, calculator)


def test_write_report_and_descriptor(tmp_path):
    pytest.importorskip("matplotlib")
    frames = path(5)
    for frame in frames:
        frame += Atoms("H", positions=[[0, 1.0, 0]])
    angles = bond_angle([frame.positions for frame in frames], (2, 0, 1))
    assert angles == pytest.approx(90.0)
    two_atom = [frame[:2] for frame in frames]
    bench = benchmark_path(
        two_atom, Spring(k=2.0), Spring(), coordinate=np.linspace(-1, 1, 5),
        coordinate_label="arc", descriptor=("∠H–H–H", angles),
    )
    written = write_report(bench, tmp_path / "report", mark=(0.0, "TS"), end_labels=("A", "B"))
    assert all(path.is_file() and path.stat().st_size > 0 for path in written.values())
    header = written["csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "energy_error_eV" in header and "∠H–H–H" in header


def test_basis_names():
    assert basis_name("def2-tzvp") == "def2-TZVP"
    assert basis_name("cc-pvtz") == "cc-pVTZ"
    assert basis_name("aug-cc-pvdz") == "aug-cc-pVDZ"


def test_cli_benchmark(tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    from ase.io import write

    frames = path(6)
    for k, frame in enumerate(frames):
        frame.info["irc_arc"] = k - 2.0
    trajectory = tmp_path / "irc.extxyz"
    write(trajectory, frames)
    model = tmp_path / "m.model"
    model.write_text("")
    made = []

    def fake(backend, model_path, **kwargs):
        made.append(backend)
        return Spring(k=2.0 if backend == "mace" else 1.0)

    monkeypatch.setattr("samson_mlip_visualizer.calculators.create_calculator", fake)
    code = benchmark.main(
        [str(trajectory), "--model", str(model), "--reference", "mace", "--reference-model",
         str(model), "--points", "6", "-o", str(tmp_path / "out")]
    )
    assert code == 0 and made == ["mace", "mace"]
    out = capsys.readouterr().out
    assert "energy_error_max_ev" in out and (tmp_path / "out_forces.png").is_file()
