import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.cluster import Icosahedron
from ase.constraints import FixAtoms
from ase.io import read

from samson_mlip_visualizer.md import (
    DistanceConstraint,
    TemperatureLimitError,
    md_warnings,
    parse_pairs,
    run_md,
)


def argon_cluster():
    atoms = Icosahedron("Ar", noshells=2, latticeconstant=5.26)
    atoms.calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=10.0, smooth=True)
    return atoms


def test_parse_pairs_reads_indices_and_targets():
    assert parse_pairs("0-1, 4-7:1.5;") == [
        DistanceConstraint(0, 1, None),
        DistanceConstraint(4, 7, 1.5),
    ]
    with pytest.raises(ValueError, match="0-based"):
        parse_pairs("a-b")


def test_nve_conserves_energy_and_reports_frames():
    atoms = argon_cluster()
    frames = []
    result = run_md(
        atoms,
        ensemble="NVE",
        temperature_k=20,
        timestep_fs=5.0,
        steps=200,
        seed=1,
        report_interval=50,
        on_progress=frames.append,
    )
    assert result.steps == 200
    assert not result.stopped
    assert [frame.step for frame in frames] == [0, 50, 100, 150, 200]
    assert frames[0].temperature_k == pytest.approx(20.0)
    assert frames[-1].time_fs == pytest.approx(1000.0)
    totals = [frame.total_ev for frame in frames]
    assert max(totals) - min(totals) < 1e-3
    assert abs(result.energy_drift_mev_per_atom_ps) < 1.0


def test_langevin_holds_and_sets_constrained_distance():
    atoms = argon_cluster()
    result = run_md(
        atoms,
        ensemble="Langevin",
        temperature_k=40,
        timestep_fs=5.0,
        steps=100,
        seed=2,
        distance_constraints=[DistanceConstraint(0, 1, target=4.2)],
    )
    assert atoms.get_distance(0, 1) == pytest.approx(4.2, abs=1e-6)
    (summary,) = result.constraint_forces
    assert (summary.i, summary.j, summary.samples) == (0, 1, 101)
    assert summary.distance == pytest.approx(4.2, abs=1e-6)
    # 4.2 Å is past the LJ minimum (~3.8 Å), so the pair is pulled together on average.
    assert summary.mean_force_ev_per_angstrom < 0


def test_fixed_atoms_stay_put():
    atoms = argon_cluster()
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    start = atoms.positions[:2].copy()
    run_md(atoms, ensemble="Langevin", temperature_k=40, timestep_fs=5.0, steps=50, seed=3)
    assert atoms.positions[:2] == pytest.approx(start)


def test_constraints_rejected_for_nose_hoover_chain():
    with pytest.raises(ValueError, match="Distance constraints"):
        run_md(
            argon_cluster(),
            ensemble="NoseHooverChain",
            steps=5,
            distance_constraints=[DistanceConstraint(0, 1)],
        )


@pytest.mark.parametrize("ensemble", ["Bussi", "NoseHooverChain"])
def test_thermostats_run(ensemble):
    result = run_md(
        argon_cluster(), ensemble=ensemble, temperature_k=30, timestep_fs=5.0, steps=40, seed=4
    )
    assert result.steps == 40
    assert result.mean_temperature_k > 0


def test_md_can_be_stopped():
    polls = 0

    def stop():
        nonlocal polls
        polls += 1
        return polls >= 3

    result = run_md(argon_cluster(), steps=100, timestep_fs=5.0, seed=5, should_stop=stop)
    assert result.stopped
    assert result.steps == 3


def test_temperature_ceiling_aborts():
    with pytest.raises(TemperatureLimitError, match="ceiling"):
        run_md(
            argon_cluster(),
            ensemble="NVE",
            temperature_k=500,
            steps=10,
            timestep_fs=5.0,
            seed=6,
            max_temperature_k=100,
        )


def test_writes_trajectory(tmp_path):
    path = tmp_path / "md.extxyz"
    path.write_text("stale")
    run_md(
        argon_cluster(),
        steps=20,
        timestep_fs=5.0,
        seed=7,
        trajectory=path,
        trajectory_interval=5,
    )
    frames = read(path, index=":")
    assert len(frames) == 5
    assert np.isfinite(frames[-1].get_potential_energy())


def test_warns_about_large_timestep_with_hydrogen():
    water = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    assert md_warnings(water, timestep_fs=2.0, ensemble="Langevin")
    assert md_warnings(water, timestep_fs=0.5, ensemble="Langevin") == []
    assert any(
        "float32" in message
        for message in md_warnings(water, timestep_fs=0.5, ensemble="NVE", dtype="float32")
    )
