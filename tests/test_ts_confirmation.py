"""Confirming a transition state: IRC end points, scan jump detection, path playback."""

import numpy as np
import pytest
from test_reaction_jobs_cli import (  # noqa: F401 - pytest fixtures
    bondwell_bridge,
    diatomic_model,
)
from test_reaction_path import bond, diatomic
from test_remote_dispatcher import result

from samson_mlip_visualizer.reaction_path import aligned_rmsd, irc, match_minimum, scan_jumps
from samson_mlip_visualizer.samson_modes import bounce_step

pytest.importorskip("sella")

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def rotated(positions, degrees=40.0):
    angle = np.radians(degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    return positions @ rotation.T + [3.0, -1.0, 2.0]


def test_aligned_rmsd_ignores_rigid_motion():
    assert aligned_rmsd(WATER, rotated(WATER)) == pytest.approx(0.0, abs=1e-9)
    stretched = WATER.copy()
    stretched[1, 0] += 0.3
    assert aligned_rmsd(WATER, stretched) > 0.05


def test_match_minimum_names_the_closest_reference():
    stretched = WATER.copy()
    stretched[1, 0] += 0.5
    references = {"reactant": WATER, "product": stretched}
    assert match_minimum(rotated(WATER), references) == "reactant"
    assert match_minimum(rotated(stretched), references) == "product"
    halfway = WATER.copy()
    halfway[1, 0] += 0.25
    assert match_minimum(halfway, references) is None


def test_scan_jumps_flags_a_snap_between_arrangements():
    # C at the origin, N at z = 1.2: H bends slowly on C's side, then snaps to N's.
    # (A rigid rotation would not count: superposition removes it.)
    frames, distances = [], []
    for k in range(6):
        hydrogen = [0.0, 0.1 * k, -1.1] if k < 4 else [0.0, 0.1 * k, 2.3]
        frames.append(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], hydrogen]))
        distances.append(2.3 - 0.1 * k)
    assert scan_jumps(frames, distances) == [4]
    assert scan_jumps(frames[:4], distances[:4]) == []


def test_irc_keeps_the_relaxed_end_geometries():
    atoms = diatomic(2.0)
    path = irc(atoms, step=0.05, fmax=1e-3)
    ends = sorted([bond(path.forward_minimum_positions), bond(path.reverse_minimum_positions)])
    assert ends == pytest.approx([1.5, 2.5], abs=1e-3)


def test_bounce_step_plays_forward_and_back():
    step, direction, seen = 0, 1, [0]
    for _ in range(8):
        step, direction = bounce_step(step, 4, direction)
        seen.append(step)
    assert seen == [0, 1, 2, 3, 2, 1, 0, 1, 2]
    assert bounce_step(0, 1, 1) == (0, 1)


def test_ts_job_confirms_with_irc(bondwell_bridge):  # noqa: F811
    samson, dispatcher, model = bondwell_bridge(diatomic_model(1.85, "guess"))
    job = result(dispatcher, "job.start", kind="ts", model=model, fmax=1e-4, check_irc=True)
    status = result(dispatcher, "job.status", id=job["id"], log_lines=50)
    assert status["state"] == "finished", status["error"]
    check = status["result"]["irc"]
    for end in ("forward", "reverse"):
        assert check[end]["minimum_minus_ts_ev"] == pytest.approx(-1.0, abs=1e-4)
    assert "connects" not in check  # no reference minima for a single-ended search
    assert any(line.startswith("IRC forward end") for line in status["log"])


def test_qst_job_irc_connects_reactant_and_product(bondwell_bridge):  # noqa: F811
    samson, dispatcher, model = bondwell_bridge(
        diatomic_model(1.5, "reactant"), diatomic_model(2.5, "product")
    )
    job = result(
        dispatcher, "job.start", kind="qst", model=model, images=5, fmax=0.01, check_irc=True
    )
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["state"] == "finished", status["error"]
    check = status["result"]["ts"]["irc"]
    assert check["connects"]
    assert {check["forward"]["matches"], check["reverse"]["matches"]} == {"reactant", "product"}


def test_scan_job_reports_jumps_and_irc_pair_distances(bondwell_bridge):  # noqa: F811
    samson, dispatcher, model = bondwell_bridge(diatomic_model(1.5, "reactant"))
    job = result(
        dispatcher, "job.start", kind="scan", model=model, pair="0-1", stop=2.5, points=11,
        check_irc=True,
    )
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["state"] == "finished", status["error"]
    scan = status["result"]
    assert scan["jumps"] == []
    check = scan["ts"]["irc"]
    distances = sorted(check[end]["pair_distance"] for end in ("forward", "reverse"))
    assert distances == pytest.approx([1.5, 2.5], abs=1e-3)
