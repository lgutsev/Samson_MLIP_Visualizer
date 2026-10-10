"""Characterization tests for the geometry helpers in reaction_path and the player step logic."""

import numpy as np
import pytest
from ase import Atoms

from samson_mlip_visualizer.reaction_path import (
    _check_endpoints,
    aligned_rmsd,
    match_minimum,
    scan_jumps,
)
from samson_mlip_visualizer.samson_modes import bounce_step

TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.5], [0.3, 0.3, 1.0]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_aligned_rmsd_ignores_translation_and_rotation():
    moved = TRIANGLE @ rotation_z(0.7).T + np.array([5.0, -3.0, 2.0])
    assert aligned_rmsd(TRIANGLE, moved) == pytest.approx(0.0, abs=1e-9)
    assert aligned_rmsd(TRIANGLE, TRIANGLE) == pytest.approx(0.0, abs=1e-12)


def test_aligned_rmsd_detects_a_real_displacement_and_is_symmetric():
    bent = TRIANGLE.copy()
    bent[3] += [0.4, 0.0, 0.0]
    forward = aligned_rmsd(TRIANGLE, bent)
    assert forward > 0.05
    assert forward == pytest.approx(aligned_rmsd(bent, TRIANGLE))


def test_aligned_rmsd_treats_a_mirror_image_as_different():
    # The Kabsch fit refuses reflections (determinant correction): inversion is not zero.
    mirrored = TRIANGLE * np.array([1.0, 1.0, -1.0])
    assert aligned_rmsd(TRIANGLE, mirrored) > 0.1


def test_match_minimum_picks_nearest_within_tolerance():
    near = TRIANGLE + np.array([0.0, 0.0, 0.0])
    far = TRIANGLE * 1.5
    refs = {"far": far, "near": near}
    assert match_minimum(TRIANGLE @ rotation_z(1.0).T, refs) == "near"
    assert match_minimum(TRIANGLE * 1.2, refs, tolerance=0.01) is None
    assert match_minimum(TRIANGLE, {}) is None


def test_scan_jumps_flags_only_branch_switches():
    first_atom = np.array([[1], [0], [0], [0]])
    smooth = [TRIANGLE + np.array([0.1 * k, 0.0, 0.0]) * first_atom for k in range(4)]
    distances = [1.0, 1.1, 1.2, 1.3]
    assert scan_jumps(smooth, distances) == []
    jumped = list(smooth)
    jumped[2] = jumped[2].copy()
    jumped[2][3] += [0.0, 2.0, 0.0]  # one atom snaps by 2 A while the scan step is 0.1 A
    assert scan_jumps(jumped, distances) == [2, 3]  # the step into the jump and the step out of it
    assert scan_jumps([], []) == []
    assert scan_jumps(smooth[:1], distances[:1]) == []


def test_scan_jumps_threshold_scales_with_step_size():
    a = TRIANGLE.copy()
    b = TRIANGLE.copy()
    b[3] += [0.0, 0.5, 0.0]
    assert scan_jumps([a, b], [1.0, 1.1]) == [1]  # 0.5 A > max(0.3, 0.3)
    assert scan_jumps([a, b], [1.0, 2.0]) == []  # a 1 A scan step allows up to 3 A


def test_check_endpoints_accepts_matching_structures():
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    b = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2]])
    _check_endpoints([a, b])


def test_check_endpoints_rejects_symbol_order_cell_and_pbc_changes():
    a = Atoms("HF", positions=[[0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="same atoms in the same order"):
        _check_endpoints([a, Atoms("FH", positions=[[0, 0, 0], [0, 0, 1]])])
    with pytest.raises(ValueError, match="one cell and PBC"):
        _check_endpoints([a, Atoms("HF", positions=[[0, 0, 0], [0, 0, 1]], pbc=True)])
    with pytest.raises(ValueError, match="one cell and PBC"):
        _check_endpoints(
            [Atoms("HF", cell=[5, 5, 5], pbc=True), Atoms("HF", cell=[6, 5, 5], pbc=True)])


@pytest.mark.parametrize(
    ("current", "steps", "direction", "expected"),
    [
        (0, 5, 1, (1, 1)),
        (3, 5, 1, (4, 1)),
        (4, 5, 1, (3, -1)),  # bounce at the end
        (0, 5, -1, (1, 1)),  # bounce at the start
        (2, 5, -1, (1, -1)),
        (0, 1, 1, (0, 1)),  # a single frame never moves
        (0, 0, -1, (0, -1)),
    ],
)
def test_bounce_step(current, steps, direction, expected):
    assert bounce_step(current, steps, direction) == expected


def test_bounce_step_ping_pongs_over_all_frames():
    current, direction, seen = 0, 1, []
    for _ in range(8):
        current, direction = bounce_step(current, 4, direction)
        seen.append(current)
    assert seen == [1, 2, 3, 2, 1, 0, 1, 2]
