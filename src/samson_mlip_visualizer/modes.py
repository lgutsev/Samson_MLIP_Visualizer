"""Geometry for displaying a vibrational mode: animation frames and arrows.

Pure NumPy, so it is testable without SAMSON; ``samson_modes`` turns the results
into SAMSON path and mesh nodes. Positions and lengths are in Å.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mode_frames(
    positions: np.ndarray,
    mode: np.ndarray,
    *,
    max_displacement: float = 0.3,
    frames: int = 24,
) -> list[np.ndarray]:
    """One period of oscillation along ``mode``, starting at ``positions``.

    Frame ``k`` is ``positions + A sin(2 pi k / frames) mode``, with ``A`` chosen
    so the most-displaced atom moves ``max_displacement`` Å at the turning points.
    """
    positions = np.asarray(positions, dtype=float)
    mode = np.asarray(mode, dtype=float)
    if mode.shape != positions.shape:
        raise ValueError(f"mode shape {mode.shape} does not match positions {positions.shape}")
    if frames < 2:
        raise ValueError("An animation needs at least two frames")
    largest = np.linalg.norm(mode, axis=1).max(initial=0.0)
    if largest == 0:
        raise ValueError("The mode does not move any atom")
    amplitude = max_displacement / largest
    return [
        positions + amplitude * np.sin(2 * np.pi * k / frames) * mode for k in range(frames)
    ]


@dataclass(frozen=True)
class TriangleMesh:
    positions: np.ndarray  # (V, 3) Å
    normals: np.ndarray  # (V, 3) unit
    triangles: np.ndarray  # (T, 3) vertex indices
    colors: np.ndarray  # (V, 4) RGBA in [0, 1]


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def arrow_mesh(
    starts: np.ndarray,
    vectors: np.ndarray,
    *,
    shaft_radius: float = 0.04,
    head_radius: float = 0.10,
    head_length: float = 0.22,
    segments: int = 16,
    min_length: float = 0.05,
    color: tuple[float, float, float, float] = (1.0, 0.55, 0.0, 1.0),
) -> TriangleMesh:
    """Solid arrows from each start along its vector; shorter ones are skipped.

    Each arrow is a capped cylinder shaft and a capped cone head, with outward
    normals and counter-clockwise triangles seen from outside.
    """
    starts = np.asarray(starts, dtype=float).reshape(-1, 3)
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 3)
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    angles = 2 * np.pi * np.arange(segments) / segments

    def add(points, point_normals):
        first = sum(len(block) for block in positions)
        positions.append(np.asarray(points, dtype=float))
        normals.append(np.asarray(point_normals, dtype=float))
        return first

    for start, vector in zip(starts, vectors, strict=True):
        length = float(np.linalg.norm(vector))
        if length < min_length:
            continue
        axis = vector / length
        first, second = _perpendicular_basis(axis)
        ring = np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
        head = min(head_length, 0.6 * length)
        neck = start + (length - head) * axis
        tip = start + length * axis
        nxt = (np.arange(segments) + 1) % segments

        # Shaft side.
        bottom = add(start + shaft_radius * ring, ring)
        top = add(neck + shaft_radius * ring, ring)
        for j, k in enumerate(nxt):
            triangles += [(bottom + j, bottom + k, top + k), (bottom + j, top + k, top + j)]
        # Shaft bottom cap.
        centre = add([start], [-axis])
        cap = add(start + shaft_radius * ring, np.repeat([-axis], segments, axis=0))
        triangles += [(centre, cap + k, cap + j) for j, k in enumerate(nxt)]
        # Cone side: one tip vertex per segment so each facet gets its own normal.
        slope = head * ring + head_radius * axis
        slope /= np.linalg.norm(slope, axis=1)[:, None]
        base = add(neck + head_radius * ring, slope)
        middle = 0.5 * (slope + slope[nxt])
        middle /= np.linalg.norm(middle, axis=1)[:, None]
        tips = add(np.repeat([tip], segments, axis=0), middle)
        triangles += [(base + j, base + k, tips + j) for j, k in enumerate(nxt)]
        # Cone base cap.
        centre = add([neck], [-axis])
        cap = add(neck + head_radius * ring, np.repeat([-axis], segments, axis=0))
        triangles += [(centre, cap + k, cap + j) for j, k in enumerate(nxt)]

    if not positions:
        empty = np.zeros((0, 3))
        return TriangleMesh(empty, empty, np.zeros((0, 3), dtype=int), np.zeros((0, 4)))
    vertices = np.concatenate(positions)
    return TriangleMesh(
        positions=vertices,
        normals=np.concatenate(normals),
        triangles=np.asarray(triangles, dtype=int),
        colors=np.tile(np.asarray(color, dtype=float), (len(vertices), 1)),
    )


def mode_arrows(
    positions: np.ndarray, mode: np.ndarray, *, max_length: float = 1.0, **style
) -> TriangleMesh:
    """Arrows on every atom along ``mode``; the longest is ``max_length`` Å."""
    mode = np.asarray(mode, dtype=float)
    largest = np.linalg.norm(mode, axis=1).max(initial=0.0)
    if largest == 0:
        raise ValueError("The mode does not move any atom")
    return arrow_mesh(positions, mode * (max_length / largest), **style)
