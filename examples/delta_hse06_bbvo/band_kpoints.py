"""Explicit k-point lists for band edges and effective masses, and the mass algebra.

VASP takes a list of k-points with weights. The SCF mesh carries the weight; extra points get
weight 0, so they are diagonalized in the converged potential without changing it (the way
hybrid band structures are done). This gives band edges, dispersion and finite-difference masses
in one SCF run, with no CHGCAR hand-off, for any cell.

Where the edges are looked for: every band of a crystal with time reversal has zero slope at the
eight TRIM points (k = G/2), so they are the natural edge candidates even in a P1 supercell. A
dense uniform grid and the lines from Γ to the other seven TRIMs show whether an edge is off them.

Masses: a Cartesian stencil around a TRIM (centre, ±h along x, y, z, ±h along the six face
diagonals) gives the Hessian H = ∂²E/∂k∂k (eV Å²) of one band; m* = ħ²/m_e · H⁻¹ in units of m_e.
A second, larger step along the axes checks parabolicity. Supercell folding does not change the
curvature at an extremum, so no unfolding is needed for the masses.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np

HBAR2_OVER_ME = 7.619964  # ħ²/m_e in eV Å²
# package 19, whose relaxed geometries packages 29 and 30 start from
SOURCE19 = Path(r"D:\MLIP_Work_Folder\hpc_smoke_tests\batch04_2026-09-30\19_vasp_bbvo_polymorphs")
ZVAL = {"Ba": 10, "V": 11, "Nb": 11, "Ta": 11, "Bi": 5, "O": 6,  # MP POTCARs (19's OUTCARs)
        "In": 13, "Sc": 11}  # In_d, Sc_sv (MP POTCAR set; package 31)


def nbands(atoms, soc=False):
    """Occupied bands plus at least max(24, N_atoms) empty ones, a multiple of 8 (x2 with SOC)."""
    nocc = sum(ZVAL[s] for s in atoms.get_chemical_symbols()) / 2
    n = 8 * math.ceil((nocc + max(24, len(atoms))) / 8)
    return 2 * n if soc else n


TRIM_NAMES = {
    (0, 0, 0): "G",
    (1, 0, 0): "X1",
    (0, 1, 0): "X2",
    (0, 0, 1): "X3",
    (0, 1, 1): "M1",
    (1, 0, 1): "M2",
    (1, 1, 0): "M3",
    (1, 1, 1): "R",
}
# step sizes in Å⁻¹ (2π included): ħ²h²/2m ≈ 6 meV at h = 0.04 for m = m_e, against the
# 0.1 meV resolution of vasprun.xml eigenvalues; 0.08 checks parabolicity
STEPS = (0.04, 0.08)


def trims():
    """The eight TRIMs, as {name: fractional k}."""
    return {name: np.array(key) / 2 for key, name in TRIM_NAMES.items()}


def full_mesh(mesh):
    """All points of a Γ-centred mesh (fractional), each of weight 1."""
    axes = [np.arange(n) / n for n in mesh]
    pts = np.array(list(itertools.product(*axes)))
    return np.where(pts > 0.5 + 1e-9, pts - 1, pts)


def uniform(n):
    return full_mesh((n, n, n))


def path(points_per_line=8):
    """Straight lines Γ → each other TRIM, end points included once (fractional, with labels)."""
    out = []
    for name, k in trims().items():
        if name == "G":
            continue
        for i in range(1, points_per_line + 1):
            out.append((k * i / points_per_line, f"path G-{name} {i}/{points_per_line}"))
    return out


def to_fractional(cell, k_cart):
    """Cartesian k (Å⁻¹, 2π included) -> fractional coordinates of the reciprocal basis."""
    return np.asarray(k_cart) @ np.asarray(cell).T / (2 * np.pi)


def to_cartesian(cell, k_frac):
    return 2 * np.pi * np.asarray(k_frac) @ np.linalg.inv(np.asarray(cell)).T


def directions():
    """The 3 axes and 6 face diagonals (unit vectors), with labels."""
    e = np.eye(3)
    dirs = [(e[i], "xyz"[i]) for i in range(3)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        dirs.append(((e[i] + e[j]) / np.sqrt(2), f"{'xyz'[i]}+{'xyz'[j]}"))
        dirs.append(((e[i] - e[j]) / np.sqrt(2), f"{'xyz'[i]}-{'xyz'[j]}"))
    return dirs


def stencil(cell, center_frac, label, steps=STEPS):
    """Points (fractional, label) around ``center_frac``: the centre, ±steps[0] along the 3 axes
    and 6 diagonals, ±steps[1] along the axes."""
    out = [(np.asarray(center_frac, float), f"{label} c")]
    for n, h in enumerate(steps):
        for d, name in directions() if n == 0 else directions()[:3]:
            for sign in (1, -1):
                k = center_frac + to_fractional(cell, sign * h * d)
                out.append((k, f"{label} {'+' if sign > 0 else '-'}{name} h{n}"))
    return out


def write_explicit(path_, mesh, extra, title):
    """KPOINTS with the full SCF mesh (weight 1) and ``extra`` (fractional, label) at weight 0.
    Returns the labels in file order (mesh points labelled 'mesh')."""
    base = full_mesh(mesh)
    lines = [
        f"{title}: SCF mesh {mesh[0]}x{mesh[1]}x{mesh[2]} (weight 1) + {len(extra)} "
        "zero-weight points",
        str(len(base) + len(extra)),
        "Reciprocal",
    ]
    lines += [f"{k[0]:14.10f} {k[1]:14.10f} {k[2]:14.10f} 1" for k in base]
    lines += [f"{k[0]:14.10f} {k[1]:14.10f} {k[2]:14.10f} 0" for k, _ in extra]
    Path(path_).write_text("\n".join(lines) + "\n", newline="\n")
    return ["mesh"] * len(base) + [label for _, label in extra]


def write_labels(folder, level, labels):
    Path(folder, f"KPOINTS.{level}.labels.json").write_text(json.dumps(labels), encoding="utf-8")


# --- analysis -----------------------------------------------------------------------------------


def hessian(energy, label, steps=STEPS):
    """The Hessian (eV Å²) of one band at a stencil centre ``label`` from ``energy``:
    {point label: eV}. Also returns the axis curvatures at the larger step."""
    e0 = energy[f"{label} c"]
    dirs = directions()

    def curv(name, n, h):
        return (energy[f"{label} +{name} h{n}"] + energy[f"{label} -{name} h{n}"] - 2 * e0) / h**2

    h0 = steps[0]
    H = np.zeros((3, 3))
    for i in range(3):
        H[i, i] = curv("xyz"[i], 0, h0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        plus, minus = curv(f"{'xyz'[i]}+{'xyz'[j]}", 0, h0), curv(f"{'xyz'[i]}-{'xyz'[j]}", 0, h0)
        H[i, j] = H[j, i] = (plus - minus) / 2  # d·H·d for (e_i ± e_j)/√2 is (Hii+Hjj)/2 ± Hij
    big = np.array([curv("xyz"[i], 1, steps[1]) for i in range(3)])
    assert len(dirs) == 9
    return H, big


def masses(H, sign=1):
    """Mass tensor (m_e) from a Hessian; ``sign`` = -1 for holes (a band maximum).
    Returns principal masses, conductivity mass 3/tr(M⁻¹) and DOS mass det(M)^(1/3),
    or None entries when the band is not an extremum of that sign there."""
    Hs = sign * np.asarray(H)
    w = np.linalg.eigvalsh(Hs)
    principal = sorted(HBAR2_OVER_ME / w) if np.all(w > 0) else None
    out = {
        "hessian_eV_A2": np.round(Hs, 4).tolist(),
        "curvature_eigs_eV_A2": np.round(w, 4).tolist(),
        "principal_me": None,
        "conductivity_me": None,
        "dos_me": None,
    }
    if principal:
        out["principal_me"] = [round(float(m), 4) for m in principal]
        out["conductivity_me"] = round(3 / sum(1 / m for m in principal), 4)
        out["dos_me"] = round(float(np.prod(principal)) ** (1 / 3), 4)
    return out
