"""Shared definitions for the SN2 free-energy example (after the VASP MD tutorial, part 3).

Two systems, both with ξ = d(C–X_leaving) − d(C–X_nucleophile), atoms C H H H X_leaving X_nuc:

- ``F``: F⁻ + CH₃Cl → CH₃F + Cl⁻, ξ = d(C–Cl) − d(C–F), with the AIMNet2 fine-tuned on
  ωB97X-D including the free fragments (``examples/sn2_f_ch3cl``), run in-process in
  the aimnet environment;
- ``Cl``: Cl⁻ + CH₃Cl → ClCH₃ + Cl⁻ (the tutorial's identity reaction), with GFN2-xTB.

Run with the aimnet environment's Python; ``SN2FE_WORK`` overrides the output folder.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # aimnet's torch.compile needs MSVC
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402

from samson_mlip_visualizer.free_energy import DistanceCombination  # noqa: E402

WORK = Path(os.environ.get("SN2FE_WORK", r"D:\MLIP_Work_Folder\sn2_free_energy"))
SN2 = Path(r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
AIMNET_MODEL = (r"D:\MLIP_Work_Folder\cache\aimnet\finetuned"
                r"\aimnet2_wb97m_d3_0_SN2-F-CH3Cl_wB97XD-def2TZVPD_fragments.pt")
TEMPERATURE = 300.0
TIMESTEP_FS = 2.0  # with tritium, as in the tutorial
ANDERSEN = 0.05
COORDINATE = DistanceCombination([(0, 4, 1.0), (0, 5, -1.0)])  # d(C–X4) − d(C–X5)

SYSTEMS = {
    "F": {"label": "F⁻ + CH₃Cl (fine-tuned AIMNet2)", "symbols": "CH3ClF",
          "masses": [12.011, 3.0, 3.0, 3.0, 35.453, 18.998], "xi_range": (-1.5, 2.0),
          "extra_windows": [-0.875, -0.625, -0.125, 0.125, 0.375], "symmetric": False},
    # Symmetric: A(ξ) = A(−ξ), so windows run on ξ ≤ 0 only and are mirrored.
    "Cl": {"label": "Cl⁻ + CH₃Cl (GFN2-xTB)", "symbols": "CH3Cl2",
           "masses": [12.011, 3.0, 3.0, 3.0, 35.453, 35.453], "xi_range": (-1.5, 1.5),
           "extra_windows": [-0.875, -0.375, -0.125], "symmetric": True},
}


def calculator(system: str):
    if system == "F":
        from aimnet.calculators import AIMNet2ASE

        return AIMNet2ASE(AIMNET_MODEL, charge=-1)
    from samson_mlip_visualizer.xtb_backend import XTBCalculator, find_xtb

    return XTBCalculator(find_xtb(), charge=-1, method="gfn2")


def folder(system: str) -> Path:
    path = WORK / system
    path.mkdir(parents=True, exist_ok=True)
    return path


def reactant_complex(system: str) -> Atoms:
    """F: the fine-tuned AIMNet2's own F⁻···CH₃Cl minimum. Cl: Cl⁻···CH₃Cl relaxed with
    GFN2-xTB from a collinear guess (cached in the work folder)."""
    if system == "F":
        evaluation = json.loads((SN2 / "finetune_aimnet2" / "evaluation_fragments.json")
                                .read_text())
        ends = evaluation["ts"]["irc_ends"].values()
        reactant = max(ends, key=lambda e: e["r_CF"])
        return Atoms(SYSTEMS["F"]["symbols"], positions=reactant["positions"])
    cached = folder("Cl") / "reactant_complex.xyz"
    if cached.exists():
        return read(cached)
    from ase.io import write
    from ase.optimize import BFGS

    h = 1.09
    atoms = Atoms(SYSTEMS["Cl"]["symbols"], positions=[
        [0, 0, 0], [h, 0, -0.36], [-h / 2, 0.94, -0.36], [-h / 2, -0.94, -0.36],
        [0, 0, 1.80], [0, 0, -3.20]])
    atoms.calc = calculator("Cl")
    BFGS(atoms, logfile=None).run(fmax=0.01, steps=300)
    write(cached, atoms)
    return atoms


def path_frames(system: str) -> list[Atoms]:
    """Frames spanning the reaction, to start fixed-ξ windows near their target."""
    if system == "F":
        return read(SN2 / "finetune_aimnet2" / "tuned_irc_fragments.extxyz", ":")
    h = 1.07  # a D3h Walden TS guess: planar CH3 between the two chlorines
    ts = Atoms(SYSTEMS["Cl"]["symbols"], positions=[
        [0, 0, 0], [h, 0, 0], [-h / 2, 0.9267, 0], [-h / 2, -0.9267, 0],
        [0, 0, 2.32], [0, 0, -2.32]])
    return [reactant_complex("Cl"), ts]


def start_near(system: str, xi: float) -> Atoms:
    """The path frame closest to ξ, with the rest of the way made chemically: the
    nucleophile (atom 5) pulled out along its C–X axis to lower ξ, or the leaving
    group (atom 4) to raise it. (Projecting onto ξ along the mass-weighted gradient
    instead moves carbon away from its hydrogens and tears CH₃ apart.)"""
    frames = path_frames(system)
    best = min(frames, key=lambda f: abs(COORDINATE.value(f.positions) - xi))
    atoms = Atoms(SYSTEMS[system]["symbols"], positions=best.positions)
    missing = xi - COORDINATE.value(atoms.positions)
    halogen = 4 if missing > 0 else 5
    axis = atoms.positions[halogen] - atoms.positions[0]
    atoms.positions[halogen] += abs(missing) * axis / np.linalg.norm(axis)
    return atoms


def prepare(atoms: Atoms, system: str) -> Atoms:
    atoms.set_masses(SYSTEMS[system]["masses"])
    atoms.calc = calculator(system)
    return atoms


def window_grid(system: str) -> np.ndarray:
    """Every 0.25 Å, plus extra points through the reactant well and the barrier, where
    the mean force changes fastest; only ξ ≤ 0 for a symmetric reaction."""
    low, high = SYSTEMS[system]["xi_range"]
    if SYSTEMS[system]["symmetric"]:
        high = 0.0
    grid = np.arange(low, high + 1e-9, 0.25)
    return np.round(np.sort(np.concatenate([grid, SYSTEMS[system]["extra_windows"]])), 3)
