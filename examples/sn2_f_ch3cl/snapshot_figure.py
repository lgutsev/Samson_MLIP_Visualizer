"""Three-panel figure of SAMSON snapshots: reactant complex, TS, product complex.

    python snapshot_figure.py [work folder] [output png]

Expects ``snap_<name>.png`` (1600x1200 SAMSON viewport captures) and
``snap_<name>.xyz`` (the geometry shown, atoms in the order C H H H Cl F) for
reactant_complex, ts and product_complex, as written by the capture step in the
README, plus ``aimnet2_irc.extxyz`` for the energies.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else WORK / "sn2_irc_snapshots.png"
CROP = (slice(390, 830), slice(380, 1300))  # rows, columns of the 1600x1200 capture
PANELS = (
    ("reactant_complex", "F⁻···CH₃Cl"),
    ("ts", "Walden transition state"),
    ("product_complex", "FCH₃···Cl⁻"),
)
INK, INK2, MUTED = "#1f1f1f", "#4a4a4a", "#7a7a7a"


def main():
    irc = read(WORK / "aimnet2_irc.extxyz", index=":")
    energies = np.array([frame.info["energy_ev"] for frame in irc])
    relative = {"reactant_complex": 0.0, "ts": energies.max() - energies[0],
                "product_complex": energies[-1] - energies[0]}
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), facecolor="#fcfcfb")
    for ax, (name, title) in zip(axes, PANELS, strict=True):
        image = plt.imread(WORK / f"snap_{name}.png")[CROP]
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d0d0d0")
        ax.set_title(title, loc="left", color=INK, fontsize=12)
        x = read(WORK / f"snap_{name}.xyz").positions
        cf, ccl = np.linalg.norm(x[5] - x[0]), np.linalg.norm(x[4] - x[0])
        energy = f"{relative[name]:+.2f} eV".replace("-", "−")
        energy = "0.00 eV" if name == "reactant_complex" else energy
        if name == "ts":
            energy += ", one imaginary mode (−743 cm⁻¹)"
        ax.text(0, -0.08, f"r(C–F) {cf:.2f} Å, r(C–Cl) {ccl:.2f} Å", transform=ax.transAxes,
                color=INK2, fontsize=10, va="top")
        ax.text(0, -0.19, energy, transform=ax.transAxes, color=MUTED, fontsize=10, va="top")
    figure.suptitle("F⁻ + CH₃Cl → CH₃F + Cl⁻ in SAMSON: frames of the AIMNet2 IRC "
                    "(charge −1; energies relative to the reactant complex)",
                    x=0.01, ha="left", color=INK, fontsize=12)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.2, wspace=0.05)
    figure.savefig(OUT, dpi=110, facecolor=figure.get_facecolor())
    print(OUT)


if __name__ == "__main__":
    main()
