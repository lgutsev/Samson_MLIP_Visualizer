"""Step 1d (laptop): ShakeNBreak distortions around one Nb or Ta, screened with MACE-MP-0.
-> snb_screen.json, snb_frames.extxyz, snb/ (VASP single points), images/bbvo_snb.png

Every relaxation of a substituted cell that starts from the symmetric host stays
near it: the forces on the dopant's neighbours are zero or small by symmetry. The
ShakeNBreak recipe (Mosquera-Lois, Kavanagh, Walsh & Scanlon, npj Comput. Mater. 9,
25 (2023)) is to break that symmetry on purpose: stretch or compress the bonds to
a few neighbours of the defect by -60 to +60 %, rattle the rest, relax each
start, and keep the lowest. Here the relaxations are MACE-MP-0, so a few dozen
starts take minutes, and only the lowest ones go to DFT.

The pieces:

- **doped** (Kavanagh et al., JOSS 9, 6433 (2024)) chooses the supercell (the
  smallest near-cubic one with at least 10 Å between periodic images of the
  dopant), identifies the V site, and guesses the charge states of Nb_V and Ta_V.
  Nb⁵⁺ and Ta⁵⁺ replace V⁵⁺, so the neutral state is the one that matters. It is
  also the only one MACE-MP-0 can describe, because the model has no charge.
- **ShakeNBreak** applies the distortions. Its own rule distorts as many
  neighbours as the defect has extra or missing electrons. For an isovalent
  substitution that is zero, so it would only rattle. This script runs that
  default (``Rattled``), and also bond distortions of the 2 and the 6 nearest O
  (``--neighbours``), because the question here is whether the octahedron
  around the dopant tilts, breathes, or goes off-center.
- **A control on the pristine host.** MACE-MP-0 finds the cubic host itself
  unstable (see ``stability_check.py``): any rattle lets the whole cell relax
  about 145 meV per formula unit lower. A distorted doped cell therefore gains
  energy from the host alone. So every distortion (same neighbours, same rattle
  seed, same atom order) is applied to a V site of the undoped supercell too,
  and the dopant's own effect is the difference:

      ΔE(M, d)  = E(M cell from distortion d) − E(M cell, unperturbed)
      ΔE(V, d)  = the same, for the undoped cell distorted around a V site
      ΔΔE(M, d) = ΔE(M, d) − ΔE(V, d)                        (plotted per start)
      lowest − lowest = min_d ΔE(M, d) − min_d ΔE(V, d)      (the summary)

  ΔE < 0 alone says a lower structure exists. The summary < 0 says the dopant
  itself stabilizes a distortion beyond what the host already does; > 0, that
  it holds its neighbourhood closer to cubic. The per-start ΔΔE is noisier: the
  same start can fall into different basins with and without the dopant. With
  ``--global-rattle`` every atom is rattled (ShakeNBreak's default). The default
  here tails the rattle off away from the site, which keeps the host's own
  collapse smaller.

A relaxation that does not converge, or ends with two atoms closer than 1.5 Å,
is rejected (kept in the JSON, left out of the minima, the plot, and DFT). Strong
compressions can drive MACE-MP-0 into a hole with no short-range repulsion: in
the first test, Ta with its 6 O compressed by 40 % (1.2 Å) went to Ta–O 0.8 Å
and −10⁷ eV.

Fixed cell at the MACE-MP-0 host lattice, as in ``dilute.py`` (the dilute limit
keeps the host's lattice).

For DFT (``snb/``): PBE+U single points on, for each dopant and the control,
the unperturbed relaxed cell and the lowest distinct distorted minima
(``--dft-keep``). If PBE+U reproduces the ordering, relax those in VASP; if not,
MACE-MP-0's distortion is an artifact.

Needs an environment with doped and ShakeNBreak as well as mace-torch (they are
not dependencies of samson-mlip-visualizer):

    micromamba create -n defects -c conda-forge python=3.12 pip
    micromamba run -n defects python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    micromamba run -n defects python -m pip install mace-torch doped shakenbreak
    PYTHONPATH=../../src micromamba run -n defects python snb_screen.py [--quick]
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE
from common import HERE, KSPACING, WORK, grouped, mace_mp0, primitive

DOPANTS = ("Nb", "Ta")
COLORS = {"Nb": "#2a78d6", "Ta": "#eb6834", "V": "#8a8985"}
# distinct minima: energies within this of each other and geometries this close are one
SAME_E_MEV, SAME_GEOM_A = 5.0, 0.05
# a relaxed structure with any two atoms closer than this has fallen into a hole of the
# model (MACE-MP-0 has no repulsion at very short range; a Ta-O compressed 40 % to 1.2 Å
# collapses to 0.8 Å and -10^7 eV). Shortest real bonds here: V-O ~1.65 Å.
MIN_DIST_A = 1.5


def arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="6 neighbours only, distortions ±20 and ±40 %% (a pipeline check)")
    parser.add_argument("--neighbours", type=int, nargs="+", default=[2, 6],
                        help="numbers of nearest O to distort (default: 2 6)")
    parser.add_argument("--distortions", type=float, nargs="+",
                        default=[-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
                        help="bond distortions as fractions (default: ±0.1 to ±0.4)")
    parser.add_argument("--global-rattle", action="store_true",
                        help="rattle every atom equally (ShakeNBreak's default) instead of "
                             "tailing off away from the site")
    parser.add_argument("--fmax", type=float, default=0.01, help="eV/Å (default 0.01)")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--min-image", type=float, default=10.0,
                        help="doped's minimum distance between periodic images, Å")
    parser.add_argument("--dft-keep", type=int, default=2,
                        help="lowest distinct distorted minima per system for DFT (default 2)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if present)")
    args = parser.parse_args()
    if args.quick:
        args.neighbours, args.distortions = [6], [-0.4, -0.2, 0.2, 0.4]
    return args


def doped_setup(host_prim, min_image):
    """doped's supercell for Nb_V and Ta_V in the host, the index of a V site in
    its bulk supercell, and doped's charge-state guesses per dopant."""
    from doped.generation import DefectsGenerator
    from pymatgen.io.ase import AseAtomsAdaptor

    generator = DefectsGenerator(
        AseAtomsAdaptor.get_structure(host_prim), extrinsic={"V": list(DOPANTS)},
        interstitial_gen_kwargs=False,
        supercell_gen_kwargs={"min_image_distance": min_image}, processes=1)
    charges, sites = {}, set()
    for name, entry in generator.items():  # names like "Nb_V_0", "Nb_V_+1"
        species, host_site = name.split("_")[:2]
        if species in DOPANTS and host_site == "V":
            charges.setdefault(species, []).append(int(entry.charge_state))
            sites.add(entry.defect.name)
    # Fm-3m has one V Wyckoff site, so any V will do; doped confirms it
    if len(sites) != len(DOPANTS):
        raise SystemExit(f"doped finds inequivalent V sites ({sorted(sites)}); "
                         "this script assumes one")
    bulk = generator.bulk_supercell
    index = next(i for i, s in enumerate(bulk) if s.specie.symbol == "V")
    return bulk, index, {k: sorted(v) for k, v in charges.items()}, generator


def starts(bulk, index, element, args):
    """(label, pymatgen Structure) pairs: unperturbed, Rattled, and each bond
    distortion of the nearest O, around ``index`` with ``element`` on it. The
    same atom order and seeds for every element, so the control sees the same
    displacements."""
    from shakenbreak.distortions import distort_and_rattle

    base = bulk.copy()
    base.replace(index, element)
    out = [("Unperturbed", base)]
    todo = [("Rattled", 1.0, 0)] + [(f"{nn}O {d:+.0%}", 1 + d, nn)
                                    for nn in args.neighbours for d in args.distortions]
    for label, factor, nn in todo:
        result = distort_and_rattle(base, factor, num_nearest_neighbours=nn, site_index=index,
                                    local_rattle=not args.global_rattle, distorted_element="O",
                                    seed=int(round(factor * 100)))
        out.append((label, result["distorted_structure"]))
    return out


def m_o_bonds(atoms, index):
    oxygen = [a.index for a in atoms if a.symbol == "O"]
    return np.sort(atoms.get_distances(index, oxygen, mic=True))[:6]


def relax(structure, calc, index, args):
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calc
    opt = FIRE(atoms, logfile=None)
    converged = opt.run(fmax=args.fmax, steps=args.steps)
    energy = float(atoms.get_potential_energy())
    atoms.calc = None
    return atoms, energy, bool(converged), opt.nsteps


def distinct_minima(rows, keep):
    """The lowest ``keep`` distorted minima that differ from the unperturbed one and
    from each other in energy or geometry."""
    ref = rows[0]
    chosen = []
    for row in sorted((r for r in rows[1:] if r["valid"]), key=lambda r: r["e"]):
        same = False
        for other in [ref, *chosen]:
            shift = row["atoms"].positions - other["atoms"].positions
            shift = np.linalg.norm(shift - np.round(
                shift @ np.linalg.inv(row["atoms"].cell)) @ row["atoms"].cell, axis=1).max()
            if abs(row["e"] - other["e"]) * 1000 < SAME_E_MEV and shift < SAME_GEOM_A:
                same = True
                break
        if not same:
            chosen.append(row)
        if len(chosen) == keep:
            break
    return chosen


def main():
    args = arguments()
    t0 = time.perf_counter()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    calc = mace_mp0(device)
    host_prim = primitive()
    host_prim.calc = calc
    BFGS(FrechetCellFilter(host_prim), logfile=None).run(fmax=1e-3, steps=500)
    host_prim.calc = None

    bulk, index, charges, generator = doped_setup(host_prim, args.min_image)
    n_atoms = len(bulk)
    lengths = bulk.lattice.abc
    print(f"doped supercell: {n_atoms} atoms, {lengths[0]:.2f} × {lengths[1]:.2f} × "
          f"{lengths[2]:.2f} Å, V site {index}; charge states guessed: {charges} "
          f"(only q = 0 is run: MACE-MP-0 has no charge) ({time.perf_counter() - t0:.0f} s)",
          flush=True)

    results = {"model": "MACE-MP-0 small", "device": device,
               "supercell_atoms": n_atoms, "supercell_abc_A": [float(x) for x in lengths],
               "supercell_matrix": np.asarray(generator.supercell_matrix).tolist(),
               "host_lattice_A": float(host_prim.cell.lengths()[0] * np.sqrt(2)),
               "doped_charge_states": charges, "site_index": index,
               "rattle": "global" if args.global_rattle else "local",
               "fmax_eV_A": args.fmax, "systems": {}}
    runs = {}
    for element in ("V", *DOPANTS):
        rows = []
        for label, structure in starts(bulk, index, element, args):
            atoms, energy, converged, steps = relax(structure, calc, index, args)
            rows.append({"label": label, "atoms": atoms, "e": energy,
                         "converged": converged, "steps": steps})
        e0 = rows[0]["e"]
        start = rows[0]["atoms"].positions
        for row in rows:
            shift = row["atoms"].positions - start
            shift -= np.round(shift @ np.linalg.inv(row["atoms"].cell)) @ row["atoms"].cell
            row["dE_meV"] = 1000 * (row["e"] - e0)
            row["max_shift_A"] = float(np.linalg.norm(shift, axis=1).max())
            row["M_O_A"] = m_o_bonds(row["atoms"], index).round(4).tolist()
            d = row["atoms"].get_all_distances(mic=True)
            row["min_distance_A"] = float(d[np.triu_indices(len(d), 1)].min())
            row["valid"] = row["converged"] and row["min_distance_A"] >= MIN_DIST_A
        if not rows[0]["valid"]:
            raise SystemExit(f"{element}: the unperturbed relaxation is not valid "
                             f"(converged {rows[0]['converged']}, shortest distance "
                             f"{rows[0]['min_distance_A']:.2f} Å)")
        runs[element] = rows
        best = min((r for r in rows if r["valid"]), key=lambda r: r["e"])
        rejected = [r["label"] for r in rows if not r["valid"]]
        print(f"{element}: {len(rows)} starts; lowest {best['label']} "
              f"{best['dE_meV']:+.1f} meV/cell, M–O {min(best['M_O_A']):.3f}–"
              f"{max(best['M_O_A']):.3f} Å; rejected (unconverged or atoms < {MIN_DIST_A} Å "
              f"apart): {rejected or 'none'} ({time.perf_counter() - t0:.0f} s)", flush=True)

    control = {row["label"]: row["dE_meV"] for row in runs["V"]}
    for element, rows in runs.items():
        results["systems"][element] = [
            {k: row[k] for k in ("label", "dE_meV", "max_shift_A", "M_O_A", "min_distance_A",
                                 "converged", "valid", "steps")}
            | ({"ddE_meV": row["dE_meV"] - control[row["label"]]} if element != "V" else {})
            for row in rows]
    # the dopant's own part: its lowest minimum minus the control's (per-start ΔΔE mixes
    # basins, because the same start can relax into different minima with and without M)
    lowest = {element: min((r for r in rows if r["valid"]), key=lambda r: r["dE_meV"])
              for element, rows in runs.items()}
    results["min_distance_A"] = MIN_DIST_A
    for dopant in DOPANTS:
        best = lowest[dopant]
        results.setdefault("summary", {})[dopant] = {
            "lowest_dE_meV": best["dE_meV"], "lowest_start": best["label"],
            "lowest_dE_control_meV": lowest["V"]["dE_meV"],
            "lowest_control_start": lowest["V"]["label"],
            "lowest_vs_lowest_meV": best["dE_meV"] - lowest["V"]["dE_meV"]}
        print(f"{dopant}: lowest ΔE {best['dE_meV']:+.1f} meV/cell (from {best['label']}), "
              f"V control {lowest['V']['dE_meV']:+.1f}; the dopant's own part, lowest minus "
              f"lowest: {best['dE_meV'] - lowest['V']['dE_meV']:+.1f} meV/cell")

    # frames for DFT: unperturbed + distinct low minima, per system
    from ase.io import write

    from samson_mlip_visualizer.finetune import Selection, write_selection
    from samson_mlip_visualizer.vasp_labeling import VaspSlurmSettings, write_vasp_package

    frames = []
    for element, rows in runs.items():
        for row in [rows[0], *distinct_minima(rows, args.dft_keep)]:
            atoms = grouped(row["atoms"])
            atoms.info = {"group": f"snb_{element.lower()}", "start": row["label"],
                          "mace_mp0_dE_meV": row["dE_meV"], "element_on_site": element}
            frames.append(atoms)
    write(WORK / "snb_frames.extxyz", frames)
    (HERE / "snb_screen.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (WORK / "snb_screen.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    target = WORK / "snb"
    if target.exists():
        print(f"{target} exists; not rewriting the VASP package (remove it to write it again)")
    else:
        selection = Selection()
        for i in range(len(frames)):
            selection.add(i, "snb-check", None, float("inf"))
        frames_path, manifest_path = write_selection(
            WORK / "_snb_frames", frames, selection,
            source="ShakeNBreak distortions around Nb/Ta/V in Ba2BiVO6, MACE-MP-0-relaxed",
            model="MACE-MP-0 small")
        write_vasp_package(target, frames_path, manifest_path, levels=("pbe_u",),
                           kspacing=KSPACING, incar_extra={"pbe_u": {"NCORE": 4, "KPAR": 4}},
                           slurm=VaspSlurmSettings(time="24:00:00", max_parallel=None))
    print(f"{len(frames)} frames -> {WORK / 'snb_frames.extxyz'} and {target}")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    labels = [row["label"] for row in runs["V"]]
    x = np.arange(len(labels))
    for element in ("V", *DOPANTS):
        name = "V (pristine control)" if element == "V" else f"{element}_V"
        rows = results["systems"][element]
        valid = [r["valid"] and runs["V"][i]["valid"] for i, r in enumerate(rows)]
        axes[0].plot(x, [r["dE_meV"] if r["valid"] else np.nan for r in rows], "o-",
                     color=COLORS[element], lw=1.6, ms=4, label=name)
        if element != "V":
            axes[1].plot(x, [r["ddE_meV"] if ok else np.nan for r, ok in zip(rows, valid,
                                                                              strict=True)],
                         "o-", color=COLORS[element], lw=1.6, ms=4, label=name)
    for ax, title, ylabel in zip(
            axes, ("Relaxed energy vs the unperturbed start",
                   "The dopant's own part: ΔE(M) − ΔE(V control)"),
            ("ΔE (meV per cell)", "ΔΔE (meV per cell)"), strict=True):
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xticks(x, labels=labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(True, color="#e4e3df", lw=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"ShakeNBreak starts around one dopant in Ba₂BiVO₆ ({n_atoms} atoms, "
                 "MACE-MP-0, fixed cell; gaps: rejected relaxations)", x=0.01, ha="left",
                 fontsize=11)
    (HERE / "images").mkdir(exist_ok=True)
    fig.savefig(HERE / "images" / "bbvo_snb.png", dpi=150)
    print(f"figure written ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
