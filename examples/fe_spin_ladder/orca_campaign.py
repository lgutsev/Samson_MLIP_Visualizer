"""The functional check at scale, as a LONI package: ORCA PBE and ωB97M-V spin gaps.

    python orca_campaign.py write DIR      # needs ASE (defects env)
    python orca_campaign.py collect DIR [--out FILE]   # after the outputs are back

Psi4 on the laptop converged only 2 of the 8 spin gaps of ``functional_check.py``
in a day (open-shell Fe₂ SCFs at ωB97M-V are slow and fragile), so the check goes to
QB4 as one ORCA 6.1.1 package, the only code there with ωB97M-V (UMA's level):

- **every held-out Fe₂XY hand-off** (the test split of ``train_fe2_spin.py``; the
  trained model is scored there too) plus the eight Psi4 picks: each gives two
  structures, the same geometry at M and M − 2;
- **the six N₂ + Fe₂O₄ structures** at M = 5: the two UBPW91 end points, and UMA's
  reactant, step-1 TS, N₂O intermediate and nitrosyl product from SAMSON.

Each input runs two jobs on the same structure: PBE/def2-TZVP (a GGA in the same
code, against the UBPW91 labels), then ωB97M-V/def2-TZVP started from the PBE
orbitals (``$new_job`` reuses the .gbw), so both functionals are evaluated in the
same basin. UKS throughout, DefGrid3, TightSCF, SlowConv (TRAH takes over if DIIS
stalls). Singlets may land on the closed-shell solution where the Gaussian ladder
had a broken-symmetry one; ``collect`` flags them by ⟨S²⟩.

Four 16-core inputs run side by side per 64-core node (workq), with the ORCA
environment verified on QB4 (ORCA_ON_LONI.md in the dispatch repo). ``collect``
parses both energies and ⟨S²⟩ per structure and writes ``campaign_results.json``
into the package folder (or ``--out``): every gap and reaction energy at UBPW91
(labels), PBE, ωB97M-V, UMA-s-1p2 and the trained spin-MACE, the MAE, r and max
error of each comparison, and the gaps whose ⟨S²⟩ is not comparable (a structure
that changes basin between PBE and ωB97M-V, or whose two ends differ in spin
contamination by more than 1), counted separately.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from common import MEV, WORK

HARTREE = 27.211386245988
NPROC = 16
MAXCORE_MB = 3500  # QB4 CPU nodes: 256 GB for 64 cores
LEVELS = {"PBE": "PBE def2-TZVP def2/J RIJCOSX", "wB97M-V": "wB97M-V def2-TZVP def2/J RIJCOSX"}
# ORCA 6.1.1 + OpenMPI 4.1.6 as verified on QB4 on 2026-09-29 (dispatch repo,
# ORCA_ON_LONI.md): ORCA starts its own mpirun, so the right one must come first
ENVIRONMENT = (
    "OMPI_DIR=/project/lgutsev/openmpi-4.1.6",
    "ORCA_DIR=/project/lgutsev/Orca_6_1_1",
    'export PATH="$ORCA_DIR:$OMPI_DIR/bin:$PATH"',
    'export LD_LIBRARY_PATH="$ORCA_DIR/lib:$OMPI_DIR/lib:${LD_LIBRARY_PATH:-}"',
    "unset OPAL_PREFIX",
    "export OMP_NUM_THREADS=1",
)


def structures():
    """The structures to label, each with its metadata (in ``atoms.info``)."""
    from ase.io import read

    from analyze import structure, test_split_parents
    from common import predictions, read_key_frames

    frames = read_key_frames()
    uma = predictions("uma-s-1p2")["energy"]
    spin_mace = predictions("fe2-spin-mace")["energy"]
    chains, test = structure(frames), test_split_parents()
    pairs = []
    for steps in chains.values():
        if frames[steps[0][2]].info["family"] != "Fe2":
            continue
        if frames[steps[0][2]].info["parent_record_id"] not in test:
            continue
        pairs += [(steps[k][3], steps[k + 1][2]) for k in range(len(steps) - 1)]
    picked = json.loads((WORK / "functional_pairs.json").read_text())
    pairs += [tuple(int(i) for i in key.split("-")) for key in picked if "-" in key]
    pairs = sorted(set(pairs))
    # A structure with an atom that has no neighbour within 3 Å is a fragment, not a
    # vertical gap of one cluster; in package 16 (batch 03) the Fe2HO3 chain of key
    # frames 1751-1758, with H 6.8 Å out, broke the PBE SCF in all 8 inputs.
    pairs = [(hi, lo) for hi, lo in pairs if not (fragmented(frames[hi]) or fragmented(frames[lo]))]
    out = []
    for hi, lo in pairs:
        for role, index in (("high", hi), ("low", lo)):
            atoms = frames[index].copy()
            atoms.info = {
                "group": "gap", "pair": f"{hi}-{lo}", "role": role, "key_frame": index,
                "formula": frames[index].info["formula"],
                "charge": int(frames[index].info["charge"]),
                "multiplicity": int(frames[index].info["multiplicity"]),
                "held_out": frames[index].info["parent_record_id"] in test,
                "UBPW91_eV": float(frames[index].info["REF_energy"]),
                "UMA_s_1p2_eV": float(uma[index]), "fe2_spin_mace_eV": float(spin_mace[index]),
            }
            out.append(atoms)
    folder = WORK / "n2_no"
    ends = read(folder / "uma_relaxed_ends.extxyz")  # SAMSON merged both models
    reaction = [
        ("ubpw91_reactant", frames[2736]), ("ubpw91_product", frames[2828]),
        ("uma_reactant", read(folder / "step1_reactant.xyz")),
        ("uma_ts", read(folder / "step1_ts.xyz")),
        ("uma_intermediate", read(folder / "step1_intermediate.xyz")),
        ("uma_product", ends[8:]),
    ]
    for name, source in reaction:
        atoms = source.copy()
        atoms.info = {"group": "reaction", "name": name, "formula": "Fe2N2O4",
                      "charge": 0, "multiplicity": 5}
        out.append(atoms)
    return out


def fragmented(atoms, cutoff: float = 3.0) -> bool:
    distances = atoms.get_all_distances()
    np.fill_diagonal(distances, np.inf)
    return bool((distances.min(axis=1) > cutoff).any())


def orca_input(atoms) -> str:
    q, m = atoms.info["charge"], atoms.info["multiplicity"]
    xyz = "".join(f"  {s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n"
                  for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions))
    jobs = []
    for level in LEVELS.values():
        jobs.append(f"""! UKS {level} TightSCF SlowConv DefGrid3
%pal
  nprocs {NPROC}
end
%maxcore {MAXCORE_MB}
%scf
  MaxIter 500
end
* xyz {q} {m}
{xyz}*
""")
    # the second job starts from the first job's orbitals (same basename .gbw)
    return "\n$new_job\n".join(jobs)


def write(directory: Path) -> None:
    from ase.io import write as write_frames

    from samson_mlip_visualizer.labeling import SlurmSettings, _script

    items = structures()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    (directory / "logs").mkdir()
    (directory / "logs" / "README.txt").write_text("SLURM writes one log per array task here.\n",
                                                   newline="\n")
    for index, atoms in enumerate(items):
        (directory / "inputs" / f"frame_{index:04d}.inp").write_text(
            orca_input(atoms), encoding="utf-8", newline="\n")
    write_frames(directory / "frames.extxyz", items)
    slurm = SlurmSettings(account="loni_perovsk27", partition="workq", modules=(),
                          cpus=NPROC, memory_gb=NPROC * MAXCORE_MB // 1000, time="24:00:00",
                          max_parallel=10, jobs_per_task=4, setup=ENVIRONMENT)
    script = _script("orca", len(items), slurm, job="energy")
    (directory / "run_orca.slurm").write_text(script, encoding="utf-8", newline="\n")
    gaps = sum(a.info["group"] == "gap" for a in items) // 2
    (directory / "package.json").write_text(json.dumps({
        "code": "orca", "job": "energy", "levels": LEVELS, "structures": len(items),
        "spin_gaps": gaps, "reaction_structures": len(items) - 2 * gaps,
        "generator": "examples/fe_spin_ladder/orca_campaign.py (Samson_MLIP_Visualizer)",
    }, indent=1), encoding="utf-8")
    (directory / "README.md").write_text(f"""# 16: ORCA PBE and ωB97M-V spin gaps of Fe₂XY, and N₂ on Fe₂O₄

Generated by `examples/fe_spin_ladder/orca_campaign.py` in Samson_MLIP_Visualizer.
Nothing was submitted from the desktop.

{len(items)} structures, one ORCA input each (`inputs/frame_NNNN.inp`), each with two
jobs on the same geometry: PBE/def2-TZVP, then ωB97M-V/def2-TZVP from the PBE orbitals
(`$new_job`). {gaps} vertical spin gaps (a geometry at M and at M − 2, from ClusterMLIP's
UBPW91 Warehouse 2 chains) and {len(items) - 2 * gaps} N₂ + Fe₂O₄ structures at M = 5.
`frames.extxyz` holds every structure with its charge, multiplicity and the reference
energies.

- `sbatch run_orca.slurm` (or `bash submit_smokes.sh` from the checkout): 4 inputs per
  64-core workq node, 16 cores each; array tasks of 24 h, 10 at a time.
- ORCA 6.1.1 with OpenMPI 4.1.6 (ORCA_ON_LONI.md); no `--mem` (QB4 rejects it).
- Done when `outputs/frame_NNNN.out` ends with `ORCA TERMINATED NORMALLY`. Both energies
  must be there: an unconverged second job still terminates normally, and `collect`
  checks for it.

Bring back `outputs/` and run, in the owning repo:
`python examples/fe_spin_ladder/orca_campaign.py collect <this folder>`.
""", encoding="utf-8", newline="\n")
    print(f"{len(items)} structures ({gaps} gaps) -> {directory}")


def parse(text: str) -> dict:
    energies = [float(e) for e in re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)]
    s2 = [float(v) for v in re.findall(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)", text)]
    converged = "SCF NOT CONVERGED" not in text and "The SCF is NOT converged" not in text
    return {"energies": energies, "s2": s2, "converged": converged,
            "terminated": "ORCA TERMINATED NORMALLY" in text}


def collect(directory: Path, out_file: Path | None = None) -> None:
    from ase.io import read

    items = read(directory / "frames.extxyz", ":")
    rows = []
    for index, atoms in enumerate(items):
        path = directory / "outputs" / f"frame_{index:04d}.out"
        row = dict(atoms.info)
        if path.is_file():
            parsed = parse(path.read_text(errors="replace"))
            ok = parsed["terminated"] and parsed["converged"] and len(parsed["energies"]) == 2
            row.update(ok=ok, s2=parsed["s2"][-2:] if len(parsed["s2"]) >= 2 else parsed["s2"])
            if ok:
                row["PBE_eV"], row["wB97M-V_eV"] = (e * HARTREE for e in parsed["energies"])
        else:
            row["ok"] = False
        rows.append(row)
    gaps = {}
    for row in rows:
        if row["group"] == "gap":
            gaps.setdefault(row["pair"], {})[row["role"]] = row
    out = {"gaps": {}, "reaction": {}}
    for key, pair in gaps.items():
        if not all(pair.get(r, {}).get("ok") for r in ("high", "low")):
            out["gaps"][key] = {"ok": False}
            continue
        hi, lo = pair["high"], pair["low"]
        flags = [f"{role} changes basin" for role, row in (("high", hi), ("low", lo))
                 if basin_change(row)]
        excess = [contamination(row) for row in (hi, lo)]
        if any(abs(a - b) > 1 for a, b in zip(*excess)):
            flags.append("<S^2> contamination of the two ends differs by > 1")
        out["gaps"][key] = {
            "formula": hi["formula"], "charge": int(hi["charge"]),
            "M": [int(hi["multiplicity"]), int(lo["multiplicity"])],
            "held_out": bool(hi["held_out"]),
            **{f"{level}_meV": MEV * (lo[f"{level}_eV"] - hi[f"{level}_eV"])
               for level in ("UBPW91", "PBE", "wB97M-V", "UMA_s_1p2", "fe2_spin_mace")},
            "s2": {"high": hi["s2"], "low": lo["s2"]}, "flags": flags,
        }
    reaction = {row["name"]: row for row in rows if row["group"] == "reaction" and row["ok"]}
    for name, (a, b) in {"n2_to_2no_ubpw91_geometries": ("ubpw91_reactant", "ubpw91_product"),
                         "n2_to_2no_uma_geometries": ("uma_reactant", "uma_product"),
                         "step1_barrier": ("uma_reactant", "uma_ts"),
                         "step1_intermediate": ("uma_reactant", "uma_intermediate")}.items():
        if a in reaction and b in reaction:
            out["reaction"][name] = {
                "PBE": reaction[b]["PBE_eV"] - reaction[a]["PBE_eV"],
                "wB97M-V": reaction[b]["wB97M-V_eV"] - reaction[a]["wB97M-V_eV"],
                "s2_wB97M-V": [reaction[a]["s2"][-1], reaction[b]["s2"][-1]],
            }
    out["failed"] = [{"frame": index, **{k: rows[index].get(k) for k in
                                         ("pair", "name", "formula", "key_frame")}}
                     for index in range(len(rows)) if not rows[index]["ok"]]
    good = [g for g in out["gaps"].values() if g.get("PBE_meV") is not None]
    clean = [g for g in good if not g["flags"]]
    if good:
        out["summary"] = {"gaps_ok": len(good), "gaps_total": len(out["gaps"]),
                          "gaps_without_s2_flags": len(clean),
                          "all": compare(good), "without_s2_flags": compare(clean)}
    target = out_file or directory / "campaign_results.json"
    target.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out.get("summary", {}), indent=1), f"\n{len(out['failed'])} failed",
          "\n" + json.dumps(out["reaction"], indent=1), "\n->", target)


def s_s1(multiplicity: int) -> float:
    s = (multiplicity - 1) / 2
    return s * (s + 1)


def contamination(row) -> list[float]:
    """<S^2> − S(S+1) at each level (PBE, ωB97M-V)."""
    return [value - s_s1(row["multiplicity"]) for value in row["s2"]]


def basin_change(row) -> bool:
    """PBE and ωB97M-V landed on visibly different spin states of the same structure
    (a singlet going from closed shell to broken symmetry, say)."""
    return len(row["s2"]) == 2 and abs(row["s2"][0] - row["s2"][1]) > 0.5


PAIRS = {"PBE_vs_UBPW91": ("PBE_meV", "UBPW91_meV"),
         "wB97M-V_vs_PBE": ("wB97M-V_meV", "PBE_meV"),
         "UMA_vs_wB97M-V": ("UMA_s_1p2_meV", "wB97M-V_meV"),
         "UMA_vs_UBPW91": ("UMA_s_1p2_meV", "UBPW91_meV"),
         "spin_mace_vs_UBPW91": ("fe2_spin_mace_meV", "UBPW91_meV")}


def compare(gaps) -> dict:
    """MAE, max error, Pearson r and the share within 0.3 eV for each comparison."""
    if len(gaps) < 2:
        return {"n": len(gaps)}
    result = {"n": len(gaps)}
    for name, (a, b) in PAIRS.items():
        x, y = np.array([g[a] for g in gaps]), np.array([g[b] for g in gaps])
        error = np.abs(x - y)
        result[name] = {"mae_meV": float(error.mean()), "max_meV": float(error.max()),
                        "r": float(np.corrcoef(x, y)[0, 1]),
                        "within_300_meV": int((error <= 300).sum())}
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    out_file = None
    if "--out" in args:
        at = args.index("--out")
        out_file = Path(args[at + 1])
        del args[at:at + 2]
    if len(args) != 2 or args[0] not in ("write", "collect"):
        sys.exit(__doc__)
    if args[0] == "write":
        write(Path(args[1]))
    else:
        collect(Path(args[1]), out_file)
