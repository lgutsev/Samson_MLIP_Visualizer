"""Is a UMA–UBPW91 spin-gap difference the model, or the functional?

    <defects env>/python functional_check.py pick [N_PAIRS]   # needs ASE
    <defects env>/python functional_check.py reaction         # adds N2 -> 2 NO
    <qm env>/python functional_check.py run [KEYS]           # Psi4 1.11, no ASE

UMA's omol head learned ωB97M-V (a range-separated hybrid); the labels are
UBPW91 (a GGA). Hybrids are known to favour high spin in Fe compounds, so the
two can disagree on a spin gap with neither model at fault. ``pick`` takes a
few small Fe₂XY hand-offs (q = 0, at most 6 atoms, spread over the UBPW91 gap
range) and writes them to ``WORK/functional_pairs.json``; ``run`` computes the
same vertical gap with Psi4 at both levels, def2-TZVP (Psi4 has no 6-311++G*
for Fe):

* BPW91 against the Gaussian UBPW91 gap checks that Psi4 lands on the same states;
* ωB97M-V against UMA separates model error from the functional shift.

Each multiplicity starts from its own SAD guess (a Psi4 restart file imposes the
occupation it was written with, so the Gaussian ladder's checkpoint hand-off
cannot be copied), and ωB97M-V starts from the BPW91 orbitals of the same state.
Psi4 may land on a different state than Gaussian; the BPW91 column shows how
often. Results go to
``WORK/functional_check.json``, one pair at a time, each skipped when done.
"""

import json
import sys

import numpy as np

from common import MEV, WORK

HARTREE = 27.211386245988
LEVELS = {"BPW91": {"name": "BPW91", "x_functionals": {"GGA_X_B88": {}},
                    "c_functionals": {"GGA_C_PW91": {}}},
          "wB97M-V": "wb97m-v"}
PAIRS = WORK / "functional_pairs.json"
OUT = WORK / "functional_check.json"


def pick(n):
    from analyze import structure
    from common import predictions, read_key_frames

    frames = read_key_frames()
    uma = predictions("uma-s-1p2")["energy"]
    candidates = []
    for steps in structure(frames).values():
        for k in range(len(steps) - 1):
            hi, lo = steps[k][3], steps[k + 1][2]
            a = frames[hi]
            if a.info["family"] == "Fe2" and a.info["charge"] == 0 and len(a) <= 6:
                candidates.append((hi, lo, frames[lo].info["REF_energy"] - a.info["REF_energy"]))
    candidates.sort(key=lambda p: p[2])
    # spread over the gap range, one pair per (formula, M, gap sign)
    picks, seen = {}, set()
    grid = np.linspace(0.02, 0.98, 4 * n)
    order = [grid[i // 2] if i % 2 == 0 else grid[-1 - i // 2] for i in range(len(grid))]
    for q in order:  # alternate between both ends of the range
        hi, lo, gap = candidates[int(q * (len(candidates) - 1))]
        a = frames[hi]
        key = (a.info["formula"], int(a.info["multiplicity"]), np.sign(gap))
        if key in seen:
            continue
        seen.add(key)
        picks[f"{hi}-{lo}"] = {
            "formula": a.info["formula"], "chain": a.info["chain"],
            "symbols": a.get_chemical_symbols(), "positions": a.positions.tolist(),
            "M_high": int(a.info["multiplicity"]),
            "M_low": int(frames[lo].info["multiplicity"]),
            "UBPW91_gaussian_meV": MEV * gap,
            "UMA_s_1p2_meV": MEV * float(uma[lo] - uma[hi]),
        }
        if len(picks) == n:
            break
    PAIRS.write_text(json.dumps(picks, indent=1))
    for key, row in picks.items():
        print(key, row["formula"], row["M_high"], "->", row["M_low"],
              f"UBPW91 {row['UBPW91_gaussian_meV']:.0f} meV, UMA {row['UMA_s_1p2_meV']:.0f} meV")


def add_reaction():
    """Append N₂ + Fe₂O₄ → Fe₂O₂(NO)₂ (M = 5, the UBPW91 geometries of
    ``n2_to_no_endpoints.py``) to the pairs: the same two functionals on a
    reaction energy instead of a spin gap."""
    from ase.io import read

    folder = WORK / "n2_no"
    ends = json.loads((folder / "endpoints.json").read_text())
    reactant, product = read(folder / "reactant.xyz"), read(folder / "product.xyz")
    pairs = json.loads(PAIRS.read_text())
    energies = ends["reaction_energy_eV_at_UBPW91_geometries"]
    pairs["reaction_n2_to_2no"] = {
        "formula": "Fe2N2O4", "chain": f"{ends['reactant_chain']} -> {ends['product_chain']}",
        "symbols": reactant.get_chemical_symbols(), "positions": reactant.positions.tolist(),
        "positions_low": product.positions.tolist(), "M_high": 5, "M_low": 5,
        "UBPW91_gaussian_meV": MEV * energies["UBPW91"],
        "UMA_s_1p2_meV": MEV * energies["uma-s-1p2"],
    }
    PAIRS.write_text(json.dumps(pairs, indent=1))
    print("added reaction_n2_to_2no")


SCF = {"basis": "def2-tzvp", "reference": "uks", "scf_type": "df", "maxiter": 300,
       "damping_percentage": 20, "level_shift": 0.3, "level_shift_cutoff": 1e-3,
       # at a converged density (d < 1e-5) the energy still wanders: ~1e-4 Eh for
       # BPW91, up to ~8e-4 Eh for ωB97M-V (meta-GGA + VV10), even on this fine
       # grid; 1e-3 Eh (27 meV) is the uncertainty accepted, against gaps of
       # hundreds of meV
       "dft_radial_points": 99, "dft_spherical_points": 590,
       "e_convergence": 1e-3, "d_convergence": 1e-5}
# SAD first; if that fails (no convergence, or Psi4's "ADIIS minimization failed"),
# the SAP guess without ADIIS and with second-order steps near convergence
ATTEMPTS = ({"guess": "sad"},
            {"guess": "sapgau", "scf_initial_accelerator": "none", "soscf": True,
             "soscf_start_convergence": 1e-3, "maxiter": 400})


def scf(psi4, mol, multiplicity, level, options, write=None, read=None):
    psi4.core.clean()
    psi4.core.clean_timers()  # a failed SCF leaves its timers running
    psi4.core.clean_options()
    psi4.set_options({**SCF, **options})
    kwargs = {"write_orbitals": str(write)} if write else {}
    if read:
        kwargs["restart_file"] = str(read)
    e, wfn = psi4.energy("scf", dft_functional=LEVELS[level], molecule=mol, return_wfn=True,
                         **kwargs)
    if wfn.nalpha() - wfn.nbeta() != multiplicity - 1:
        raise RuntimeError(f"SCF ended with {wfn.nalpha()}a/{wfn.nbeta()}b, not M={multiplicity}")
    return e * HARTREE


def state(psi4, symbols, positions, multiplicity, orbitals):
    """BPW91 and ωB97M-V energies (eV) of one geometry at one multiplicity.

    BPW91 starts from SAD (then SAP); ωB97M-V starts from the BPW91 orbitals of
    the same state, so both functionals are evaluated in the same basin, the one
    the Gaussian ladder labelled. (Only across multiplicities does a restart file
    go wrong: it imposes the occupation it was written with.)
    """
    lines = [f"0 {multiplicity}"] + [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z)
                                      in zip(symbols, positions)]
    mol = psi4.geometry("\n".join(lines + ["symmetry c1", "no_reorient", "no_com"]))
    out, errors = {}, {}
    for options in ATTEMPTS:
        try:
            out["BPW91"] = scf(psi4, mol, multiplicity, "BPW91", options, write=orbitals)
            break
        except Exception as exc:  # noqa: BLE001 - recorded
            errors["BPW91"] = f"{type(exc).__name__}: {exc}"[:200]
    starts = ([{"guess": "read"}] if "BPW91" in out else []) + list(ATTEMPTS)
    for options in starts:
        try:
            read = orbitals if options.get("guess") == "read" else None
            out["wB97M-V"] = scf(psi4, mol, multiplicity, "wB97M-V",
                                 {k: v for k, v in options.items() if k != "guess"}
                                 if read else options, read=read)
            break
        except Exception as exc:  # noqa: BLE001 - recorded
            errors["wB97M-V"] = f"{type(exc).__name__}: {exc}"[:200]
    return out, {k: v for k, v in errors.items() if k not in out}


def run(only=None):
    """Every pair, or with ``only`` those whose key contains one of its comma-separated
    parts, in their own result file (so two runs can go side by side)."""
    import psi4

    pairs = json.loads(PAIRS.read_text())
    out = OUT
    if only == "gaps":  # the spin gaps only (keys "<high>-<low>"), in the main file
        pairs = {k: v for k, v in pairs.items() if "-" in k}
    elif only:
        parts = only.split(",")
        pairs = {k: v for k, v in pairs.items() if any(part in k for part in parts)}
        out = OUT.with_name(f"functional_check_{only.replace(',', '_')}.json")
    results = json.loads(out.read_text()) if out.is_file() else {}
    psi4.set_memory("3 GB")  # 6 GB failed (bad allocation) beside the GPU training
    psi4.set_num_threads(6)  # 24 cores: two runs side by side and the GPU training
    psi4.core.set_output_file(str(out.with_suffix(".out")), True)
    scratch = WORK / "psi4_orbitals"
    scratch.mkdir(exist_ok=True)
    for key, pair in pairs.items():
        row = results.get(key) or {}
        if all(f"{level}_meV" in row for level in LEVELS):
            continue
        # every missing level is recomputed for both states, with the protocol above
        row = {k: v for k, v in pair.items() if k not in ("symbols", "positions", "positions_low")}
        hi, hi_err = state(psi4, pair["symbols"], pair["positions"], pair["M_high"],
                           scratch / f"{key}_high.npy")
        # a reaction entry has a second geometry; a spin gap reuses the first
        lo, lo_err = state(psi4, pair["symbols"], pair.get("positions_low", pair["positions"]),
                           pair["M_low"], scratch / f"{key}_low.npy")
        for level in LEVELS:
            if level in hi and level in lo:
                row[f"{level}_meV"] = MEV * (lo[level] - hi[level])
            else:
                row[f"{level}_error"] = "; ".join(filter(None, [hi_err.get(level),
                                                               lo_err.get(level)]))
        results[key] = row
        out.write_text(json.dumps(results, indent=1))
        print(key, json.dumps(row), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["pick"]:
        pick(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif sys.argv[1:2] == ["reaction"]:
        add_reaction()
    elif sys.argv[1:2] == ["run"]:
        run(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
