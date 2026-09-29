"""Compare every model's spin ladders with UBPW91 and draw the figures.

    python analyze.py

Reads the key frames and ``WORK/predictions/*.npz``; writes ``results.json`` here
and the figures to ``images/``. Three tests, each split into Fe₂XY and Fe16:

* **vertical gaps**: at each hand-off, E(M − 2) − E(M) at the same geometry
  (positive = the higher spin is lower in energy). A spin-blind model gives 0.
* **chain ground state**: along a chain with at least two steps, which step's
  relaxed end is lowest. Different steps are different geometries, so a
  spin-blind model can still get this right from geometry alone.
* **relaxation energies**: E(end) − E(start) within one step (same M), and force
  RMSE on all key frames.

Frames with a UBPW91 force RMS above 5 eV/Å (the label report's outliers: stuck
SCF roots or blown-up geometries) are left out of every test.
"""

import json
from collections import defaultdict

import numpy as np

from common import HERE, IMAGES, MEV, WORK, predictions, read_key_frames

MODELS = {  # name -> (label, colour); spin-aware first
    "uma-s-1p2": ("UMA-s-1p2 (omol)", "#2a78d6"),
    "uma-s-1p1": ("UMA-s-1p1 (omol)", "#1baf7a"),
    "mace-mpa-0-medium": ("MACE-MPA-0 (spin-blind)", "#eb6834"),
    "mace-mp-0-small": ("MACE-MP-0 small (spin-blind)", "#8a8985"),
    "fe2-spin-mace": ("Fe₂ spin-MACE (trained here)", "#4a3aa7"),
}
OUTLIER_RMS = 5.0
TRAINED = {"fe2-spin-mace"}  # scored on its test chains only


def stats(pred, ref):
    pred, ref = np.asarray(pred), np.asarray(ref)
    err = pred - ref
    return {"n": int(len(ref)), "mae_meV": float(MEV * np.abs(err).mean()),
            "rmse_meV": float(MEV * np.sqrt((err ** 2).mean())),
            "bias_meV": float(MEV * err.mean())}


def test_split_parents():
    """Source structures of the trained model's test split (empty before training)."""
    from ase.io import read

    path = WORK / "fe2_spin" / "data" / "test.extxyz"
    if not path.is_file():
        return set()
    return {a.info["parent_record_id"] for a in read(path, ":")}


def structure(frames):
    """{chain: [(step, M, start index, end index)]} over the key frames."""
    chains = defaultdict(dict)
    for i, atoms in enumerate(frames):
        info = atoms.info
        entry = chains[info["chain"]].setdefault(
            int(info["step"]), [int(info["multiplicity"]), None, None])
        if info["role"] in ("start", "start_end"):
            entry[1] = i
        if info["role"] in ("end", "start_end"):
            entry[2] = i
    return {c: [(s, *steps[s]) for s in sorted(steps)] for c, steps in chains.items()}


def main():
    frames = read_key_frames()
    ref_e = np.array([a.info["REF_energy"] for a in frames])
    ref_f = [a.arrays["REF_forces"] for a in frames]
    fam = np.array([a.info["family"] for a in frames])
    bad = np.array([np.sqrt((f ** 2).sum(1).mean()) > OUTLIER_RMS for f in ref_f])
    chains = structure(frames)
    models = {name: predictions(name) for name in MODELS
              if (WORK / "predictions" / f"{name}.npz").is_file()}

    # chain groups: the two families, and the Fe₂XY chains of the trained
    # model's test split (whole source structures it never saw)
    groups = {f: {c for c, st in chains.items() if fam[st[0][2]] == f} for f in ("Fe2", "Fe16")}
    test_parents = test_split_parents()
    if test_parents:
        groups["Fe2_test"] = {c for c in groups["Fe2"]
                              if frames[chains[c][0][2]].info["parent_record_id"] in test_parents}
    # vertical hand-offs: (high-M end, low-M start, chain) triples
    pairs = [(steps[k][3], steps[k + 1][2], c)
             for c, steps in chains.items() for k in range(len(steps) - 1)]
    pairs = [p for p in pairs if not (bad[p[0]] or bad[p[1]])]
    # relaxations within a step
    relax = [(s[2], s[3], c) for c, steps in chains.items() for s in steps
             if s[2] != s[3] and not (bad[s[2]] or bad[s[3]])]
    # chains with >= 2 steps and no outlier ends
    ladders = {c: steps for c, steps in chains.items()
               if len(steps) >= 2 and not any(bad[s[3]] for s in steps)}

    results = {"frames": len(frames), "outlier_frames": int(bad.sum()),
               "handoffs": {g: sum(p[2] in cs for p in pairs) for g, cs in groups.items()},
               "ladders": {g: sum(c in cs for c in ladders) for g, cs in groups.items()},
               "reference_ladders": {}, "models": {}}
    ref_gap = {g: np.array([ref_e[lo] - ref_e[hi] for hi, lo, c in pairs if c in cs])
               for g, cs in groups.items()}
    for g in groups:
        gaps = ref_gap[g]
        results["reference_ladders"][g] = {
            "gap_median_meV": float(MEV * np.median(gaps)),
            "gap_iqr_meV": [float(MEV * q) for q in np.percentile(gaps, [25, 75])],
            "fraction_high_spin_lower": float(np.mean(gaps > 0)),
        }

    for name, pred in models.items():
        e, forces = pred["energy"], pred["forces"]
        out = {}
        for g, cs in groups.items():
            if name in TRAINED and g != "Fe2_test":
                continue  # it was trained on those chains
            gap = np.array([e[lo] - e[hi] for hi, lo, c in pairs if c in cs])
            ref = ref_gap[g]
            vertical = stats(gap, ref)
            # a spin-blind model's gaps are 0 up to float noise, whose sign means nothing
            vertical["sign_agreement"] = (float(np.mean(np.sign(gap) == np.sign(ref)))
                                          if np.abs(gap).max() > 1e-3 else None)
            vertical["pearson_r"] = (float(np.corrcoef(gap, ref)[0, 1])
                                     if np.std(gap) > 1e-9 else None)
            rel = [(a, b) for a, b, c in relax if c in cs]
            relaxation = stats([e[b] - e[a] for a, b in rel],
                               [ref_e[b] - ref_e[a] for a, b in rel])
            hits = total = 0
            for c, steps in ladders.items():
                if c not in cs:
                    continue
                ends = [s[3] for s in steps]
                hits += int(np.argmin(e[ends])) == int(np.argmin(ref_e[ends]))
                total += 1
            idx = [i for i in range(len(frames)) if frames[i].info["chain"] in cs and not bad[i]]
            diff = np.concatenate([(forces[i] - ref_f[i]).ravel() for i in idx])
            out[g] = {"vertical_gap": vertical, "relaxation": relaxation,
                      "chain_ground_state": {"correct": hits, "chains": total,
                                             "fraction": hits / total if total else None},
                      "force_rmse_eV_per_A": float(np.sqrt((diff ** 2).mean()))}
        results["models"][name] = out
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    figures(frames, ref_e, pairs, ref_gap, ladders, models, results, groups)
    print(json.dumps(results, indent=1))


def parity(ax, ref, series, title):
    """Model spin gaps against UBPW91 on one panel; ``series`` = [(name, gaps, stats)]."""
    for name, gap, s in series:
        label, color = MODELS[name]
        ax.plot(ref, gap, "o", ms=4, alpha=0.55, color=color, mec="none",
                label=f"{label}: MAE {s['mae_meV'] / MEV:.2f} eV, "
                      f"sign {100 * s['sign_agreement']:.0f}%")
    lo, hi = np.percentile(ref, [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="#c3c2b7", lw=1, zorder=0)
    ax.axhline(0, color="#8a8985", lw=1.5, ls="--", zorder=0,
               label="spin-blind MACE (always 0)")
    ax.axvline(0, color="#e4e3df", lw=0.8, zorder=0)
    ax.set_xlim(lim)
    ax.set_xlabel("UBPW91 E(M−2) − E(M), same geometry (eV)")
    ax.set_ylabel("model E(M−2) − E(M) (eV)")
    ax.set_title(f"{title}: {len(ref)} vertical spin gaps", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(True, color="#e4e3df", lw=0.6)


def figures(frames, ref_e, pairs, ref_gap, ladders, models, results, groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMAGES.mkdir(exist_ok=True)
    titles = {"Fe2": "Fe₂XY molecules", "Fe16": "Fe16, Fe16N₂",
              "Fe2_test": "Fe₂XY, held-out chains"}
    panels = [("Fe2", ("uma-s-1p1", "uma-s-1p2")), ("Fe16", ("uma-s-1p1", "uma-s-1p2"))]
    if "Fe2_test" in groups and "fe2-spin-mace" in models:
        panels.append(("Fe2_test", ("uma-s-1p2", "fe2-spin-mace")))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    for ax, (g, names) in zip(axes, panels):
        series = []
        for name in names:
            if name not in models:
                continue
            e = models[name]["energy"]
            gap = np.array([e[lo] - e[hi] for hi, lo, c in pairs if c in groups[g]])
            series.append((name, gap, results["models"][name][g]["vertical_gap"]))
        parity(ax, ref_gap[g], series, titles[g])

    ax = axes[3]
    shown = [g for g in ("Fe2", "Fe16", "Fe2_test") if g in groups]
    names = [n for n in MODELS if n in models]
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        label, color = MODELS[name]
        x = np.arange(len(shown)) + (k - (len(names) - 1) / 2) * width
        vals = [results["models"][name].get(g, {}).get("chain_ground_state", {}).get("fraction")
                for g in shown]
        keep = [(xi, v) for xi, v in zip(x, vals) if v is not None]
        ax.bar([xi for xi, _ in keep], [100 * v for _, v in keep], width * 0.92,
               color=color, label=label)
        for xi, v in keep:
            ax.text(xi, 100 * v + 1.5, f"{100 * v:.0f}", ha="center", fontsize=7,
                    color="#52514e")
    counts = results["ladders"]
    ax.set_xticks(range(len(shown)), [f"{titles[g]}\n({counts[g]} chains)" for g in shown],
                  fontsize=9)
    ax.set_ylabel("chains whose lowest step matches UBPW91 (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Lowest-energy step along each chain", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    ax.grid(True, axis="y", color="#e4e3df", lw=0.6)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    if len(panels) < 3:
        axes[2].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "spin_gaps.png", dpi=150)
    plt.close(fig)

    # example ladders: the longest held-out Fe₂XY chain (every model is fair
    # there) and the longest Fe16 chain
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fe2 = groups.get("Fe2_test") or groups["Fe2"]
    for ax, members in zip(axes, (fe2, groups["Fe16"])):
        chain = max((c for c in ladders if c in members), key=lambda c: (len(ladders[c]), c))
        steps = ladders[chain]
        idx = [i for s in steps for i in (s[2], s[3])]
        ms = [s[1] for s in steps for _ in (0, 1)]
        x = np.arange(len(idx))
        ax.plot(x, ref_e[idx] - ref_e[idx].min(), "o-", color="#0b0b0b", lw=2, ms=6,
                label="UBPW91", zorder=5)
        for name in names:
            if name in TRAINED and chain not in groups.get("Fe2_test", ()):
                continue
            label, color = MODELS[name]
            e = models[name]["energy"][idx]
            ax.plot(x, e - e.min(), "o-", color=color, lw=2, ms=5, label=label)
        ax.set_xticks(x, [f"M={m}\n{'start' if j % 2 == 0 else 'end'}"
                          for j, m in enumerate(ms)], fontsize=7)
        ax.set_ylabel("energy relative to the chain minimum (eV)")
        ax.set_title(f"{frames[idx[0]].info['formula']} (q={frames[idx[0]].info['charge']}),"
                     f" chain {chain.split('-')[0][:8]}", fontsize=11)
        ax.grid(True, color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES / "example_chains.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
