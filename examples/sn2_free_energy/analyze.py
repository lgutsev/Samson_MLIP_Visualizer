"""Analyze the SN2 free-energy runs: profiles, P(ξ), rate, figures.

    python analyze.py [--no-figures]

For each system present in ``<work>``: thermodynamic integration of the
blue-moon windows (with error bars), ξ* (where the mean force crosses zero,
written to ``xi_star.json`` for ``launch.py ts``), slow-growth profiles both ways
(hysteresis), P(ξ) from free MD and −k_B T ln P(ξ), the metadynamics free
energy, and, once the ξ* run exists, ⟨|ξ̇*|⟩, the rate constant and the
phenomenological ΔA‡. Writes ``<work>/results.json`` and figures to
``<work>/figures`` (copied into ``images/`` of this example by hand).
"""

import json
import sys

import numpy as np
from common import COORDINATE, SN2, SYSTEMS, TEMPERATURE, WORK, calculator, folder

from samson_mlip_visualizer import free_energy as fe

KT = fe.KB * TEMPERATURE
KCAL = 23.0605
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
COLORS = {"blue_moon": "#2a78d6", "forward": "#eb6834", "reverse": "#1baf7a",
          "metadynamics": "#eda100", "free_md": "#e87ba4"}


def load(path):
    return dict(np.load(path)) if path.exists() else None


def blue_moon(system):
    files = sorted((folder(system) / "windows").glob("xi_*.npz"),
                   key=lambda p: float(p.stem[3:]))
    if not files:
        return None
    xi, gradient, error = [], [], []
    for path in files:
        data = load(path)
        record = fe.ConstrainedRecord(**{k: data[k] for k in (
            "target", "value", "lam", "z", "g", "temperature", "energy")})
        value, err = fe.blue_moon_gradient(record, TEMPERATURE, skip=500)
        xi.append(float(data["target"][0]))
        gradient.append(value)
        error.append(err)
    xi, gradient, error = map(np.array, (xi, gradient, error))
    if SYSTEMS[system]["symmetric"]:
        # dA/dξ is odd in ξ: mirror the ξ < 0 windows, and at ξ = 0 it vanishes exactly.
        gradient = np.where(xi == 0, 0.0, gradient)
        left = xi < 0
        xi = np.concatenate([xi, -xi[left][::-1]])
        gradient = np.concatenate([gradient, -gradient[left][::-1]])
        error = np.concatenate([error, error[left][::-1]])
    profile = fe.integrate_gradient(xi, gradient)
    # Error of the running trapezoid sum, windows independent.
    increments = (0.5 * np.diff(xi)) ** 2 * (error[1:] ** 2 + error[:-1] ** 2)
    profile_error = np.concatenate([[0.0], np.sqrt(np.cumsum(increments))])
    # The reactant minimum and ξ*: the first upward and downward zero crossings of dA/dξ.
    return {"xi": xi, "gradient": gradient, "error": error, "profile": profile,
            "profile_error": profile_error, "xi_min": zero_crossing(xi, gradient, up=True),
            "xi_star": zero_crossing(xi, gradient, up=False)}


def zero_crossing(xi, gradient, up):
    """ξ where dA/dξ first crosses zero upward (a minimum) or downward (a maximum)."""
    for k in range(len(xi) - 1):
        a, b = gradient[k], gradient[k + 1]
        if (a < 0 <= b) if up else (a > 0 >= b):
            return float(xi[k] - a * (xi[k + 1] - xi[k]) / (b - a))
    return None


def slow_growth(system):
    out = {}
    for direction in ("forward", "reverse"):
        data = load(folder(system) / f"slow_growth_{direction}.npz")
        if data is None:
            continue
        record = fe.ConstrainedRecord(**{k: data[k] for k in (
            "target", "value", "lam", "z", "g", "temperature", "energy")})
        xi, work = fe.slow_growth_profile(record)
        order = np.argsort(xi)
        xi, work = xi[order], work[order]
        out[direction] = {"xi": xi, "profile": work - work[0],
                          "temperature": float(record.temperature.mean())}
    return out


def static_path(system):
    """Potential energy along the model's own IRC, against ξ (F only)."""
    if system != "F":
        return None
    from ase.io import read

    frames = read(SN2 / "finetune_aimnet2" / "tuned_irc_fragments.extxyz", ":")[::4]
    calc = calculator("F")
    xi, energy = [], []
    for frame in frames:
        frame.calc = calc
        energy.append(frame.get_potential_energy())
        xi.append(COORDINATE.value(frame.positions))
    order = np.argsort(xi)
    return {"xi": np.array(xi)[order], "energy": np.array(energy)[order]}


def interpolate(xi, profile, at):
    return float(np.interp(at, xi, profile))


def analyze(system):
    result = {"label": SYSTEMS[system]["label"]}
    bm = blue_moon(system)
    sg = slow_growth(system)
    free = load(folder(system) / "free_md.npz")
    meta = load(folder(system) / "metadynamics.npz")
    ts = load(folder(system) / "ts_velocity.npz")
    static = static_path(system)
    if bm is not None:
        (folder(system) / "xi_star.json").write_text(json.dumps({"xi": bm["xi_star"]}))
        barrier = interpolate(bm["xi"], bm["profile"], bm["xi_star"]) - interpolate(
            bm["xi"], bm["profile"], bm["xi_min"])
        result["blue_moon"] = {
            "xi_star": bm["xi_star"], "xi_min": bm["xi_min"],
            "barrier_ev": barrier, "barrier_kcal": barrier * KCAL,
            "reaction_free_energy_ev": float(bm["profile"][-1] - interpolate(
                bm["xi"], bm["profile"], bm["xi_min"])),
            "profile_error_at_end_ev": float(bm["profile_error"][-1]),
        }
    for direction, data in sg.items():
        top = int(np.argmax(data["profile"][: len(data["profile"]) * 3 // 4]))
        low = int(np.argmin(data["profile"][:top + 1]))
        result[f"slow_growth_{direction}"] = {
            "barrier_ev": float(data["profile"][top] - data["profile"][low]),
            "xi_top": float(data["xi"][top]), "xi_min": float(data["xi"][low]),
            "end_ev": float(data["profile"][-1]), "mean_temperature_k": data["temperature"]}
    if len(sg) == 2:
        grid = np.linspace(max(sg["forward"]["xi"][0], sg["reverse"]["xi"][0]),
                           min(sg["forward"]["xi"][-1], sg["reverse"]["xi"][-1]), 200)
        gap = np.interp(grid, sg["forward"]["xi"], sg["forward"]["profile"]) - np.interp(
            grid, sg["reverse"]["xi"], sg["reverse"]["profile"])
        result["slow_growth_hysteresis_ev"] = {"max_abs": float(np.abs(gap).max()),
                                               "at_end": float(gap[-1])}
    if free is not None:
        centers, density = fe.probability_density(free["xi"], bins=60)
        reference = float(centers[np.argmax(density)])
        result["free_md"] = {"xi_ref": reference, "density_at_ref_per_A": float(density.max()),
                             "xi_mean": float(free["xi"].mean()),
                             "xi_range": [float(free["xi"].min()), float(free["xi"].max())]}
        if ts is not None and bm is not None:
            velocity = fe.generalized_velocity(ts["z"][500:], TEMPERATURE)
            barrier = interpolate(bm["xi"], bm["profile"], bm["xi_star"]) - interpolate(
                bm["xi"], bm["profile"], reference)
            rate, phenomenological = fe.rate_constant(barrier, float(density.max()), velocity,
                                                      TEMPERATURE)
            result["rate"] = {"xi_ref": reference, "delta_a_ref_to_star_ev": barrier,
                              "generalized_velocity_A_per_s": velocity,
                              "rate_constant_per_s": rate,
                              "phenomenological_barrier_ev": phenomenological,
                              "phenomenological_barrier_kcal": phenomenological * KCAL}
    if meta is not None:
        upper = float(meta["upper"]) if "upper" in meta else 1.0
        grid = np.linspace(-1.55, upper - 0.05, 200)
        fes = fe.fes_from_hills(grid, meta["centers"], meta["heights"], float(meta["sigma"]),
                                float(meta["bias_factor"]))
        split = int(np.argmin(np.abs(grid - (bm["xi_star"] if bm else 0.0))))
        low = int(np.argmin(fes[:split]))
        top = low + int(np.argmax(fes[low:split + 20]))
        crossings = np.count_nonzero(np.diff(np.sign(meta["xi"] - grid[split])))
        result["metadynamics"] = {"hills": len(meta["centers"]),
                                  "barrier_ev": float(fes[top] - fes[low]),
                                  "xi_min": float(grid[low]), "xi_top": float(grid[top]),
                                  "last_hill_height_ev": float(meta["heights"][-1]),
                                  "barrier_crossings": int(crossings)}
        meta_curve = (grid, fes)
    else:
        meta_curve = None
    if static is not None:
        low = int(np.argmin(np.where(static["xi"] < 0, static["energy"], np.inf)))
        top = int(np.argmax(np.where((static["xi"] > -0.5) & (static["xi"] < 0.8),
                                     static["energy"], -np.inf)))
        result["static"] = {"barrier_ev": float(static["energy"][top] - static["energy"][low]),
                            "xi_min": float(static["xi"][low]), "xi_top": float(static["xi"][top])}
    return result, {"bm": bm, "sg": sg, "free": free, "meta": meta_curve, "static": static}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)


def figures(system, result, data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bm, sg = data["bm"], data["sg"]
    figure, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURFACE)
    style(ax)
    zero_xi = bm["xi_min"] if bm else sg["forward"]["xi"][0]
    shift = interpolate(bm["xi"], bm["profile"], zero_xi) if bm else 0.0
    if data["static"] is not None:
        s = data["static"]
        e0 = s["energy"][int(np.argmin(np.abs(s["xi"] - zero_xi)))]
        ax.plot(s["xi"], s["energy"] - e0, color=MUTED, lw=1.3, ls=(0, (2, 2)),
                label=f"potential energy along the IRC (static barrier "
                      f"{result['static']['barrier_ev']:.3f} eV)")
    for direction, d in sg.items():
        ref = interpolate(d["xi"], d["profile"], zero_xi)
        ax.plot(d["xi"], d["profile"] - ref, color=COLORS[direction], lw=1.4,
                label=f"slow growth, {direction}")
    if data["meta"] is not None:
        grid, fes = data["meta"]
        fes = fes - interpolate(grid, fes, zero_xi)
        ax.plot(grid, fes, color=COLORS["metadynamics"], lw=2,
                label=f"metadynamics ({result['metadynamics']['hills']} hills)")
    if data["free"] is not None and bm is not None:
        centers, density = fe.probability_density(data["free"]["xi"], bins=60)
        keep = density > density.max() * 0.02
        curve = -KT * np.log(density[keep])
        curve -= curve.min()
        ax.plot(centers[keep], curve, color=COLORS["free_md"], lw=2,
                label="−k_B T ln P(ξ), free MD")
    if bm is not None:
        ax.fill_between(bm["xi"], bm["profile"] - shift - bm["profile_error"],
                        bm["profile"] - shift + bm["profile_error"],
                        color=COLORS["blue_moon"], alpha=0.18, lw=0)
        ax.plot(bm["xi"], bm["profile"] - shift, color=COLORS["blue_moon"], lw=2.4,
                marker="o", ms=5, mec=SURFACE, mew=1,
                label=f"blue moon + TI (barrier {result['blue_moon']['barrier_ev']:.3f} eV)")
        ax.axvline(bm["xi_star"], color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_xlabel("ξ = d(C–leaving group) − d(C–nucleophile) (Å)", color=INK2)
    ax.set_ylabel("Free energy relative to the reactant well (eV)", color=INK2)
    ax.legend(frameon=False, loc="lower left" if system == "F" else "upper right",
              labelcolor=INK2, fontsize=8.5)
    ax.set_title(f"{result['label']}: free energy at {TEMPERATURE:.0f} K", loc="left",
                 color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(out / f"{system}_free_energy.png", dpi=110, facecolor=SURFACE)
    plt.close(figure)


def main():
    results = {}
    out = WORK / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for system in SYSTEMS:
        if not folder(system).exists():
            continue
        result, data = analyze(system)
        results[system] = result
        if "--no-figures" not in sys.argv and (data["bm"] is not None or data["sg"]):
            figures(system, result, data, out)
    (WORK / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
