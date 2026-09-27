# SN2 free energies: constrained MD, blue moon, and metadynamics

This example redoes part 3 of the VASP molecular-dynamics tutorial,
[free-energy methods for Cl⁻ + CH₃Cl](https://vasp.at/tutorials/latest/md/part3/),
with the tools in this repository. The main system is F⁻ + CH₃Cl → CH₃F + Cl⁻,
driven by the AIMNet2 model fine-tuned on ωB97X-D in [`../sn2_f_ch3cl`](../sn2_f_ch3cl).
The tutorial's own identity reaction, Cl⁻ + CH₃Cl, is run with GFN2-xTB for
comparison. All numbers come from runs on 2026-09-27 on a laptop (Intel
i9-14900HX), using one CPU thread per simulation.

![Free-energy profiles of F⁻ + CH₃Cl at 300 K](images/F_free_energy.png)

*F⁻ + CH₃Cl, fine-tuned AIMNet2, 300 K. Every method is zeroed at the reactant
well. The blue band is the propagated statistical error of the thermodynamic
integration. The dashed vertical line marks ξ*, where the blue-moon mean force
crosses zero. The grey dashes show the potential energy along the model's IRC.*

## Conclusions

- **The free-energy barrier is about twice the static one.**
  - For F⁻ + CH₃Cl, blue-moon thermodynamic integration gives
    ΔA‡ = 0.268 ± 0.01 eV (6.2 kcal/mol) at 300 K.
  - The potential-energy barrier from the ion–dipole complex is 0.141 eV
    (3.25 kcal/mol, the fine-tuned value that matches CCSD(T)).
  - Almost all of the difference is entropy. In the reactant well, F⁻ roams
    over a wide range of ξ (−1.7 to −0.5 Å in 20 ps of free MD), and the free
    minimum sits at ξ = −0.91 Å, not at the static complex (−0.65 Å). At the
    TS, the three heavy atoms are locked in a line.
- **The independent estimates agree.**
  - Well-tempered metadynamics gives 0.33 eV. That is within its own run-to-run drift
    (about ±0.04 eV over the last 200 hills) of the blue-moon value.
  - The reactant well of −k_BT ln P(ξ) from free MD lies on top of the
    blue-moon profile.
  - The rate formula gives k = 3.5 × 10⁸ s⁻¹ from the complex and a
    phenomenological barrier of 0.253 eV (5.8 kcal/mol).
- **Slow growth, as the tutorial warns, is not converged** at the tutorial's
  speed (1 mÅ per 2 fs step).
  - Forward it overshoots (0.351 eV); reverse it undershoots (0.215 eV).
  - The two branches differ by up to 0.25 eV.
  - They bracket the blue-moon profile but are no substitute for it.
- **Cl⁻ + CH₃Cl with GFN2-xTB reproduces the tutorial's workflow and its internal
  consistency.**
  - The generalized velocity at the TS matches the tutorial to three digits:
    7.85 against 7.84 × 10¹² Å/s.
  - The phenomenological barrier equals the free-energy difference, as in the
    tutorial.
  - The barrier itself comes out higher: 0.51 eV against the tutorial's
    0.41 eV. That difference comes from the potential, not the method.
- **The numbers are only as good as the model away from its training path.**
  - The fine-tuned AIMNet2 was checked to about 1 kcal/mol a small step off
    the IRC.
  - At 300 K the MD samples much further from it than that, above all the
    roaming F⁻ in the reactant well.
  - That off-path error is not measured here. The
    [active-learning loop](../sn2_f_ch3cl/README.md#active-learning-with-the-library-loop) run
    on these MD frames would be the way to check it.
  - The dynamics is classical (no zero-point energy or tunnelling), as in the
    tutorial.

## Method

The reaction coordinate is the tutorial's: ξ = d(C–X_leaving) − d(C–X_nucleophile).
It is negative in the reactant complex, near zero at the Walden TS, and positive
on the product side. Everything is in
[`samson_mlip_visualizer.free_energy`](../../src/samson_mlip_visualizer/free_energy.py),
which is ASE-based and works with any calculator:

| Tutorial (VASP)                                   | Here                                                                                                                  |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `ICONST` … `R` combination of distances           | `DistanceCombination([(0, 4, 1.0), (0, 5, -1.0)])`, with ξ, ∇ξ, Z and the blue-moon G term                            |
| SHAKE in the Andersen NVT MD (`MDALGO = 1`)       | `constrained_md`: velocity Verlet with SHAKE and RATTLE, Andersen thermostat, constraint multiplier λ recorded each step |
| slow growth, `INCREM`                             | `constrained_md(increment=…)` and `slow_growth_profile`: ∫λ dξ along the moving target                                |
| blue moon, `REPORT` output                        | `blue_moon_gradient`: dA/dξ = ⟨Z^-½(λ + k_BT·G)⟩ / ⟨Z^-½⟩, with block-averaged errors; `integrate_gradient`            |
| free MD, P(ξ)                                     | ASE `Andersen` and `probability_density`                                                                              |
| ⟨\|ξ̇*\|⟩ = √(2k_BT/π) / ⟨Z^-½⟩ at ξ*             | `generalized_velocity`                                                                                                |
| k = ½⟨\|ξ̇*\|⟩ P(ξ_ref) e^(−ΔA/k_BT), ΔA‡          | `rate_constant`                                                                                                       |
| (not in the tutorial)                             | `MetadynamicsCalculator` (well-tempered Gaussian bias on ξ, harmonic walls) and `metadynamics`, `fes_from_hills`       |

The sign convention matters when comparing with the tutorial. The constraint
force is +λ∇ξ, and dA/dξ is the average of λ plus the Z and G corrections. The
unit tests check this, the G term, and the whole estimator against a
three-atom model whose A(ξ) is known exactly by quadrature
([`tests/test_free_energy.py`](../../tests/test_free_energy.py)).

Settings as in the tutorial:
- 300 K, 2 fs step, hydrogen given tritium mass (3.0);
- Andersen collision probability 0.05 per step.

For F⁻ + CH₃Cl the runs were:

| Run                    | What                                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| blue moon              | 20 windows from ξ = −1.5 to +2.0 Å (0.25 Å apart, 0.125 Å through the well and barrier), 3000 steps each, first 500 dropped |
| slow growth            | 500 steps held at the start, then ξ moved over 3.5 Å at 1 mÅ per step, forward and reverse                            |
| free MD                | 10 000 steps from the complex, with a wall keeping F⁻ within 5 Å of carbon                                            |
| constrained run at ξ*  | 3000 steps, for ⟨\|ξ̇*\|⟩                                                                                             |
| metadynamics           | well-tempered (bias factor 10), 20 meV × 0.08 Å hills every 50 steps, 25 000 steps (50 ps), walls at ξ = −1.6 and +0.5 Å |

Windows start from the fine-tuned model's IRC frame nearest their target. The
last stretch is made chemically: the nucleophile or the leaving group is pulled
along its C–X axis. Projecting straight onto ξ along the mass-weighted gradient
instead pulls carbon away from its hydrogens; one early window started that
way reached 1800 K.

The static curve is the fine-tuned model's potential energy along its own IRC.

## Results: F⁻ + CH₃Cl (fine-tuned AIMNet2, 300 K)

| Quantity                                   | Value                                                      |
| ------------------------------------------ | ---------------------------------------------------------- |
| reactant free-energy minimum ξ_min          | −0.91 Å (blue moon), −0.89 Å (peak of P(ξ))                |
| ξ* (top of the barrier)                     | +0.08 Å                                                    |
| **ΔA‡, blue moon + TI**                     | **0.268 eV (6.19 kcal/mol)**, ± 0.01 eV                    |
| ΔA‡, well-tempered metadynamics             | 0.33 eV (0.24–0.33 over the run), 42 barrier crossings      |
| ΔA‡, slow growth forward / reverse          | 0.351 / 0.215 eV (hysteresis up to 0.25 eV)                |
| ΔE‡, static, from the complex               | 0.141 eV (3.25 kcal/mol)                                   |
| ΔA, well → ξ = 2.0 Å (products)             | −1.09 ± 0.01 eV                                            |
| P(ξ_ref), ξ_ref = −0.89 Å                   | 2.91 Å⁻¹                                                   |
| ⟨\|ξ̇*\|⟩                                    | 8.09 × 10¹² Å/s (tutorial, Cl system: 7.84 × 10¹²)         |
| rate constant k                             | 3.5 × 10⁸ s⁻¹                                              |
| phenomenological ΔA‡                        | 0.253 eV (5.83 kcal/mol)                                   |

The metadynamics run needed one correction. With the upper wall at ξ = +1.0 Å,
the walker crossed the barrier once, after 12 ps, and never came back: the
product side is 1 eV downhill, and 50 ps of 20 meV hills could not fill it. Its
barrier (0.31 eV) therefore rested on a single crossing. With the wall at
+0.5 Å, just past ξ*, it went from the well to beyond the TS and back 42 times.

The barrier read from the accumulated bias still moves:
- 0.29 eV after 100 hills;
- 0.24 eV after 200;
- 0.33 eV after 300;
- 0.32 eV after 400;
- 0.33 eV after 500.

The final hills are 3 meV high.

Metadynamics is the cheapest route to the whole profile, but here it does not
pin the barrier down to better than about 0.05 eV. The ~0.06 eV it sits above
the blue-moon value is within that. A longer run, or several walkers, would
settle it. The blue-moon integration, 20 independent windows with per-window
error bars, is the number to quote.

## Comparison: Cl⁻ + CH₃Cl (GFN2-xTB), the tutorial's system

![Free-energy profiles of Cl⁻ + CH₃Cl at 300 K](images/Cl_free_energy.png)

*Cl⁻ + CH₃Cl, GFN2-xTB, 300 K. The blue-moon windows cover ξ ≤ 0 and are
mirrored, since A(ξ) = A(−ξ) for an identity reaction.*

Runs:
- slow growth both ways, over ξ = −1.5 … +1.5 Å;
- 10 blue-moon windows on ξ ≤ 0, 3000 steps each;
- 10 000 steps of free MD;
- a constrained run at ξ* = 0.

All settings are as for F⁻. The reverse slow growth starts from the mirror
image of the forward start, with the two chlorines swapped.

| Quantity                                | GFN2-xTB (here)                                  | VASP tutorial (MLFF)        |
| --------------------------------------- | ------------------------------------------------ | --------------------------- |
| reactant free-energy minimum             | ξ = −1.41 Å (blue moon), −1.47 Å (peak of P(ξ))  | ξ_ref = −1.5 Å              |
| **ΔA‡, blue moon + TI**                  | **0.512 eV (11.8 kcal/mol)**, ± 0.01 eV          | literature blue moon 0.418 eV |
| ΔA‡, slow growth forward / reverse       | 0.64 / 0.55 eV, each from its own start (hysteresis up to 0.21 eV) | 0.406 eV (forward) |
| ΔE‡, static, from the complex (ξ = −1.17 Å) | 0.460 eV (10.6 kcal/mol)                     | —                           |
| P(ξ_ref)                                 | 2.30 Å⁻¹                                         | 1.54 Å⁻¹                    |
| ⟨\|ξ̇*\|⟩                                 | 7.85 × 10¹² Å/s                                  | 7.84 × 10¹² Å/s             |
| rate constant k                          | 2.2 × 10⁴ s⁻¹                                    | 9.2 × 10⁵ s⁻¹               |
| phenomenological ΔA‡                     | 0.504 eV (11.6 kcal/mol)                         | 0.407 eV                    |

What agrees:
- **The whole chain of methods.** The phenomenological barrier matches the
  free-energy difference, 0.504 against 0.513 eV from ξ_ref, just as the
  tutorial's 0.407 matches its 0.406. The generalized velocity agrees to three
  digits. The rate constants differ by a factor of 40, which is
  exp(−Δ(ΔA‡)/k_BT) for the 0.1 eV gap in barriers: the gap in k is the gap in
  the barrier and nothing else.
- **The reverse slow growth.** It lies on the blue-moon profile and ends within
  0.04 eV of zero, the right answer for an identity reaction.

What differs:
- **The forward slow growth** overshoots by 0.13 eV and ends 0.17 eV above its
  start.
- **The barrier** is higher than the tutorial's. The tutorial does not say
  which functional its force field was trained on, and GGA functionals are
  known to underestimate SN2 barriers. So the 0.1 eV gap says more about the
  reference potentials than about xTB. For F⁻, where the model was fitted to
  CCSD(T)-quality data, the static barrier is right.

Entropy raises the Cl⁻ barrier much less than the F⁻ one: 0.05 eV above the
static value here, against 0.13 eV for F⁻. In free MD the chloride drifts out to
the 5 Å wall (the flat stretch of −k_BT ln P(ξ) below ξ = −2 Å). Most of that
extra room lies beyond the part of the well that sets the barrier.

## Reproducing

Use the aimnet environment's Python, which has ASE and AIMNet2; `xtb` must be
findable for the Cl system. Outputs go to `D:\MLIP_Work_Folder\sn2_free_energy`,
or to `SN2FE_WORK`.

```bash
python examples/sn2_free_energy/launch.py main
python examples/sn2_free_energy/analyze.py
python examples/sn2_free_energy/launch.py ts
python examples/sn2_free_energy/analyze.py
```

- `launch.py main` runs every task in parallel, one thread each, and skips
  tasks that are already done. `launch.py main F` or `launch.py main Cl`
  limits it to one system.
- The first `analyze.py` finds ξ*.
- `launch.py ts` runs the constrained simulation at ξ*.
- The second `analyze.py` adds the rate, then writes `results.json` and the
  figures.

The F⁻ set is about 110 000 MD steps at roughly 40 ms per step. On the laptop
it ran in about half an hour of wall time; the 50 ps metadynamics run is the
longest. xTB costs about 180 ms per step, so the Cl⁻ set takes about 35 minutes
in parallel.

| File                                    | Role                                                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [`common.py`](common.py)                | systems, masses, coordinate, calculators, starting structures, window grid                             |
| [`run.py`](run.py)                      | one task: `slow_growth forward\|reverse`, `window XI`, `free_md`, `ts_velocity XI`, `metadynamics`      |
| [`launch.py`](launch.py)                | runs the tasks in parallel                                                                             |
| [`analyze.py`](analyze.py)              | integration, errors, hysteresis, P(ξ), rate, metadynamics free energy, figures                         |
