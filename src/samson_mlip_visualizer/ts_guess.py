"""Refine a transition-state guess on an MLIP before handing it back to DFT.

The rescue loop for failed quantum-chemistry saddle searches: take the best
index-1 frame of the failed run (and, when known, the two minima it should
connect), refine it on a cheap MLIP surface, check that the result has exactly
one imaginary mode there, and return it as the starting geometry of a new DFT
TS search (:func:`qm_export.gaussian_ts_input_from_template`). The MLIP is only
a guess generator: the DFT frequency run decides whether the saddle is real.

Two routes, both from the existing machinery:

- one frame: :func:`ts.prfo_search` with an exact initial Hessian (Gaussian's
  ``CalcFC``), then optionally an :func:`reaction_path.irc` on the MLIP to find
  the minima the saddle connects and the barrier from each side;
- frame plus endpoints (or endpoints alone): :func:`reaction_path.qst`, i.e.
  IDPP, CI-NEB and P-RFO, with the frame as the QST3 middle image when given.

The atoms' ``info`` (charge, multiplicity) is copied to every structure the MLIP
evaluates, so a charge/spin-aware model sees the right state throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from ase import Atoms

from .engine import relax
from .reaction_path import aligned_rmsd, irc, qst
from .ts import prfo_search
from .vibrations import harmonic_frequencies


@dataclass
class GuessResult:
    """What the MLIP made of one guess. Energies in eV, wavenumbers in cm^-1."""

    method: str  # "prfo" or "qst2"/"qst3"
    converged: bool
    steps: int
    energy_ev: float
    max_force_ev_per_angstrom: float
    n_imaginary: int
    imaginary_cm: list[float]
    lowest_real_cm: list[float]
    rmsd_from_start: float | None  # Å, after superposition; None without a start frame
    barrier_forward_ev: float | None = None  # E(TS) - E(reactant side)
    barrier_reverse_ev: float | None = None  # E(TS) - E(product side)
    endpoint_rmsd: dict[str, float] = field(default_factory=dict)
    irc_connects_endpoints: bool | None = None
    notes: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)  # earlier, rejected refinements
    positions: np.ndarray | None = None

    @property
    def one_imaginary(self) -> bool:
        return self.n_imaginary == 1

    @property
    def saddle(self) -> bool:
        """Converged on the MLIP to a first-order saddle."""
        return self.converged and self.one_imaginary

    @property
    def ok(self) -> bool:
        """A first-order saddle whose MLIP IRC reaches the given endpoints (if any):
        worth a DFT TS search."""
        return self.saddle and self.irc_connects_endpoints is not False

    def summary(self) -> dict:
        data = asdict(self)
        data.pop("positions")
        data.update(ok=self.ok, saddle=self.saddle, one_imaginary=self.one_imaginary)
        return data


def _on(calc, source: Atoms, positions=None) -> Atoms:
    atoms = source.copy()  # keeps info (charge, multiplicity) for spin-aware models
    if positions is not None:
        atoms.set_positions(positions)
    atoms.calc = calc
    return atoms


def refine_guess(
    calc,
    guess: Atoms | None = None,
    *,
    reactant: Atoms | None = None,
    product: Atoms | None = None,
    fmax: float = 0.01,
    max_steps: int = 300,
    relax_endpoints: bool = True,
    endpoint_fmax: float = 0.01,
    images: int = 7,
    run_irc: bool = True,
    strategy: str = "auto",
    imaginary_threshold_cm: float | None = None,
) -> GuessResult:
    """Refine ``guess`` (and/or the ``reactant``/``product`` pair) to a saddle on ``calc``.

    With only ``guess``: P-RFO from it. With only the endpoints: QST2. With all
    three, ``strategy`` decides: ``"auto"`` (default) runs P-RFO from the frame
    and falls back to QST3 when that does not give one imaginary mode with an MLIP
    IRC reaching both endpoints (the failed attempt is kept in ``attempts``);
    ``"prfo"`` and ``"qst"`` run only that method. Endpoints are relaxed on the MLIP first unless
    ``relax_endpoints`` is false (they should be MLIP minima for the barrier to
    mean anything). ``run_irc`` follows the IRC from the result on the MLIP: the
    barriers then come from the minima it reaches, and with endpoints it checks
    that those are the given minima (0.25 Å RMSD).
    """
    if strategy not in ("auto", "prfo", "qst"):
        raise ValueError("strategy must be 'auto', 'prfo' or 'qst'")
    if guess is None and (reactant is None or product is None):
        raise ValueError("Give a guess frame, a reactant/product pair, or both")
    if (reactant is None) != (product is None):
        raise ValueError("Give both reactant and product, or neither")
    reference = guess if guess is not None else reactant
    for other in (reactant, product, guess):
        if other is not None and other.get_chemical_symbols() != reference.get_chemical_symbols():
            raise ValueError("Guess, reactant and product must list the same atoms in the same "
                             "order")
    notes: list[str] = []
    ends: dict[str, Atoms] = {}
    end_energy: dict[str, float] = {}
    if reactant is not None:
        for name, source in (("reactant", reactant), ("product", product)):
            atoms = _on(calc, source)
            if relax_endpoints:
                result = relax(atoms, fmax=endpoint_fmax, max_steps=2000, optimizer="LBFGS",
                               min_distance=None)
                if not result.converged:
                    notes.append(f"{name} relaxation did not reach fmax {endpoint_fmax}")
            ends[name] = atoms
            end_energy[name] = float(atoms.get_potential_energy())
        if aligned_rmsd(ends["reactant"].positions, ends["product"].positions) < 0.05:
            notes.append("reactant and product relaxed to the same minimum on the MLIP")

    attempts: list[GuessResult] = []
    use_prfo = guess is not None and (not ends or strategy in ("auto", "prfo"))
    if use_prfo:
        ts = _on(calc, guess)
        ts_result = prfo_search(ts, fmax=fmax, max_steps=max_steps, exact_hessian=True,
                                min_distance=None)
        result = _characterize(calc, reference, ts, ts_result, "prfo", guess, ends, end_energy,
                               run_irc, imaginary_threshold_cm, list(notes))
        if not ends or strategy == "prfo" or result.ok:
            return result
        attempts.append(result)
        notes.append("P-RFO from the frame gave "
                     + ("a saddle that does not connect the endpoints" if result.saddle
                        else f"{result.n_imaginary} imaginary modes"
                        + ("" if result.converged else ", not converged"))
                     + "; tried QST3")
    path = qst(ends["reactant"], ends["product"], calc,
               guess=_on(calc, guess) if guess is not None else None,
               images=images, max_steps=max(max_steps, 500), refine=True, refine_fmax=fmax)
    if not path.neb_converged:
        notes.append(f"CI-NEB stopped before fmax 0.05 after {path.neb_steps} steps")
    ts = _on(calc, reference, path.ts_positions)
    result = _characterize(calc, reference, ts, path.ts, "qst3" if guess is not None else "qst2",
                           guess, ends, end_energy, run_irc, imaginary_threshold_cm, notes)
    if attempts and not result.ok and attempts[0].saddle:
        # neither passes; keep the P-RFO saddle, which stays closest to the DFT frame
        best, rejected = attempts[0], result
        best.notes.append(f"QST3 did not pass either ({rejected.n_imaginary} imaginary modes, "
                          f"IRC connects: {rejected.irc_connects_endpoints}); kept P-RFO")
    else:
        best, rejected = result, (attempts[0] if attempts else None)
    if rejected is not None:
        best.attempts = [rejected.summary()]
    return best


def _characterize(calc, reference, ts, ts_result, method, guess, ends, end_energy, run_irc,
                  imaginary_threshold_cm, notes) -> GuessResult:
    """Frequencies, barriers, RMSDs and (optionally) the MLIP IRC of a refined saddle."""
    frequency_options = {}
    if imaginary_threshold_cm is not None:
        frequency_options["imaginary_threshold_cm"] = imaginary_threshold_cm
    frequencies = harmonic_frequencies(ts, **frequency_options)
    hint = frequencies.soft_mode_hint()
    if hint:
        notes.append(hint)
    wavenumbers = frequencies.wavenumbers_cm
    real = wavenumbers[wavenumbers >= -frequencies.imaginary_threshold_cm]
    energy = float(ts.get_potential_energy())
    result = GuessResult(
        method=method,
        converged=bool(ts_result.converged),
        steps=int(ts_result.steps),
        energy_ev=energy,
        max_force_ev_per_angstrom=float(ts_result.evaluation.max_force_ev_per_angstrom),
        n_imaginary=frequencies.n_imaginary,
        imaginary_cm=[round(float(w), 1) for w in frequencies.imaginary],
        lowest_real_cm=[round(float(w), 1) for w in real[:3]],
        rmsd_from_start=(aligned_rmsd(guess.positions, ts.positions)
                         if guess is not None else None),
        notes=notes,
        positions=ts.get_positions().copy(),
    )
    for name, atoms in ends.items():
        result.endpoint_rmsd[name] = round(aligned_rmsd(atoms.positions, ts.positions), 4)
    if ends:
        result.barrier_forward_ev = energy - end_energy["reactant"]
        result.barrier_reverse_ev = energy - end_energy["product"]
    if not run_irc or frequencies.n_imaginary < 1:
        return result

    path = irc(_on(calc, reference, ts.positions), relax_ends=True)
    sides = [(path.reverse_minimum_ev, path.reverse_minimum_positions),
             (path.forward_minimum_ev, path.forward_minimum_positions)]
    if ends:
        def distance(side, name):
            return aligned_rmsd(side[1], ends[name].positions)

        # orient the IRC so that its first side is the reactant's
        if distance(sides[0], "reactant") + distance(sides[1], "product") >                 distance(sides[1], "reactant") + distance(sides[0], "product"):
            sides.reverse()
        gaps = [distance(sides[0], "reactant"), distance(sides[1], "product")]
        result.irc_connects_endpoints = bool(max(gaps) < 0.25)
        if not result.irc_connects_endpoints:
            notes.append("the MLIP IRC does not reach the given endpoints (RMSD "
                         f"{gaps[0]:.2f} / {gaps[1]:.2f} Å): another saddle")
    else:
        result.barrier_forward_ev = energy - sides[0][0]
        result.barrier_reverse_ev = energy - sides[1][0]
        if aligned_rmsd(sides[0][1], sides[1][1]) < 0.05:
            notes.append("both IRC branches relax to the same minimum")
    return result
