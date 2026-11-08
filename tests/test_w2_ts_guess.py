"""W2 saddle rescue: the route-preserving Gaussian writer and the MLIP guess refinement.

The Fe2 cases are the pilot cases in ``examples/w2_ts_guess/cases`` (pairs of
UBPW91 minima from the Warehouse 2 spin ladders plus their IDPP midpoint).
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from test_reaction_path import BondWell, diatomic

from samson_mlip_visualizer.qm_export import (
    TS_OPT,
    gaussian_ts_input_from_template,
    parse_gaussian_template,
    route_tokens,
    ts_route_from_template,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "w2_ts_guess"
CASES = ("fe2h2_m9", "fe2h2o2_m7", "fe2h2o3_m7")
TEMPLATE = (EXAMPLE / "w2_ts_template_example.gjf").read_text(encoding="utf-8")
W2_ROUTE = ("#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) "
            "IOP(5/13=1) Int=UltraFine Pop=Regular")


def case(name, role="guess"):
    return read(EXAMPLE / "cases" / name / f"{role}.extxyz")


def written_route(text):
    lines = text.splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith("#"))
    end = lines.index("", start)
    return route_tokens(" ".join(lines[start:end]))


def is_opt_or_freq(token):
    return token.lower().startswith(("opt", "freq"))


@pytest.mark.parametrize("name", CASES)
def test_writer_keeps_the_template_route_token_for_token(name):
    atoms = case(name)
    charge, multiplicity = atoms.info["charge"], atoms.info["multiplicity"]
    text, changes = gaussian_ts_input_from_template(
        TEMPLATE, atoms, charge=charge, multiplicity=multiplicity, title=name,
        chk=f"{name}.ts.chk")
    template = parse_gaussian_template(TEMPLATE)
    route = written_route(text)
    # apart from Opt/Freq, only Geom=Checkpoint goes (explicit atoms follow)
    expected = [t for t in template.route if not is_opt_or_freq(t) and t != "Geom=Checkpoint"]
    assert [t for t in route if not is_opt_or_freq(t)] == expected
    assert [t for t in route if is_opt_or_freq(t)] == [TS_OPT, "Freq"]
    assert "IOP(5/13=1)" in route and "Int=UltraFine" in route and "Pop=Regular" in route
    assert "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)" in route
    assert "geom" not in text.lower()
    lines = text.splitlines()
    assert lines[:4] == [f"%chk={name}.ts.chk", "%oldchk=failed_saddle.chk", "%mem=24GB",
                         "%nprocshared=20"]
    spec = lines.index(f"{charge} {multiplicity}")
    coordinates = lines[spec + 1: spec + 1 + len(atoms)]
    assert [line.split()[0] for line in coordinates] == atoms.get_chemical_symbols()
    xyz = np.array([[float(v) for v in line.split()[1:]] for line in coordinates])
    assert xyz == pytest.approx(atoms.positions, abs=1e-7)
    assert lines[spec + 1 + len(atoms)] == ""  # blank line ends the molecule section
    assert any("Geom=Checkpoint" in c for c in changes)


def test_writer_adds_opt_and_freq_and_keeps_a_combined_iop():
    template = ("%nprocshared=12\n%mem=24GB\n"
                "#p UBPW91/6-311++G* SCF=(VShift=5, NoIncFock,\n"
                " MaxCyc=200,Tight,NoVarAcc) NoSymm IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine\n"
                "\nsome W2 job\n\n0 7\nFe 0 0 1\nFe 0 0 -1\n\n--Link1--\n%chk=x\n#p Freq\n")
    parsed = parse_gaussian_template(template)
    assert (parsed.charge, parsed.multiplicity, parsed.extra_jobs) == (0, 7, 1)
    atoms = case("fe2h2_m9", "reactant")
    text, changes = gaussian_ts_input_from_template(parsed, atoms, charge=0, multiplicity=9,
                                                    title="t")
    route = written_route(text)
    assert route == ["#p", "UBPW91/6-311++G*", TS_OPT, "Freq",
                     "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)", "NoSymm",
                     "IOP(5/13=1,5/36=1,8/11=1)", "Int=UltraFine"]
    assert text.splitlines()[:2] == ["%nprocshared=12", "%mem=24GB"]
    assert "added Freq" in changes and any("Link1" in c for c in changes)


def test_writer_keeps_an_existing_freq_variant_and_harmless_geom_options():
    tokens = route_tokens("#P UBPW91/6-311++G* opt(calcall,ts) Freq=NoRaman "
                          "Geom=(AllCheck,Connectivity) iop(5/13=1)")
    route, changes = ts_route_from_template(tokens)
    assert route == ["#P", "UBPW91/6-311++G*", TS_OPT, "Freq=NoRaman", "Geom=connectivity",
                     "iop(5/13=1)"]
    assert len(changes) == 2


@pytest.mark.parametrize("route", [
    "#p UBPW91/6-311++G* Opt=TS Freq SCF=Tight Int=UltraFine",
    "#p UBPW91/6-311++G* Opt=TS IOP(5/36=1) Int=UltraFine",
    "#p UBPW91/6-311++G* Opt=TS IOP(5/130=1) Int=UltraFine",
])
def test_writer_refuses_a_template_without_iop_5_13(route):
    with pytest.raises(ValueError, match="IOP"):
        gaussian_ts_input_from_template(route + "\n\nt\n\n0 7\n", case("fe2h2_m9"),
                                        charge=0, multiplicity=9, title="t")


def test_writer_refuses_periodic_structures():
    atoms = case("fe2h2_m9")
    atoms.pbc = True
    atoms.cell = [10, 10, 10]
    with pytest.raises(ValueError, match="Periodic"):
        gaussian_ts_input_from_template(TEMPLATE, atoms, charge=0, multiplicity=9, title="t")


# --- MLIP refinement ------------------------------------------------------------

pytest.importorskip("sella")


class RecordingWell(BondWell):
    """BondWell that records the spin state each structure carries."""

    seen: set

    def calculate(self, atoms=None, properties=None, system_changes=None):
        RecordingWell.seen.add((atoms.info.get("charge"), atoms.info.get("spin")))
        super().calculate(atoms, properties, system_changes or ["positions"])


def tagged(r):
    atoms = diatomic(r)
    atoms.info.update(charge=0, spin=3)
    return atoms


def test_refine_guess_prfo_with_irc_barriers():
    from samson_mlip_visualizer.ts_guess import refine_guess

    RecordingWell.seen = set()
    result = refine_guess(RecordingWell(), tagged(1.85))
    assert result.ok and result.method == "prfo" and result.n_imaginary == 1
    assert np.linalg.norm(result.positions[0] - result.positions[1]) == pytest.approx(2.0, abs=1e-3)
    assert result.barrier_forward_ev == pytest.approx(1.0, abs=1e-3)
    assert result.barrier_reverse_ev == pytest.approx(1.0, abs=1e-3)
    assert result.rmsd_from_start == pytest.approx(0.075, abs=2e-3)  # 0.15 Å stretch / 2 atoms
    assert RecordingWell.seen == {(0, 3)}  # every evaluation saw the frame's charge/spin
    assert result.summary()["ok"] and "positions" not in result.summary()


@pytest.mark.parametrize("strategy,method", [("auto", "prfo"), ("qst", "qst3")])
def test_refine_guess_with_endpoints(strategy, method):
    from samson_mlip_visualizer.ts_guess import refine_guess

    result = refine_guess(BondWell(), tagged(1.85), reactant=tagged(1.45),
                          product=tagged(2.6), strategy=strategy)
    assert result.method == method and result.ok and result.irc_connects_endpoints
    assert result.barrier_forward_ev == pytest.approx(1.0, abs=1e-3)
    assert set(result.endpoint_rmsd) == {"reactant", "product"}


def test_refine_guess_qst2_and_argument_checks():
    from samson_mlip_visualizer.ts_guess import refine_guess

    result = refine_guess(BondWell(), reactant=tagged(1.5), product=tagged(2.5), run_irc=False)
    assert result.method == "qst2" and result.ok and result.rmsd_from_start is None
    with pytest.raises(ValueError, match="both reactant and product"):
        refine_guess(BondWell(), tagged(1.85), reactant=tagged(1.5))
    with pytest.raises(ValueError, match="strategy"):
        refine_guess(BondWell(), tagged(1.85), strategy="dimer")


def test_cli_refuses_a_template_without_iop_before_loading_a_model(tmp_path):
    spec = importlib.util.spec_from_file_location("refine_ts_guess",
                                                  EXAMPLE / "refine_ts_guess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = tmp_path / "no_iop.gjf"
    template.write_text("#p UBPW91/6-311++G* Opt=TS Freq\n\nt\n\n0 9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="IOP"):
        module.main([str(EXAMPLE / "cases" / "fe2h2_m9" / "guess.extxyz"),
                     "--route-from", str(template), "--out", str(tmp_path / "out"),
                     "--model", str(tmp_path / "missing.model")])


FE2_MODEL = Path(os.environ.get(
    "FE2_SPIN_MODEL", r"D:\MLIP_Work_Folder\fe_spin_ladder\fe2_spin\seed1\fe2_spin_seed1.model"))


@pytest.mark.skipif(not FE2_MODEL.is_file(), reason="Fe2 spin-MACE not on this machine")
def test_fe2h2o3_hydrogen_hop_on_the_fe2_spin_mace():
    """The one pilot case where the MLIP saddle connects the two W2 minima."""
    pytest.importorskip("mace")
    from mace.calculators import MACECalculator

    from samson_mlip_visualizer.ts_guess import refine_guess

    calc = MACECalculator(model_paths=str(FE2_MODEL), device="cpu", default_dtype="float64")
    structures = []
    for role in ("guess", "reactant", "product"):
        atoms = case("fe2h2o3_m7", role)
        atoms.info.update(spin=atoms.info["multiplicity"], total_spin=atoms.info["multiplicity"])
        structures.append(atoms)
    result = refine_guess(calc, structures[0], reactant=structures[1], product=structures[2])
    assert result.ok and result.method == "prfo" and result.irc_connects_endpoints
    assert result.imaginary_cm[0] == pytest.approx(-520, abs=30)
    assert 0.0 < result.barrier_reverse_ev < result.barrier_forward_ev < 1.0
    assert result.rmsd_from_start < 0.4
