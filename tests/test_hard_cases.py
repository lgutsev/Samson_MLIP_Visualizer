"""Hard-case tools: exact-Hessian P-RFO, scan-to-TS, and quantum-chemistry export."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from test_reaction_path import bond, diatomic

from samson_mlip_visualizer.qm_export import (
    gaussian_input,
    orca_input,
    program_for,
    write_qm_input,
)
from samson_mlip_visualizer.reaction_path import scan_to_ts
from samson_mlip_visualizer.ts import prfo_search
from samson_mlip_visualizer.vibrations import cartesian_hessian

pytest.importorskip("sella")


def counted(atoms):
    calls = {"n": 0}
    calculate = atoms.calc.calculate

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        return calculate(*args, **kwargs)

    atoms.calc.calculate = wrapper
    return calls


def test_cartesian_hessian_of_the_bond_well():
    atoms = diatomic(2.0)
    hessian = cartesian_hessian(atoms)
    assert hessian.shape == (6, 6)
    assert np.allclose(hessian, hessian.T)
    # d²E/dr² = -16 eV/Å² at the barrier; along x the atoms move oppositely.
    assert hessian[0, 0] == pytest.approx(-16.0, rel=1e-3)
    assert hessian[0, 3] == pytest.approx(16.0, rel=1e-3)
    assert atoms.positions[1, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("options", [{"exact_hessian": True}, {"recompute_every": 2}])
def test_prfo_with_exact_hessians(options):
    atoms = diatomic(1.8)
    calls = counted(atoms)
    result = prfo_search(atoms, fmax=1e-4, **options)
    assert result.converged
    assert bond(atoms.positions) == pytest.approx(2.0, abs=1e-4)
    assert calls["n"] >= 12  # at least one 6N-call Hessian
    with pytest.raises(ValueError, match="recompute_every"):
        prfo_search(diatomic(1.8), recompute_every=0)


def test_scan_brackets_and_refines_the_barrier():
    atoms = diatomic(1.5)
    seen = []
    result = scan_to_ts(
        atoms, (0, 1), stop=2.5, points=11, on_progress=lambda *args: seen.append(args[1])
    )
    assert result.distances == pytest.approx(list(np.linspace(1.5, 2.5, 11)))
    assert seen == pytest.approx(result.distances)
    assert result.bracketed and result.distances[result.highest] == pytest.approx(2.0)
    assert result.ts.converged
    assert bond(result.ts_positions) == pytest.approx(2.0, abs=1e-3)
    assert atoms.constraints == []  # the scan constraint is removed


def test_scan_reports_an_unbracketed_maximum():
    result = scan_to_ts(diatomic(1.5), (0, 1), stop=1.9, points=5, refine=False)
    assert not result.bracketed and result.highest == 4 and result.ts is None


def test_scan_validates_input():
    with pytest.raises(ValueError, match="Invalid atom pair"):
        scan_to_ts(diatomic(1.5), (0, 0), stop=2.5)
    with pytest.raises(ValueError, match="at least 3"):
        scan_to_ts(diatomic(1.5), (0, 1), stop=2.5, points=2)


# --- quantum-chemistry export -------------------------------------------------------------


def water(shift=0.0):
    atoms = molecule("H2O")
    atoms.positions[:, 0] += shift
    return atoms


def test_gaussian_ts_and_qst_inputs():
    ts = gaussian_input([water()], job="ts", charge=0, multiplicity=1, nproc=8, memory_gb=16)
    assert "%nprocshared=8" in ts and "%mem=16GB" in ts
    assert "Opt=(TS,CalcFC,NoEigenTest) Freq" in ts
    assert "\n0 1\nO " in ts and ts.endswith("\n\n")

    qst3 = gaussian_input([water(), water(0.1), water(0.05)], job="qst3", level="M062X/def2TZVP")
    assert qst3.startswith("#p M062X/def2TZVP Opt=QST3 Freq")
    assert qst3.count("\n0 1\n") == 3
    assert "Reactant" in qst3 and "Product" in qst3 and "Transition-state guess" in qst3


def test_orca_inputs_and_side_files():
    text, extra = orca_input([water()], job="ts", nproc=4, memory_gb=8)
    assert text.startswith("! B3LYP D3BJ def2-SVP OptTS Freq")
    assert "Calc_Hess true" in text and "nprocs 4" in text and "%maxcore 2000" in text
    assert extra == {}
    text, extra = orca_input([water(), water(0.2), water(0.1)], job="qst3", stem="rxn")
    assert "NEB-TS" in text and '"rxn_product.xyz"' in text and '"rxn_ts_guess.xyz"' in text
    assert set(extra) == {"rxn_product.xyz", "rxn_ts_guess.xyz"}
    assert extra["rxn_product.xyz"].startswith("3\nproduct\nO")


def test_write_qm_input_and_validation(tmp_path):
    files = write_qm_input(tmp_path / "ts.gjf", [water()], job="irc")
    assert files == [tmp_path / "ts.gjf"] and "IRC=(CalcFC" in files[0].read_text()
    files = write_qm_input(tmp_path / "path.inp", [water(), water(0.2)], job="qst2")
    assert [file.name for file in files] == ["path.inp", "path_product.xyz"]
    assert program_for("x.com") == "gaussian"
    with pytest.raises(ValueError, match=".gjf"):
        program_for("x.txt")
    with pytest.raises(ValueError, match="needs 2"):
        gaussian_input([water()], job="qst2")
    with pytest.raises(ValueError, match="same atoms"):
        gaussian_input([water(), molecule("NH3")], job="qst2")
    periodic = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(ValueError, match="Periodic"):
        gaussian_input([periodic], job="opt")
