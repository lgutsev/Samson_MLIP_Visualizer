"""Characterization tests for the QM input writers beyond the jobs covered in test_qm_export."""

import pytest
from ase import Atoms

from samson_mlip_visualizer import qm_export


def water(shift=0.0):
    return Atoms("OH2", positions=[[0.0 + shift, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def test_program_for_by_extension():
    assert qm_export.program_for("a.gjf") == "gaussian"
    assert qm_export.program_for("A.COM") == "gaussian"
    assert qm_export.program_for("dir/b.inp") == "orca"
    with pytest.raises(ValueError, match="Use .gjf or .com"):
        qm_export.program_for("a.xyz")


def test_check_rejects_bad_jobs_counts_periodic_and_mismatched_atoms():
    with pytest.raises(ValueError, match="Unknown job"):
        qm_export.gaussian_input([water()], job="nope")
    with pytest.raises(ValueError, match="needs 2 structure"):
        qm_export.gaussian_input([water()], job="qst2")
    periodic = water()
    periodic.pbc = True
    with pytest.raises(ValueError, match="Periodic"):
        qm_export.orca_input([periodic], job="opt")
    other = Atoms("HOH", positions=water().positions)
    with pytest.raises(ValueError, match="same atoms"):
        qm_export.gaussian_input([water(), other], job="qst2")


def test_gaussian_single_structure_layout():
    text = qm_export.gaussian_input(
        [water()], job="opt", nproc=4, memory_gb=8, charge=1, multiplicity=2)
    lines = text.split("\n")
    assert lines[:2] == ["%nprocshared=4", "%mem=8GB"]
    assert lines[2] == "#p B3LYP/6-31G(d) EmpiricalDispersion=GD3BJ Opt Freq"
    assert lines[4] == "Continued from an MLIP result (SAMSON MLIP Visualizer)"
    assert lines[6] == "1 2"
    assert lines[7].split() == ["O", "0.00000000", "0.00000000", "0.00000000"]
    assert text.endswith("\n\n")  # Gaussian needs a terminating blank line


def test_gaussian_qst3_has_one_titled_section_per_structure():
    text = qm_export.gaussian_input([water(), water(0.1), water(0.2)], job="qst3", title="T")
    assert "Opt=QST3 Freq" in text
    for heading in ("Reactant (T)", "Product (T)", "Transition-state guess (T)"):
        assert heading in text
    assert text.count("0 1") == 3


def test_gaussian_level_override_and_job_routes():
    routes = {"ts": "Opt=(TS,CalcFC,NoEigenTest) Freq", "irc": "IRC=(CalcFC,MaxPoints=30)",
              "force": "Force NoSymm SCF=Tight", "energy": "SP NoSymm SCF=Tight"}
    for job, route in routes.items():
        text = qm_export.gaussian_input([water()], job=job, level="HF/STO-3G")
        assert f"#p HF/STO-3G {route}" in text


def test_orca_job_blocks():
    ts, _ = qm_export.orca_input([water()], job="ts", nproc=4, memory_gb=8)
    assert ts.startswith("! B3LYP D3BJ def2-SVP OptTS Freq\n%pal\n  nprocs 4\nend\n%maxcore 2000\n")
    assert "%geom\n  Calc_Hess true\nend" in ts and "* xyz 0 1" in ts and ts.rstrip().endswith("*")
    irc, extra = qm_export.orca_input([water()], job="irc")
    assert "%irc\n  InitHess calc_anfreq\nend" in irc and extra == {}
    force = qm_export.orca_input([water()], job="force")[0]
    energy = qm_export.orca_input([water()], job="energy")[0]
    assert force.startswith("! B3LYP D3BJ def2-SVP EnGrad TightSCF")
    assert energy.startswith("! B3LYP D3BJ def2-SVP TightSCF")


def test_orca_maxcore_without_nproc_and_truncation():
    text, _ = qm_export.orca_input([water()], job="opt", memory_gb=3)
    assert "%maxcore 3000" in text and "%pal" not in text
    text, _ = qm_export.orca_input([water()], job="opt", memory_gb=1, nproc=3)
    assert "%maxcore 333" in text  # int() truncates MB per core


def test_orca_qst_writes_extra_xyz_files():
    text, extra = qm_export.orca_input([water(), water(0.1), water(0.2)], job="qst3", stem="rxn")
    assert sorted(extra) == ["rxn_product.xyz", "rxn_ts_guess.xyz"]
    assert 'NEB_End_XYZFile "rxn_product.xyz"' in text
    assert 'NEB_TS_XYZFile "rxn_ts_guess.xyz"' in text
    assert extra["rxn_product.xyz"].splitlines()[:2] == ["3", "product"]
    _, extra2 = qm_export.orca_input([water(), water(0.1)], job="qst2", stem="rxn")
    assert sorted(extra2) == ["rxn_product.xyz"]


def test_write_qm_input_gaussian_and_orca(tmp_path):
    written = qm_export.write_qm_input(tmp_path / "a.gjf", [water()], job="opt", level="")
    assert written == [tmp_path / "a.gjf"]
    assert "B3LYP/6-31G(d)" in written[0].read_text()  # empty level falls back to the default
    written = qm_export.write_qm_input(
        tmp_path / "b.inp", [water(), water(0.1)], job="qst2", nproc=2)
    assert [p.name for p in written] == ["b.inp", "b_product.xyz"]
    assert (tmp_path / "b_product.xyz").read_text().startswith("3\nproduct\n")
    with pytest.raises(ValueError):
        qm_export.write_qm_input(tmp_path / "c.txt", [water()])
    assert not (tmp_path / "c.txt").exists()
