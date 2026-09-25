"""Hard-case tools through the CLI and the bridge."""

import pytest
from ase.io import read
from test_reaction_jobs_cli import (  # noqa: F401 - pytest fixtures
    bondwell_bridge,
    bondwell_cli,
    diatomic_model,
    write_diatomic,
)
from test_remote_dispatcher import error, result

from samson_mlip_visualizer import cli
from samson_mlip_visualizer.remote.protocol import INVALID_PARAMS

pytest.importorskip("sella")


def test_cli_scan_trajectory_and_qm_export(bondwell_cli, capsys):  # noqa: F811
    tmp_path, model = bondwell_cli
    structure = tmp_path / "r.xyz"
    write_diatomic(structure, 1.5)
    scan = tmp_path / "scan.extxyz"
    gjf = tmp_path / "ts.gjf"
    code = cli.main(
        [str(structure), str(model), "--scan", "0-1:2.5:11", "--exact-hessian",
         "--trajectory", str(scan), "--export-qm", str(gjf), "--charge", "1",
         "--multiplicity", "2"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Highest point r = 2.0000 A (bracketed)" in out
    assert len(read(scan, index=":")) == 11
    text = gjf.read_text()
    assert "Opt=(TS,CalcFC,NoEigenTest)" in text and "\n1 2\n" in text
    with pytest.raises(SystemExit, match="I-J:STOP"):
        cli.main([str(structure), str(model), "--scan", "0-1"])


def test_cli_ts_with_recomputed_hessian(bondwell_cli, capsys):  # noqa: F811
    tmp_path, model = bondwell_cli
    structure = tmp_path / "g.xyz"
    write_diatomic(structure, 1.8)
    assert cli.main(
        [str(structure), str(model), "--ts", "--recompute-hessian", "2", "--fmax", "0.001"]
    ) == 0
    assert "Finished (converged)" in capsys.readouterr().out


def test_scan_job_and_qm_export_method(bondwell_bridge, tmp_path):  # noqa: F811
    samson, dispatcher, model = bondwell_bridge(diatomic_model(1.5, "reactant"))
    job = result(
        dispatcher, "job.start", kind="scan", model=model, pair="0-1", stop=2.5, points=11
    )
    status = result(dispatcher, "job.status", id=job["id"])
    assert status["state"] == "finished", status["error"]
    scan = status["result"]
    assert scan["bracketed"] and scan["distances"][scan["highest"]] == pytest.approx(2.0)
    assert scan["ts"]["converged"] and scan["ts"]["frequencies"]["n_imaginary"] == 1

    exported = result(dispatcher, "qm.export", path=str(tmp_path / "ts.inp"), charge=0)
    assert exported["job"] == "ts" and exported["files"][0].endswith("ts.inp")
    assert "OptTS" in (tmp_path / "ts.inp").read_text()
    assert error(dispatcher, "qm.export", path=str(tmp_path / "x.txt"))["code"] == INVALID_PARAMS


def test_qm_export_qst2_from_two_models(bondwell_bridge, tmp_path):  # noqa: F811
    samson, dispatcher, model = bondwell_bridge(
        diatomic_model(1.5, "reactant"), diatomic_model(2.5, "product")
    )
    exported = result(dispatcher, "qm.export", path=str(tmp_path / "path.gjf"))
    assert exported["job"] == "qst2"
    assert "Opt=QST2" in (tmp_path / "path.gjf").read_text()
