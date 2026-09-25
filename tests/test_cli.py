import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from samson_mlip_visualizer import cli
from samson_mlip_visualizer.sanity import StructureSanityError


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": -1.5, "forces": np.zeros((len(atoms), 3))}


class CommitteeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        self.results = {
            "energy": -1.5,
            "forces": np.zeros((n, 3)),
            "energy_comm": np.array([-1.4, -1.5, -1.6]),
            "forces_comm": np.zeros((3, n, 3)),
        }


def _write_structure(path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    write(path, atoms)


def test_cli_single_point(monkeypatch, tmp_path, capsys):
    structure = tmp_path / "h2.xyz"
    model = tmp_path / "model.model"
    _write_structure(structure)
    model.write_bytes(b"x")

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: ConstantCalculator())

    exit_code = cli.main([str(structure), str(model), "--backend", "mace"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Energy: -1.5000000000 eV" in out
    assert "Max force: 0.000000 eV/A" in out


def test_cli_relax_writes_output(monkeypatch, tmp_path, capsys):
    structure = tmp_path / "h2.xyz"
    model = tmp_path / "model.pb"
    output = tmp_path / "relaxed.xyz"
    _write_structure(structure)
    model.write_bytes(b"x")

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: ConstantCalculator())

    exit_code = cli.main(
        [str(structure), str(model), "--backend", "deepmd", "--relax", "-o", str(output)]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert output.is_file()
    assert "Finished (converged)" in out


def test_cli_reports_committee_uncertainty(monkeypatch, tmp_path, capsys):
    structure = tmp_path / "h2.xyz"
    _write_structure(structure)
    models = [tmp_path / "a.model", tmp_path / "b.model", tmp_path / "c.model"]
    for model in models:
        model.write_bytes(b"x")

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: CommitteeCalculator())

    exit_code = cli.main([str(structure), *[str(m) for m in models]])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "committee:      3 models" in out
    assert "Final committee energy std" in out


def test_cli_aborts_on_overlapping_input(monkeypatch, tmp_path):
    structure = tmp_path / "bad.xyz"
    model = tmp_path / "model.model"
    write(structure, Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.2]]))
    model.write_bytes(b"x")

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: ConstantCalculator())

    with pytest.raises(StructureSanityError, match="unphysical forces"):
        cli.main([str(structure), str(model)])


def test_cli_passes_optimizer_choice(monkeypatch, tmp_path, capsys):
    structure = tmp_path / "h2.xyz"
    model = tmp_path / "model.model"
    _write_structure(structure)
    model.write_bytes(b"x")

    seen = {}
    real_relax = cli.relax

    def spy_relax(atoms, **kwargs):
        seen.update(kwargs)
        return real_relax(atoms, **kwargs)

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: ConstantCalculator())
    monkeypatch.setattr(cli, "relax", spy_relax)

    cli.main([str(structure), str(model), "--relax", "--optimizer", "LBFGS"])
    assert seen["optimizer"] == "LBFGS"


def test_cli_md_with_constraint(monkeypatch, tmp_path, capsys):
    structure = tmp_path / "h2.xyz"
    model = tmp_path / "model.model"
    trajectory = tmp_path / "md.extxyz"
    _write_structure(structure)
    model.write_bytes(b"x")

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: ConstantCalculator())

    exit_code = cli.main(
        [
            str(structure),
            str(model),
            "--md",
            "--md-steps",
            "20",
            "--seed",
            "0",
            "--fix-distance",
            "0-1:0.8",
            "--report-interval",
            "10",
            "--trajectory",
            str(trajectory),
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Langevin MD: 20 steps x 0.5 fs at 300 K" in out
    assert "Finished after 20 steps (10 fs)" in out
    assert "Constraint 0-1 at 0.8000 A: mean force +0.00000" in out
    assert trajectory.is_file()


def test_cli_ts_and_frequencies(monkeypatch, tmp_path, capsys):
    from test_ts import DoubleWell

    structure = tmp_path / "h.xyz"
    model = tmp_path / "model.model"
    write(structure, Atoms("H", positions=[[0.4, 0.2, -0.1]]))
    model.write_bytes(b"x")

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: DoubleWell())

    # A lone atom in an external field has no vibrational modes, so start at random.
    exit_code = cli.main(
        [str(structure), str(model), "--ts", "--ts-start", "random"]
        + ["--fmax", "0.001", "--seed", "0"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Finished (converged)" in out
    assert "curvature -4.0000" in out


def test_cli_relax_then_frequencies(monkeypatch, tmp_path, capsys):
    from ase.calculators.emt import EMT

    structure = tmp_path / "cu2.xyz"
    model = tmp_path / "model.model"
    write(structure, Atoms("Cu2", positions=[[0, 0, 0], [2.3, 0, 0]]))
    model.write_bytes(b"x")

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: EMT())

    exit_code = cli.main(
        [str(structure), str(model), "--relax", "--optimizer", "BFGS", "--fmax", "1e-4", "--freq"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Stationary point: minimum (no imaginary modes)" in out


def test_cli_modes_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "x.xyz"), str(tmp_path / "m"), "--md", "--relax"])


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], {"use_hessian": True, "pair": None}),
        (["--ts-pair", "0-1"], {"use_hessian": False, "pair": (0, 1)}),
        (["--ts-start", "random"], {"use_hessian": False, "pair": None}),
    ],
)
def test_cli_ts_start_direction(monkeypatch, tmp_path, extra, expected):
    from samson_mlip_visualizer.ts import TSResult

    structure = tmp_path / "h2.xyz"
    model = tmp_path / "model.model"
    _write_structure(structure)
    model.write_bytes(b"x")
    seen = {}

    def spy_dimer(atoms, **kwargs):
        seen.update(kwargs)
        return TSResult(cli.evaluate(atoms), -1.0, 1, True, False)

    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: ConstantCalculator())
    monkeypatch.setattr(cli, "dimer_search", spy_dimer)

    assert cli.main([str(structure), str(model), "--ts", *extra]) == 0
    assert {key: seen[key] for key in expected} == expected


def test_cli_ts_pair_start_requires_pair(monkeypatch, tmp_path):
    structure = tmp_path / "h2.xyz"
    model = tmp_path / "model.model"
    _write_structure(structure)
    model.write_bytes(b"x")
    monkeypatch.setattr(cli, "create_calculator", lambda *a, **k: ConstantCalculator())

    with pytest.raises(SystemExit, match="--ts-pair"):
        cli.main([str(structure), str(model), "--ts", "--ts-start", "pair"])
