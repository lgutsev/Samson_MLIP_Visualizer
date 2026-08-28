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
