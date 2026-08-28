import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from samson_mlip_visualizer import cli


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": -1.5, "forces": np.zeros((len(atoms), 3))}


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
