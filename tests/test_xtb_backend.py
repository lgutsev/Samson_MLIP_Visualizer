"""xTB backend: the xtb subprocess calculator, with a fake xtb and (if installed) the real one."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculationFailed
from ase.units import Bohr, Hartree

from samson_mlip_visualizer import xtb_backend
from samson_mlip_visualizer.calculators import CalculatorLoadError, create_calculator
from samson_mlip_visualizer.compat import supported_species
from samson_mlip_visualizer.provenance import collect_provenance
from samson_mlip_visualizer.xtb_backend import (
    XTBCalculator,
    find_xtb,
    normalize_method,
    read_engrad,
)

SPRING = 0.3  # Eh/bohr² between every pair of atoms in the fake xtb


def spring_energy_gradient(positions_bohr):
    """A rotation-invariant pair potential: E = k/2 Σ (|r_ij| - 2)²."""
    energy, gradient = 0.0, np.zeros_like(positions_bohr)
    for i in range(len(positions_bohr)):
        for j in range(i + 1, len(positions_bohr)):
            delta = positions_bohr[i] - positions_bohr[j]
            r = np.linalg.norm(delta)
            energy += 0.5 * SPRING * (r - 2.0) ** 2
            g = SPRING * (r - 2.0) * delta / r
            gradient[i] += g
            gradient[j] -= g
    return energy, gradient


@pytest.fixture
def fake_xtb(tmp_path, monkeypatch):
    """Replace the subprocess with a fake xtb that writes an .engrad file."""
    executable = tmp_path / "xtb.exe"
    executable.write_text("")
    calls = []

    def run(command, cwd, **kwargs):
        calls.append(command)
        lines = (Path(cwd) / command[1]).read_text().splitlines()
        natoms = int(lines[0])
        positions = np.array([[float(v) for v in line.split()[1:]] for line in lines[2:]])
        energy, gradient = spring_energy_gradient(positions / Bohr)
        body = [f"{natoms}", f"{energy:.12f}", *[f"{g:.12f}" for g in gradient.ravel()]]
        (Path(cwd) / "samson_xtb.engrad").write_text(
            "#\n# Number of atoms\n#\n" + "\n".join(body[:1]) + "\n#\n# energy\n#\n"
            + body[1] + "\n#\n# gradient\n#\n" + "\n".join(body[2:]) + "\n"
        )
        (Path(cwd) / "xtbrestart").write_text("restart")
        return subprocess.CompletedProcess(command, 0, "normal termination", "")

    monkeypatch.setattr(xtb_backend.subprocess, "run", run)
    return executable, calls


def test_forces_come_back_in_the_structure_frame(fake_xtb):
    executable, calls = fake_xtb
    atoms = molecule("HCN")
    atoms.positions[2, 1] += 0.3
    atoms.calc = calc = XTBCalculator(executable)
    energy, gradient = spring_energy_gradient(atoms.positions / Bohr)
    assert atoms.get_potential_energy() == pytest.approx(energy * Hartree, rel=1e-9)
    assert np.allclose(atoms.get_forces(), -gradient * Hartree / Bohr, atol=1e-6)
    assert calls[0][:3] == [str(executable), "samson_xtb.xyz", "--grad"]
    # xtb saw a rotated copy, with no atom left on the z axis.
    lines = (calc._workdir / "samson_xtb.xyz").read_text().splitlines()[2:]
    sent = np.array([[float(v) for v in line.split()[1:]] for line in lines])
    assert np.abs(sent[:, :2]).min() > 1e-3


def test_command_line_options(fake_xtb):
    executable, _ = fake_xtb
    calc = XTBCalculator(executable, method="GFN-FF", charge=-1, multiplicity=2, solvent="water")
    command = calc.command()
    assert "--gfnff" in command
    assert command[command.index("--chrg") + 1] == "-1"
    assert command[command.index("--uhf") + 1] == "1"
    assert command[command.index("--alpb") + 1] == "water"


def test_restart_is_dropped_for_a_different_system(fake_xtb):
    executable, _ = fake_xtb
    calc = XTBCalculator(executable)
    water = molecule("H2O")
    water.calc = calc
    water.get_potential_energy()
    restart = calc._workdir / "xtbrestart"
    assert restart.exists()
    ammonia = molecule("NH3")
    ammonia.calc = calc
    restart.write_text("stale")
    ammonia.get_potential_energy()
    assert restart.read_text() == "restart"  # rewritten by the new run, not reused


def test_failures_and_periodic_cells_are_reported(fake_xtb, monkeypatch):
    executable, _ = fake_xtb
    slab = molecule("H2O", vacuum=3.0)
    slab.pbc = True
    slab.calc = XTBCalculator(executable)
    with pytest.raises(CalculationFailed, match="molecules only"):
        slab.get_potential_energy()

    def fail(command, cwd, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "[ERROR] SCF not converged")

    monkeypatch.setattr(xtb_backend.subprocess, "run", fail)
    water = molecule("H2O")
    water.calc = XTBCalculator(executable)
    with pytest.raises(CalculationFailed, match="SCF not converged"):
        water.get_potential_energy()


def test_read_engrad_checks_the_atom_count(tmp_path):
    path = tmp_path / "x.engrad"
    path.write_text("# n\n2\n# e\n-1.5\n# g\n" + "\n".join(["0.1"] * 6) + "\n")
    energy, gradient = read_engrad(path, 2)
    assert energy == -1.5 and gradient.shape == (2, 3)
    with pytest.raises(CalculationFailed):
        read_engrad(path, 3)


def test_method_names():
    assert normalize_method("GFN2") == "gfn2"
    assert normalize_method("GFN2-xTB") == "gfn2"
    assert normalize_method("GFN-FF") == "gfnff"
    with pytest.raises(ValueError):
        normalize_method("PM7")


def test_create_calculator_and_element_check(fake_xtb, tmp_path):
    executable, _ = fake_xtb
    calc = create_calculator("xtb", executable, options={"method": "gfn1", "charge": 1})
    assert calc.method == "gfn1" and calc.charge == 1
    assert {"H", "C", "Rn"} <= supported_species(calc)
    assert "Fr" not in supported_species(calc)
    model = tmp_path / "model.model"
    model.write_text("")
    with pytest.raises(CalculatorLoadError, match="xtb executable"):
        create_calculator("xtb", model)
    with pytest.raises(CalculatorLoadError, match="Invalid xTB settings"):
        create_calculator("xtb", executable, options={"method": "pm7"})


def test_provenance_records_the_settings(fake_xtb, monkeypatch):
    executable, _ = fake_xtb
    monkeypatch.setattr(xtb_backend, "xtb_version", lambda path: "6.7.1")
    provenance = collect_provenance(
        backend="xtb",
        model_path=executable,
        device="cuda",  # the panel's MACE settings, which do not apply to xtb
        dtype="float32",
        settings={"method": "gfn2", "charge": 0, "solvent": None},
    )
    flat = provenance.as_dict()
    assert flat["mlip_device"] == "cpu" and flat["mlip_dtype"] == "float64"
    assert flat["mlip_version_xtb"] == "6.7.1"
    assert flat["mlip_setting_method"] == "gfn2"
    assert "mlip_setting_solvent" not in flat


def test_find_xtb_prefers_the_environment_variable(tmp_path, monkeypatch):
    executable = tmp_path / "xtb.exe"
    executable.write_text("")
    monkeypatch.setenv("XTB_EXE", str(executable))
    assert find_xtb() == executable


@pytest.mark.skipif(find_xtb() is None, reason="xtb is not installed")
@pytest.mark.parametrize("method", ["gfn2", "gfnff"])
def test_real_xtb_forces_match_finite_differences(method):
    # Bent HCN built by ASE lies in the yz plane with C-N along z: the case where
    # xtb's own GFN2 gradient is wrong unless the structure is rotated first.
    atoms = molecule("HCN")
    atoms.positions[2, 1] += 0.3
    calc = XTBCalculator(find_xtb(), method=method)
    atoms.calc = calc
    forces = atoms.get_forces()
    step, numeric = 1e-3, np.zeros_like(forces)
    for i in range(len(atoms)):
        for k in range(3):
            energies = []
            for sign in (1, -1):
                displaced = atoms.copy()
                displaced.positions[i, k] += sign * step
                displaced.calc = calc
                energies.append(displaced.get_potential_energy())
            numeric[i, k] = -(energies[0] - energies[1]) / (2 * step)
    assert np.abs(forces - numeric).max() < 2e-3
