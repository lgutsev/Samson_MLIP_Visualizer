"""VASP labeling packages: inputs per frame and level, the array script, collecting."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from samson_mlip_visualizer import vasp_labeling
from samson_mlip_visualizer.finetune import Selection, write_selection
from samson_mlip_visualizer.labeling import OutputError
from samson_mlip_visualizer.vasp_labeling import (
    collect_vasp_labels,
    incar,
    kpoint_mesh,
    potcar_names,
    write_vasp_package,
)

OUTCAR_OK = " vasp.6.4.2 18Apr23\n...\n aborting loop because EDIFF is reached\n" \
            "  General timing and accounting informations for this job:\n"


def bbvo():
    """Ba₂BiVO₆, the 10-atom rock-salt-ordered primitive cell (PBE+U geometry)."""
    a = 4.2436937690695196
    cell = [[0, a, a], [a, 0, a], [a, a, 0]]
    u, v = 0.7258100020151517, 0.2741899979848483
    frac = [[0.25] * 3, [0.75] * 3, [0.5] * 3, [0.0] * 3,
            [u, v, v], [v, u, u], [v, u, v], [u, v, u], [v, v, u], [u, u, v]]
    return Atoms("Ba2VBiO6", scaled_positions=frac, cell=cell, pbc=True)


def package(tmp_path, frames, **options):
    selection = Selection()
    for index in range(len(frames)):
        selection.add(index, "test", None, float("inf"))
    frames_path, manifest_path = write_selection(tmp_path / "sel", frames, selection,
                                                 source="test")
    return write_vasp_package(tmp_path / "pkg", frames_path, manifest_path, **options)


def test_inputs_follow_materials_project_settings():
    atoms = bbvo()
    assert potcar_names(["Ba", "V", "Bi", "O"]) == ["Ba_sv", "V_pv", "Bi", "O"]
    assert kpoint_mesh(atoms, 0.25) == (6, 6, 6)
    pbe_u = incar("pbe_u", atoms)
    assert "LDAUU = 0 3.25 0 0" in pbe_u and "LDAUL = -1 2 -1 -1" in pbe_u
    assert "ISPIN = 1" in pbe_u and "LHFCALC" not in pbe_u
    hse = incar("hse06", atoms)
    assert "LHFCALC = .TRUE." in hse and "AEXX = 0.25" in hse and "HFSCREEN = 0.2" in hse
    assert "LDAU" not in hse and "ISTART = 1" in hse
    for text in (pbe_u, hse):  # what makes the two levels comparable
        for shared in ("ENCUT = 520", "LREAL = .FALSE.", "ISYM = 0", "ISIF = 2", "NSW = 0"):
            assert shared in text
    # No U without O or F, as in the Materials Project
    assert "LDAU" not in incar("pbe_u", Atoms("V2", cell=[3, 3, 3], pbc=True))
    spin = incar("pbe_u", atoms, magmom={"V": 1.0})
    assert "ISPIN = 2" in spin and "MAGMOM = 0 0 1 0 0 0 0 0 0 0" in spin
    with pytest.raises(ValueError, match="POTCAR"):
        potcar_names(["Og"])


def test_package_layout_and_script(tmp_path):
    rattled = bbvo()
    rattled.positions += np.random.default_rng(0).normal(0, 0.02, rattled.positions.shape)
    pkg = package(tmp_path, [bbvo(), rattled], incar_extra={"hse06": {"NCORE": 4}})
    frame = pkg / "inputs" / "frame_0001"
    assert (frame / "POTCAR.names").read_text().split() == ["Ba_sv", "V_pv", "Bi", "O"]
    assert "NCORE = 4" in (frame / "INCAR.hse06").read_text()
    assert "NCORE" not in (frame / "INCAR.pbe_u").read_text()
    assert (frame / "KPOINTS").read_text().splitlines()[3] == "6 6 6"
    assert (frame / "POSCAR").read_text().splitlines()[5].split() == ["Ba", "V", "Bi", "O"]
    script = (pkg / "run_vasp.slurm").read_text()
    assert "#SBATCH --array=0-1%10" in script
    assert script.index('INCAR.pbe_u') < script.index('INCAR.hse06')  # PBE+U first
    assert 'cp "$work/$previous/WAVECAR"' in script  # HSE06 restarts from it
    info = json.loads((pkg / "package.json").read_text())
    assert info["levels"] == ["pbe_u", "hse06"] and info["code"] == "vasp"
    assert info["placeholders_left"] == ["<ACCOUNT>", "<PARTITION>", "<POTPAW_PBE_DIR>",
                                         "<VASP_COMMAND>", "<VASP_MODULE>"]
    assert "6×6×6" in (pkg / "README.md").read_text(encoding="utf-8")


def test_refuses_what_it_cannot_label(tmp_path):
    with pytest.raises(ValueError, match="periodic"):
        package(tmp_path / "a", [Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]])])
    mixed = Atoms("OBaO", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0, 0]],
                  cell=[4, 4, 4], pbc=True)
    with pytest.raises(ValueError, match="Group the atoms"):
        package(tmp_path / "b", [mixed])


def fake_outputs(pkg, frames, *, broken=None):
    """Runs as VASP would leave them; ``broken`` (frame, level) did not converge."""
    for index in range(len(frames)):
        for k, level in enumerate(("pbe_u", "hse06")):
            folder = pkg / "outputs" / f"frame_{index:04d}" / level
            folder.mkdir(parents=True)
            text = OUTCAR_OK if (index, level) != broken else OUTCAR_OK.replace("EDIFF", "")
            (folder / "OUTCAR").write_text(text)
            (folder / "vasprun.xml").write_text(f"{index} {k}")


def fake_read(frames, moved=None):
    """vasprun.xml as ASE reads it; frame ``moved`` was computed at another geometry."""

    def read(path):
        index, k = map(int, path.read_text().split())
        atoms = frames[index].copy()
        if index == moved:
            atoms.positions[0] += 0.01
        atoms.calc = SinglePointCalculator(
            atoms, energy=-66.0 - 18.0 * k - index, free_energy=-66.0 - 18.0 * k - index,
            forces=np.full((len(atoms), 3), 0.1 * (k + 1)), stress=np.full(6, 0.001 * (k + 1)))
        return atoms
    return read


def test_collect_keeps_frames_with_every_level_good(tmp_path, monkeypatch):
    frames = [bbvo(), bbvo(), bbvo()]
    frames[1].positions[4] += 0.03
    frames[2].positions[5] -= 0.03
    pkg = package(tmp_path, frames)
    fake_outputs(pkg, frames, broken=(1, "hse06"))
    monkeypatch.setattr(vasp_labeling, "_read_vasprun", fake_read(frames, moved=2))
    result = collect_vasp_labels(pkg)
    assert len(result.labeled) == 1
    assert "EDIFF" in result.rejected[1]["hse06"] and "pbe_u" not in result.rejected[1]
    assert "geometry differs" in result.rejected[2]["pbe_u"]
    good = result.labeled[0]
    assert good.info["PBEU_energy"] == -66.0 and good.info["HSE06_energy"] == -84.0
    np.testing.assert_allclose(good.arrays["HSE06_forces"], 0.2)
    np.testing.assert_allclose(good.info["PBEU_stress"], 0.001)
    report = json.loads((pkg / "collect_report.json").read_text())
    assert report["labeled"] == 1 and set(report["rejected"]) == {"frame_0001", "frame_0002"}
    assert (pkg / "labeled.extxyz").is_file()


def test_parse_reports_why_a_run_is_unusable(tmp_path):
    with pytest.raises(OutputError, match="missing"):
        vasp_labeling.parse_vasp_run(tmp_path)
    (tmp_path / "OUTCAR").write_text("started\n")
    (tmp_path / "vasprun.xml").write_text("<modeling>")
    with pytest.raises(OutputError, match="did not finish"):
        vasp_labeling.parse_vasp_run(tmp_path)
    (tmp_path / "OUTCAR").write_text(OUTCAR_OK)
    with pytest.raises(OutputError, match="could not be read"):
        vasp_labeling.parse_vasp_run(tmp_path)


def test_a_site_potcar_generator_is_run_checked_and_recorded(tmp_path, monkeypatch):
    frames = [bbvo()]
    pkg = package(tmp_path, frames, slurm=vasp_labeling.VaspSlurmSettings(
        potcar_command="/home/me/bin/POTCAR_gen",
        setup=("export SINGULARITYENV_OMP_NUM_THREADS=1",)))
    script = (pkg / "run_vasp.slurm").read_text()
    assert '(cd "$work" && /home/me/bin/POTCAR_gen)' in script
    assert "POTCAR.used" in script and "POTCARs differ from POTCAR.names" in script
    assert "export SINGULARITYENV_OMP_NUM_THREADS=1" in script
    assert "$POTCARS/$name/POTCAR" not in script
    assert "<POTPAW_PBE_DIR>" not in json.loads((pkg / "package.json").read_text())[
        "placeholders_left"]
    fake_outputs(pkg, frames)
    (pkg / "outputs" / "frame_0000" / "POTCAR.used").write_text("Ba_sv\nV_pv\nBi\nO\n")
    monkeypatch.setattr(vasp_labeling, "_read_vasprun", fake_read(frames))
    result = collect_vasp_labels(pkg)
    assert result.labeled[0].info["potcars"] == "Ba_sv V_pv Bi O"
    assert json.loads((pkg / "collect_report.json").read_text())["potcars_used"] == [
        "Ba_sv V_pv Bi O"]
