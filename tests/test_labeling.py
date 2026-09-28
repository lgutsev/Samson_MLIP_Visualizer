"""Labeling packages (Gaussian, ORCA) and the collectors for their outputs.

The outputs here are written by hand in the formats of Gaussian 16 and ORCA 5
(neither program is available in CI); run one real output through the collector
before trusting it with a new code version.
"""

import json

import numpy as np
import pytest
from ase.build import molecule
from ase.io import read
from ase.units import Bohr, Hartree

from samson_mlip_visualizer.finetune import (
    fit_element_offsets,
    select_for_labeling,
    write_selection,
)
from samson_mlip_visualizer.labeling import (
    OutputError,
    SlurmSettings,
    collect_labels,
    parse_gaussian_log,
    parse_orca,
    write_label_package,
)

NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8}


def gaussian_log(atoms, energy, forces, *, terminated=True):
    rows = "\n".join(
        f"{k + 1:7d}{NUMBERS[s]:11d}{0:12d}    {x:12.6f}{y:12.6f}{z:12.6f}"
        for k, (s, (x, y, z)) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions,
                                               strict=True))
    )
    force_rows = "\n".join(
        f"{k + 1:7d}{NUMBERS[s]:9d}     {fx:15.9f}{fy:15.9f}{fz:15.9f}"
        for k, (s, (fx, fy, fz)) in enumerate(zip(atoms.get_chemical_symbols(), forces,
                                                  strict=True))
    )
    dash = " " + "-" * 69
    text = f""" Entering Gaussian System, Link 0=g16
 ******************************************
 Gaussian 16:  ES64L-G16RevC.01  3-Jul-2019
                26-Sep-2026
 ******************************************
 #p PBEPBE/def2TZVP Force NoSymm SCF=Tight
                          Input orientation:
{dash}
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
{dash}
{rows}
{dash}
 SCF Done:  E(RPBE-PBE) =  {energy:.10f}     A.U. after   12 cycles
 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
{force_rows}
 -------------------------------------------------------------------
 Cartesian Forces:  Max     0.012345678 RMS     0.004567890
"""
    if terminated:
        text += " Normal termination of Gaussian 16 at Sat Sep 26 12:00:00 2026.\n"
    return text


def orca_files(atoms, energy, gradient, *, converged=True):
    bohr = atoms.positions / Bohr
    engrad = "\n".join([
        "#", "# Number of atoms", "#", f"{len(atoms)}", "#",
        "# The current total energy in Eh", "#", f"{energy:.12f}", "#",
        "# The current gradient in Eh/bohr", "#",
        *[f"{value:.12f}" for value in np.asarray(gradient).ravel()], "#",
        "# The atomic numbers and current coordinates in Bohr", "#",
        *[f"{NUMBERS[s]:4d} {x:13.7f}{y:13.7f}{z:13.7f}"
          for s, (x, y, z) in zip(atoms.get_chemical_symbols(), bohr, strict=True)],
    ]) + "\n"
    scf = ("SCF CONVERGED AFTER  12 CYCLES" if converged
           else "SCF NOT CONVERGED AFTER 125 CYCLES")
    out = f"""                                 * O   R   C   A *
                           Program Version 5.0.4 -  RELEASE  -
 *****************************************************
 *                     SUCCESS                       *
 *           {scf}          *
 *****************************************************
-------------------------   --------------------
FINAL SINGLE POINT ENERGY      {energy:.12f}
-------------------------   --------------------
                             ****ORCA TERMINATED NORMALLY****
"""
    return out, engrad


def selection(tmp_path, count=3):
    frames = []
    for k in range(count):
        atoms = molecule("H2O")
        atoms.positions[1] *= 1 + 0.05 * k
        frames.append(atoms)
    chosen = select_for_labeling(frames, n_diverse=count, min_distance=0.001)
    return write_selection(tmp_path / "selected", frames, chosen, source="test stretch",
                           model="MACE-MP-0 small")


def fake_forces(atoms, k):
    return np.full((len(atoms), 3), 0.001 * (k + 1))


def test_gaussian_package_and_collect(tmp_path):
    frames_path, manifest_path = selection(tmp_path)
    package = write_label_package(
        tmp_path / "gaussian", frames_path, manifest_path, code="gaussian", charge=0,
        slurm=SlurmSettings(account="chem123", cpus=16, memory_gb=32),
    )
    gjf = (package / "inputs" / "frame_0000.gjf").read_text()
    assert "%nprocshared=16" in gjf and "%mem=27GB" in gjf
    assert "#p PBEPBE/def2TZVP Force NoSymm SCF=Tight" in gjf and "\n0 1\n" in gjf
    script = (package / "run_gaussian.slurm").read_text()
    assert "--array=0-2%20" in script and "--account=chem123" in script
    assert "g16 < \"inputs/$frame.gjf\"" in script
    info = json.loads((package / "package.json").read_text())
    assert info["placeholders_left"] == ["<MODULE>", "<PARTITION>"]
    assert "`<PARTITION>`" in (package / "README.md").read_text()
    assert (package / "logs").is_dir()  # SLURM needs it for --output

    frames = read(package / "frames.extxyz", ":")
    (package / "outputs").mkdir()
    for k, frame in enumerate(frames[:2]):  # frame 2 never ran
        (package / "outputs" / f"frame_{k:04d}.log").write_text(
            gaussian_log(frame, -76.3 - 0.01 * k, fake_forces(frame, k)))
    result = collect_labels(package)
    assert len(result.labeled) == 2 and result.rejected == {2: "missing output"}
    first = result.labeled[0]
    assert first.info["REF_energy_raw"] == pytest.approx(-76.3 * Hartree)
    assert first.arrays["REF_forces"] == pytest.approx(fake_forces(frames[0], 0) * Hartree / Bohr)
    assert first.info["code"] == "Gaussian 16 ES64L-G16RevC.01"
    assert first.info["tag"] == "diversity" and first.info["source"] == "test stretch"
    report = json.loads((package / "collect_report.json").read_text())
    assert report["labeled"] == 2 and report["rejected"] == {"frame_0002": "missing output"}
    assert len(read(package / "labeled.extxyz", ":")) == 2


def test_orca_package_and_collect_with_offsets(tmp_path):
    frames_path, manifest_path = selection(tmp_path, count=2)
    package = write_label_package(tmp_path / "orca", frames_path, manifest_path, code="orca")
    inp = (package / "inputs" / "frame_0001.inp").read_text()
    assert inp.startswith("! PBE def2-TZVP def2/J EnGrad TightSCF")
    assert "%pal" in inp and "nprocs 8" in inp and "* xyz 0 1" in inp
    assert "$ORCA_BIN" in (package / "run_orca.slurm").read_text()
    frames = read(package / "frames.extxyz", ":")
    (package / "outputs").mkdir()
    for k, frame in enumerate(frames):
        out, engrad = orca_files(frame, -76.4, -fake_forces(frame, k))
        (package / "outputs" / f"frame_{k:04d}.out").write_text(out)
        (package / "outputs" / f"frame_{k:04d}.engrad").write_text(engrad)
    offsets = fit_element_offsets([molecule("H2O"), molecule("H2"), molecule("O2")],
                                  [-2078.0, -31.0, -4086.0], [-14.0, -7.0, -10.0])
    result = collect_labels(package, offsets=offsets)
    assert not result.rejected
    labeled = result.labeled[1]
    assert labeled.arrays["REF_forces"] == pytest.approx(fake_forces(frames[1], 1) * Hartree / Bohr)
    assert labeled.info["REF_energy"] == pytest.approx(
        -76.4 * Hartree - offsets.correction(frames[1]))
    assert labeled.info["code"] == "ORCA 5.0.4"
    assert "offsets" in json.loads((package / "collect_report.json").read_text())


def test_collector_rejects_bad_outputs(tmp_path):
    frames_path, manifest_path = selection(tmp_path, count=3)
    package = write_label_package(tmp_path / "g", frames_path, manifest_path, code="gaussian")
    frames = read(package / "frames.extxyz", ":")
    (package / "outputs").mkdir()
    unfinished = gaussian_log(frames[0], -76.3, fake_forces(frames[0], 0), terminated=False)
    moved = frames[1].copy()
    moved.positions[0, 0] += 0.01  # an output for another geometry
    (package / "outputs" / "frame_0000.log").write_text(unfinished)
    (package / "outputs" / "frame_0001.log").write_text(
        gaussian_log(moved, -76.3, fake_forces(moved, 1)))
    (package / "outputs" / "frame_0002.log").write_text(
        gaussian_log(frames[2], -76.3, fake_forces(frames[2], 2)).replace(
            " SCF Done:", " Convergence failure -- run terminated.\n SCF Done:"))
    result = collect_labels(package, write=False)
    assert not result.labeled
    assert "Normal termination" in result.rejected[0]
    assert "geometry differs" in result.rejected[1]
    assert "convergence failure" in result.rejected[2]


def test_parsers_reject_incomplete_files():
    water = molecule("H2O")
    out, engrad = orca_files(water, -76.4, np.zeros((3, 3)), converged=False)
    with pytest.raises(OutputError, match="not converged"):
        parse_orca(out, engrad)
    out, engrad = orca_files(water, -76.4, np.zeros((3, 3)))
    with pytest.raises(OutputError, match="disagree"):
        parse_orca(out.replace("-76.4", "-76.5"), engrad)
    log = gaussian_log(water, -76.3, np.zeros((3, 3)))
    with pytest.raises(OutputError, match="Forces"):
        parse_gaussian_log(log.split(" -----------------------------------------------------"
                                     "--------------\n Center     Atomic                   "
                                     "Forces")[0] + " Normal termination of Gaussian 16\n")
    parsed = parse_gaussian_log(log)
    assert parsed["positions_angstrom"] == pytest.approx(water.positions, abs=1e-6)


def test_package_refuses_periodic_or_changed_frames(tmp_path):
    from ase.build import bulk

    from samson_mlip_visualizer.finetune import Selection

    frames_path, manifest_path = selection(tmp_path, count=2)
    with pytest.raises(ValueError, match="code"):
        write_label_package(tmp_path / "x", frames_path, manifest_path, code="vasp")
    crystal = [bulk("Cu"), bulk("Cu", a=3.7)]
    chosen = Selection()
    chosen.add(0, "seed", None, np.inf)
    periodic = write_selection(tmp_path / "periodic", crystal, chosen, source="bulk")
    with pytest.raises(ValueError, match="periodic"):
        write_label_package(tmp_path / "p", *periodic, code="orca")
    frames = read(frames_path, ":")
    frames[0].positions[0, 0] += 0.1
    from ase.io import write

    write(frames_path, frames)
    with pytest.raises(ValueError, match="does not match"):
        write_label_package(tmp_path / "c", frames_path, manifest_path, code="orca")


def orca_energy_out(atoms, energy):
    """An ORCA single-point .out: the coordinate block and the final energy."""
    coordinates = "\n".join(f"  {s:<2} {x:12.6f} {y:12.6f} {z:12.6f}"
                            for s, (x, y, z) in zip(atoms.get_chemical_symbols(),
                                                    atoms.positions, strict=True))
    return (" Program Version 6.0.1 -  RELEASE  -\n"
            "---------------------------------\nCARTESIAN COORDINATES (ANGSTROEM)\n"
            f"---------------------------------\n{coordinates}\n\n"
            f"FINAL SINGLE POINT ENERGY      {energy:.12f}\n"
            "                             ****ORCA TERMINATED NORMALLY****\n")


def test_orca_energy_only_package_for_dlpno_ccsdt(tmp_path):
    frames_path, manifest_path = selection(tmp_path, count=2)
    package = write_label_package(tmp_path / "cc", frames_path, manifest_path, code="orca",
                                  level="DLPNO-CCSD(T) def2-TZVP def2-TZVP/C TightPNO",
                                  job="energy")
    inp = (package / "inputs" / "frame_0000.inp").read_text()
    assert inp.startswith("! DLPNO-CCSD(T) def2-TZVP def2-TZVP/C TightPNO TightSCF")
    assert "EnGrad" not in inp
    assert ".engrad" not in (package / "run_orca.slurm").read_text()
    assert json.loads((package / "package.json").read_text())["job"] == "energy"
    frames = read(package / "frames.extxyz", ":")
    (package / "outputs").mkdir()
    (package / "outputs" / "frame_0000.out").write_text(orca_energy_out(frames[0], -76.3))
    moved = frames[1].copy()
    moved.positions[0] += 0.01
    (package / "outputs" / "frame_0001.out").write_text(orca_energy_out(moved, -76.2))
    result = collect_labels(package)
    assert len(result.labeled) == 1 and "geometry differs" in result.rejected[1]
    labeled = result.labeled[0]
    assert labeled.info["REF_energy"] == pytest.approx(-76.3 * Hartree)
    assert "REF_forces" not in labeled.arrays
    with pytest.raises(ValueError, match="energy"):
        write_label_package(tmp_path / "g", frames_path, manifest_path, code="gaussian",
                            job="energy")
