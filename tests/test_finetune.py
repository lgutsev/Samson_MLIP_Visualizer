"""Fine-tuning building blocks: distances, frame selection, energy offsets, manifests."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import read

from samson_mlip_visualizer.finetune import (
    Manifest,
    committee_spread,
    fit_element_offsets,
    frame_checksum,
    labeled_structure,
    select_for_labeling,
    structure_distance,
    write_selection,
)


def water(stretch=0.0, rotate=0.0):
    atoms = molecule("H2O")
    atoms.positions[1] += stretch * (atoms.positions[1] - atoms.positions[0])
    atoms.rotate(rotate, "z", center="COM")
    return atoms


# --- distances --------------------------------------------------------------------


def test_distance_ignores_rigid_motion_of_molecules():
    assert structure_distance(water(), water(rotate=35.0)) == pytest.approx(0.0, abs=1e-9)
    shifted = water()
    shifted.translate([1.0, -2.0, 0.5])
    assert structure_distance(water(), shifted) == pytest.approx(0.0, abs=1e-9)
    assert structure_distance(water(), water(stretch=0.2)) > 0.05
    with pytest.raises(ValueError, match="same atoms"):
        structure_distance(water(), molecule("NH3"))


def test_periodic_distance_uses_the_minimum_image_and_removes_translation():
    crystal = bulk("Cu", "fcc", a=3.6, cubic=True)
    moved = crystal.copy()
    moved.positions += 0.3  # a rigid translation
    assert structure_distance(crystal, moved) == pytest.approx(0.0, abs=1e-9)
    wrapped = crystal.copy()
    wrapped.positions[0] += [3.6 - 0.1, 0, 0]  # 0.1 Å across the boundary
    expected = np.sqrt(np.mean(np.sum((np.array([[-0.1, 0, 0]] + [[0, 0, 0]] * 3)
                                        - [-0.025, 0, 0]) ** 2, axis=1)))
    assert structure_distance(crystal, wrapped) == pytest.approx(expected)


# --- selection ----------------------------------------------------------------------


def candidates():
    # Twelve waters along a stretch; duplicates at 0.0 and 0.30 by construction.
    return [water(stretch=s) for s in [0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.35,
                                       0.4, 0.45]]


def test_committee_pass_takes_the_largest_spread_but_skips_near_duplicates():
    frames = candidates()
    spread = np.zeros(len(frames))
    spread[7], spread[8], spread[3] = 0.9, 0.8, 0.5  # 7 and 8 are identical frames
    chosen = select_for_labeling(frames, spread=spread, n_committee=2, min_distance=0.01)
    assert chosen.indices == [7, 3] and chosen.reasons == ["committee", "committee"]
    assert chosen.scores == [0.9, 0.5]


def test_diversity_pass_spreads_out_and_respects_labeled_frames():
    frames = candidates()
    chosen = select_for_labeling(frames, n_diverse=3, labeled=[water()], min_distance=0.01)
    # Farthest from the labeled unstretched water first, then from everything chosen.
    assert chosen.indices[0] == 11
    assert all(reason == "diversity" for reason in chosen.reasons)
    assert chosen.nearest_distance[0] == pytest.approx(structure_distance(water(), frames[11]))
    assert 0 not in chosen.indices and 1 not in chosen.indices  # duplicates of labeled
    # With a large minimum distance the pass stops early: after the farthest frame,
    # nothing else is 60 % of that distance from both it and the labeled water.
    farthest = structure_distance(water(), frames[11])
    few = select_for_labeling(frames, n_diverse=10, labeled=[water()], min_distance=0.6 * farthest)
    assert few.indices == [11]


def test_spot_checks_are_random_but_reproducible_and_not_duplicates():
    frames = candidates()
    first = select_for_labeling(frames, n_random=4, seed=3, min_distance=0.01)
    again = select_for_labeling(frames, n_random=4, seed=3, min_distance=0.01)
    assert first.indices == again.indices and set(first.reasons) == {"spot-check"}
    picked = [frames[i] for i in first.indices]
    pairs = [structure_distance(a, b) for k, a in enumerate(picked) for b in picked[k + 1 :]]
    assert min(pairs) >= 0.01
    with pytest.raises(ValueError, match="spread"):
        select_for_labeling(frames, n_committee=1)


def test_committee_spread_reads_a_committee_calculator():
    from ase.calculators.calculator import Calculator, all_changes

    class Committee(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            forces = np.zeros((len(atoms), 3))
            self.results = {"energy": 0.0, "forces": forces,
                            "energy_comm": [0.0, 0.2], "forces_comm": [forces, forces + 0.1]}

    energy, force = committee_spread([water(), water(0.1)], Committee())
    assert energy == pytest.approx([0.1, 0.1])
    assert force == pytest.approx([np.linalg.norm([0.05] * 3)] * 2)


# --- energy offsets -----------------------------------------------------------------


def test_offsets_recover_per_element_shifts_from_several_compositions():
    true = {"H": -13.6, "O": -2040.0, "N": -1480.0}
    structures = [molecule(name) for name in ("H2O", "NH3", "H2", "N2", "O2", "H2O2")]
    rng = np.random.default_rng(1)
    foundation = rng.normal(-20, 5, len(structures))
    reference = [f + sum(true[s] for s in a.get_chemical_symbols()) + rng.normal(0, 1e-3)
                 for a, f in zip(structures, foundation, strict=True)]
    fit = fit_element_offsets(structures, reference, foundation)
    assert fit.rank == 3 and not fit.warnings
    for element, value in true.items():
        assert fit.offsets[element] == pytest.approx(value, abs=0.01)
    assert fit.residual_rms_ev < 0.01
    held_out = molecule("N2H4")
    shift = sum(true[s] for s in held_out.get_chemical_symbols())
    assert fit.residuals([held_out], [shift - 5.0], [-5.0])[0] == pytest.approx(0.0, abs=0.05)
    assert fit.to_foundation_scale(shift - 5.0, held_out) == pytest.approx(-5.0, abs=0.05)
    with pytest.raises(ValueError, match="No fitted offset"):
        fit.correction(molecule("CH4"))


def test_offsets_warn_when_one_composition_cannot_pin_them_down():
    frames = [Atoms("CNH", positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.07 - d]])
              for d in (0.0, 0.05, 0.1)]
    fit = fit_element_offsets(frames, [-2538.0, -2537.9, -2537.7], [-19.5, -19.4, -19.2])
    assert fit.rank == 1 and "not unique" in fit.warnings[0]
    # It still aligns that composition: the HCN constant shift, split over the atoms.
    assert fit.correction(frames[0]) == pytest.approx(-2518.5, abs=0.05)
    with pytest.raises(ValueError, match="one reference"):
        fit_element_offsets(frames, [1.0], [1.0, 2.0, 3.0])


# --- manifests and labeled frames -----------------------------------------------------


def test_selection_round_trips_through_frames_and_manifest(tmp_path):
    frames = candidates()
    chosen = select_for_labeling(frames, n_diverse=3, n_random=1, min_distance=0.01)
    frames_path, manifest_path = write_selection(
        tmp_path / "round1", frames, chosen, source="stretch scan", model="MACE-MP-0 small",
        notes={"target": "water"},
    )
    manifest = Manifest.load(manifest_path)
    written = read(frames_path, ":")
    assert len(written) == len(chosen) == len(manifest.frames)
    assert manifest.verify(written) == []
    assert [e.extra["candidate_index"] for e in manifest.frames] == chosen.indices
    assert manifest.frames[0].nearest_labeled_distance is None  # nothing labeled before it
    written[1].positions[0, 0] += 1e-3
    assert manifest.verify(written) == [1]
    assert manifest.verify(written[:2]) == [1, 2, 3]
    assert frame_checksum(frames[0]) == frame_checksum(frames[1])


def test_labeled_structure_keys_and_offsets(tmp_path):
    from ase.io import write

    crystal = bulk("Cu", "fcc", a=3.6)
    stress = np.diag([0.1, 0.2, 0.3])
    stress[0, 1] = stress[1, 0] = 0.05
    fit = fit_element_offsets([crystal, crystal * (2, 1, 1)], [-10.0, -20.0], [-3.0, -6.0])
    labeled = labeled_structure(crystal, -10.0, np.zeros((1, 3)), stress=stress, offsets=fit,
                                tag="bulk", meta={"code": "VASP 6.4"})
    assert labeled.info["REF_energy_raw"] == -10.0
    assert labeled.info["REF_energy"] == pytest.approx(-3.0)
    assert labeled.info["REF_stress"] == pytest.approx([0.1, 0.2, 0.3, 0.0, 0.0, 0.05])
    write(tmp_path / "labeled.extxyz", [labeled])
    back = read(tmp_path / "labeled.extxyz")
    assert back.info["REF_energy"] == pytest.approx(-3.0) and back.info["code"] == "VASP 6.4"
    assert back.arrays["REF_forces"].shape == (1, 3)
