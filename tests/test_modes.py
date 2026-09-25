import numpy as np
import pytest
from samson_fakes import Atom, FakeSamson, Model, install_samson_module

from samson_mlip_visualizer.modes import arrow_mesh, mode_arrows, mode_frames
from samson_mlip_visualizer.samson_bridge import extract_structure
from samson_mlip_visualizer.samson_modes import add_mode_arrows, add_mode_path, full_mode
from samson_mlip_visualizer.vibrations import FrequencyResult

POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
MODE = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, -1.0]])


def test_frames_span_one_period_from_the_start_geometry():
    frames = mode_frames(POSITIONS, MODE, max_displacement=0.3, frames=8)
    assert len(frames) == 8
    assert frames[0] == pytest.approx(POSITIONS)
    displacement = np.linalg.norm(frames[2] - POSITIONS, axis=1)  # sin = 1
    assert displacement.max() == pytest.approx(0.3)
    assert frames[6] - POSITIONS == pytest.approx(-(frames[2] - POSITIONS))
    with pytest.raises(ValueError):
        mode_frames(POSITIONS, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        mode_frames(POSITIONS, MODE[:1])


def test_arrow_mesh_geometry():
    mesh = arrow_mesh([[0, 0, 0]], [[0, 0, 1.0]], segments=8, head_length=0.2)
    # Shaft (2 rings), shaft cap (centre + ring), cone (ring + tips), cone cap.
    assert len(mesh.positions) == 8 * 2 + 9 + 8 * 2 + 9
    assert len(mesh.triangles) == 8 * 2 + 8 + 8 + 8
    assert mesh.triangles.max() < len(mesh.positions)
    assert np.linalg.norm(mesh.normals, axis=1) == pytest.approx(1.0)
    assert mesh.positions[:, 2].max() == pytest.approx(1.0)  # tip
    assert mesh.positions[:, 2].min() == pytest.approx(0.0)  # base
    assert mesh.colors.shape == (len(mesh.positions), 4)


def test_triangles_face_outward():
    mesh = arrow_mesh([[0, 0, 0]], [[0, 0, 1.0]], segments=12)
    corners = mesh.positions[mesh.triangles]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    vertex_normals = mesh.normals[mesh.triangles].mean(axis=1)
    assert (np.einsum("ij,ij->i", face_normals, vertex_normals) > 0).all()


def test_short_vectors_are_skipped_and_longest_is_scaled():
    assert len(arrow_mesh([[0, 0, 0]], [[0, 0, 0.01]]).positions) == 0
    mesh = mode_arrows(POSITIONS, MODE, max_length=2.0, min_length=0.5)
    # Only the second atom's arrow survives; it ends 2 Å below the atom.
    assert mesh.positions[:, 2].min() == pytest.approx(-2.0)
    assert mesh.positions[:, 0].min() > 0.5


class Recorder:
    """Stand-ins for the SAMSON node classes the mode display creates."""

    def __init__(self, samson):
        self.samson = samson
        recorder = self

        class SBNodeIndexer(list):
            def addNode(self, node):
                self.append(node)

        class SBConformation:
            def __init__(self, name, atoms):
                self.name = name
                self.positions = [list(atom.position) for atom in atoms]

        class Node:
            def create(self):
                self.created = True

        class SBPath(Node):
            def __init__(self, name, conformations):
                self.name = name
                self.conformations = list(conformations)

        class SBSurface:
            def __init__(self, triangles, count, indices, positions, normals, colors, *rest):
                assert len(indices) == 3 * triangles and len(positions) == 3 * count
                assert len(colors) == 4 * count
                recorder.surface_positions = np.reshape(positions, (-1, 3))

        class SBMesh(Node):
            def __init__(self, surfaces):
                self.surfaces = surfaces

        self.classes = dict(
            SBNodeIndexer=SBNodeIndexer,
            SBConformation=SBConformation,
            SBPath=SBPath,
            SBSurface=SBSurface,
            SBMesh=SBMesh,
        )
        self.added = []
        samson.getActiveDocument = lambda: type("Doc", (), {"addChild": self._add})()

    def _add(self, node):
        self.added.append(node)


@pytest.fixture
def document(monkeypatch):
    atoms = [Atom("H", [0, 0, 0]), Atom("H", [1, 0, 0])]
    samson = FakeSamson([Model(atoms, cell=None)])
    recorder = Recorder(samson)
    module = install_samson_module(monkeypatch, samson)
    for name, cls in recorder.classes.items():
        setattr(module, name, cls)
    return samson, recorder, atoms


def test_add_mode_path_snapshots_frames_and_restores(document):
    samson, recorder, atoms = document
    structure = extract_structure(samson)
    path = add_mode_path(structure, MODE, name="Mode 1", frames=4, max_displacement=0.5)
    assert path.created and recorder.added == [path]
    assert samson.holds == ["Add Mode 1"]
    assert len(path.conformations) == 4
    assert path.conformations[1].positions[1][2] == pytest.approx(-0.5)
    assert atoms[1].position == pytest.approx([1, 0, 0])  # geometry restored


def test_add_mode_arrows_builds_a_mesh_in_picometres(document):
    samson, recorder, _ = document
    structure = extract_structure(samson)
    node = add_mode_arrows(structure, MODE, name="Mode 1 arrows", max_length=1.0)
    assert node.created and node.name == "Mode 1 arrows"
    assert recorder.added == [node]
    assert recorder.surface_positions[:, 2].min() == pytest.approx(-100.0)


def test_full_mode_puts_zeros_on_fixed_atoms(document):
    samson, _, _ = document
    structure = extract_structure(samson)
    frequencies = FrequencyResult(np.array([-500.0]), np.array([[[0, 0, 1.0]]]), (1,), 20.0)
    expected = np.array([[0, 0, 0], [0, 0, 1.0]])
    assert full_mode(structure, frequencies, 0) == pytest.approx(expected)
