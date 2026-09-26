"""Where models, their mirror copies, and large downloads live."""

import pytest

from samson_mlip_visualizer import paths


@pytest.fixture
def user(monkeypatch, tmp_path):
    """A user with an empty home, config folder, and no path variables."""
    for name in (paths.MACE_DIR_ENV, paths.MIRROR_DIR_ENV, "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setattr(paths, "config_file", lambda: tmp_path / "state" / "config.json")
    return tmp_path


def test_mace_folder_order(user, monkeypatch):
    assert paths.mace_dir() == user / "home" / ".cache" / "mace"
    monkeypatch.setenv("XDG_CACHE_HOME", str(user / "xdg"))
    assert paths.mace_dir() == user / "xdg" / "mace"
    paths.set_mace_dir(user / "configured")
    assert paths.mace_dir() == (user / "configured").resolve()
    monkeypatch.setenv(paths.MACE_DIR_ENV, str(user / "env"))
    assert paths.mace_dir() == user / "env"
    assert paths.foundation_model() == user / "env" / paths.MACE_MP_SMALL
    assert paths.finetuned_dir() == user / "env" / "finetuned"
    paths.set_mace_dir(None)
    monkeypatch.delenv(paths.MACE_DIR_ENV)
    assert paths.mace_dir() == user / "xdg" / "mace"


def test_mirror_takes_downloads_only_while_reachable(user, monkeypatch):
    assert paths.mirror_dir() is None and paths.downloads_dir() == paths.mace_dir()
    paths.set_mirror_dir(user / "work" / "cache" / "mace")
    assert paths.downloads_dir() == (user / "work" / "cache" / "mace").resolve()
    environment = paths.mace_environment({})
    assert environment["XDG_CACHE_HOME"] == str((user / "work" / "cache").resolve())
    # A mirror on a drive that is not plugged in is ignored.
    monkeypatch.setenv(paths.MIRROR_DIR_ENV, "Q:\\unplugged\\mace" if paths.os.name == "nt"
                       else "/nonexistent-drive/mace")
    if paths.os.name == "nt":
        assert paths.mirror_dir() is None and paths.downloads_dir() == paths.mace_dir()


def test_mirror_copies_files_and_folders(user):
    assert paths.mirror(user / "anything") is None  # no mirror set
    model_folder = paths.finetuned_dir() / "demo"
    model_folder.mkdir(parents=True)
    (model_folder / "demo.model").write_text("weights")
    (model_folder / "demo.model.json").write_text("{}")
    paths.set_mirror_dir(user / "mirror")
    copy = paths.mirror(model_folder)
    assert copy == (user / "mirror").resolve() / "finetuned" / "demo"
    assert (copy / "demo.model").read_text() == "weights"
    foundation = paths.foundation_model()
    foundation.write_text("foundation")
    assert paths.mirror(foundation).read_text() == "foundation"
