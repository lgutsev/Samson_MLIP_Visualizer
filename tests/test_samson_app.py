"""The PySide6 panel, offscreen; skipped where PySide6 is not installed."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from samson_fakes import install_samson_module, water_document  # noqa: E402

from samson_mlip_visualizer import samson_app  # noqa: E402

MACE_SMALL = "20231210mace128L0_energy_epoch249model"


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A fresh user: empty home directory and settings file, fake SAMSON."""
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv(samson_app.SETTINGS_ENV, str(tmp_path / "panel.ini"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    install_samson_module(monkeypatch, water_document())
    return tmp_path


def test_settings_persist_between_sessions(home):
    first = samson_app._make_window()
    first.device.setCurrentText("cuda")
    first.dtype.setCurrentText("float32")
    first.fmax.setValue(0.02)
    first.md_steps.setValue(500)
    first.ts_check.setChecked(False)
    first.tabs.setCurrentIndex(2)
    first.bridge_exec.setChecked(True)
    first.fixed_distances.setText("0-1")
    first.trajectory.setText("run.extxyz")

    second = samson_app._make_window()
    assert second.device.currentText() == "cuda"
    assert second.dtype.currentText() == "float32"
    assert second.fmax.value() == pytest.approx(0.02)
    assert second.md_steps.value() == 500
    assert not second.ts_check.isChecked()
    assert second.tabs.currentIndex() == 2
    # Never remembered: the Python-execution opt-in and structure-specific inputs.
    assert not second.bridge_exec.isChecked()
    assert second.fixed_distances.text() == ""
    assert second.trajectory.text() == ""


def test_model_path_defaults_to_the_mace_cache(home):
    assert samson_app._make_window().model_path.text() == ""
    model = home / ".cache" / "mace" / MACE_SMALL
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    assert samson_app._make_window().model_path.text() == str(model)


def test_saved_model_path_wins_over_the_default(home):
    model = home / ".cache" / "mace" / MACE_SMALL
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    samson_app._make_window().model_path.setText("C:/models/custom.model")
    assert samson_app._make_window().model_path.text() == "C:/models/custom.model"


def test_tab_area_fits_the_open_tab(home):
    window = samson_app._make_window()
    heights = []
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        heights.append(window.tabs.maximumHeight())
    relax, md, ts, path = heights
    assert md > relax and md > ts and md > path
    # The Relax tab no longer reserves room for the taller MD page.
    assert relax < window.tabs.widget(1).sizeHint().height()
