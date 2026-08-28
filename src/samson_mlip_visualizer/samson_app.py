"""Small PySide6 control panel intended to run inside SAMSON."""

from __future__ import annotations

from pathlib import Path

from .calculators import create_calculator
from .compat import assert_model_covers_structure
from .engine import evaluate, relax
from .provenance import collect_provenance
from .samson_bridge import extract_structure, sync_positions

_WINDOW = None


def _qt():
    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError as exc:
        raise RuntimeError("PySide6 is provided by SAMSON; run this module inside SAMSON") from exc
    return QtCore, QtWidgets


def _samson_main_window():
    """Best-effort handle to SAMSON's main window, so the panel docks sensibly."""
    try:
        from samson import SAMSON

        getter = getattr(SAMSON, "getMainWindow", None)
        return getter() if callable(getter) else None
    except Exception:
        return None


def _make_window():
    QtCore, QtWidgets = _qt()

    class MLIPWindow(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("SAMSON MLIP Visualizer")
            self.setMinimumWidth(520)
            self._stop_requested = False
            self._running = False

            self.backend = QtWidgets.QComboBox()
            self.backend.addItems(["MACE", "DeepMD"])
            self.model_path = QtWidgets.QLineEdit()
            browse = QtWidgets.QPushButton("Browse…")
            browse.clicked.connect(self._browse)
            model_row = QtWidgets.QHBoxLayout()
            model_row.addWidget(self.model_path, 1)
            model_row.addWidget(browse)

            self.device = QtWidgets.QComboBox()
            self.device.addItems(["cpu", "cuda"])
            self.dtype = QtWidgets.QComboBox()
            self.dtype.addItems(["float64", "float32"])
            self.fmax = QtWidgets.QDoubleSpinBox()
            self.fmax.setDecimals(4)
            self.fmax.setRange(0.0001, 10.0)
            self.fmax.setValue(0.05)
            self.fmax.setSuffix(" eV/Å")
            self.steps = QtWidgets.QSpinBox()
            self.steps.setRange(1, 100000)
            self.steps.setValue(250)

            form = QtWidgets.QFormLayout()
            form.addRow("Backend", self.backend)
            form.addRow("Model file", model_row)
            form.addRow("MACE device", self.device)
            form.addRow("MACE dtype", self.dtype)
            form.addRow("Force threshold", self.fmax)
            form.addRow("Maximum FIRE steps", self.steps)

            self.evaluate_button = QtWidgets.QPushButton("Single point")
            self.relax_button = QtWidgets.QPushButton("Relax positions")
            self.stop_button = QtWidgets.QPushButton("Stop")
            self.stop_button.setEnabled(False)
            self.evaluate_button.clicked.connect(self._evaluate)
            self.relax_button.clicked.connect(self._relax)
            self.stop_button.clicked.connect(self._request_stop)
            actions = QtWidgets.QHBoxLayout()
            actions.addWidget(self.evaluate_button)
            actions.addWidget(self.relax_button)
            actions.addWidget(self.stop_button)

            note = QtWidgets.QLabel(
                "Operates on one complete structural model. Select it in Document View when "
                "multiple models exist. SAMSON fixed-atom flags become ASE FixAtoms constraints."
            )
            note.setWordWrap(True)
            self.status = QtWidgets.QPlainTextEdit()
            self.status.setReadOnly(True)
            self.status.setMaximumBlockCount(500)

            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(form)
            layout.addWidget(note)
            layout.addLayout(actions)
            layout.addWidget(self.status, 1)

            self.backend.currentTextChanged.connect(self._backend_changed)
            self._backend_changed(self.backend.currentText())

        def _backend_changed(self, text):
            is_mace = text.lower() == "mace"
            self.device.setEnabled(is_mace)
            self.dtype.setEnabled(is_mace)

        def _browse(self):
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose MLIP model")
            if filename:
                self.model_path.setText(filename)

        def _log(self, message):
            self.status.appendPlainText(str(message))
            from samson import SAMSON

            SAMSON.processEvents()

        def _set_running(self, running):
            self._running = running
            self.evaluate_button.setEnabled(not running)
            self.relax_button.setEnabled(not running)
            self.stop_button.setEnabled(running)

        def _prepare(self):
            model_file = Path(self.model_path.text()).expanduser()
            backend = self.backend.currentText().lower()
            device = self.device.currentText()
            dtype = self.dtype.currentText()
            calculator = create_calculator(backend, model_file, device=device, dtype=dtype)
            try:
                provenance = collect_provenance(
                    backend=backend, model_path=model_file, device=device, dtype=dtype
                )
                self._log("Run provenance:\n" + provenance.as_text())
            except OSError as exc:
                self._log(f"Could not hash the model file for provenance: {exc}")
            structure = extract_structure()
            structure.ase_atoms.calc = calculator
            supported = assert_model_covers_structure(calculator, structure.ase_atoms)
            if supported is not None:
                self._log("Model training elements: " + ", ".join(supported))
            else:
                self._log(
                    "Could not read the model's element list. Confirm manually that it "
                    "was trained for every element and environment in this structure."
                )
            return structure

        def _evaluate(self):
            try:
                self._set_running(True)
                structure = self._prepare()
                result = evaluate(structure.ase_atoms)
                self._log(
                    f"Energy: {result.energy_ev:.10f} eV\n"
                    f"Maximum constrained force: {result.max_force_ev_per_angstrom:.6f} eV/Å"
                )
            except Exception as exc:
                self._show_error(exc)
            finally:
                self._set_running(False)

        def _relax(self):
            try:
                self._stop_requested = False
                self._set_running(True)
                structure = self._prepare()
                self._log("Starting position-only FIRE relaxation…")

                def progress(step, energy, max_force, positions):
                    sync_positions(structure, positions)
                    self._log(
                        f"Step {step:4d} | E {energy:.10f} eV | Fmax {max_force:.6f} eV/Å"
                    )

                from samson import SAMSON

                with SAMSON.holding("MLIP position relaxation"):
                    result = relax(
                        structure.ase_atoms,
                        fmax=self.fmax.value(),
                        max_steps=self.steps.value(),
                        on_progress=progress,
                        should_stop=lambda: self._stop_requested,
                    )
                    sync_positions(structure, structure.ase_atoms.get_positions())
                if result.stopped:
                    state = "stopped"
                elif result.converged:
                    state = "converged"
                else:
                    state = "step limit"
                self._log(
                    f"Finished ({state}) after {result.steps} steps; "
                    f"E = {result.evaluation.energy_ev:.10f} eV, "
                    f"Fmax = {result.evaluation.max_force_ev_per_angstrom:.6f} eV/Å"
                )
            except Exception as exc:
                self._show_error(exc)
            finally:
                self._set_running(False)

        def _request_stop(self):
            self._stop_requested = True
            self.stop_button.setEnabled(False)
            self._log("Stop requested; finishing the current MLIP evaluation…")

        def _show_error(self, exc):
            self._log(f"ERROR: {exc}")
            QtWidgets.QMessageBox.critical(self, "SAMSON MLIP Visualizer", str(exc))

    return MLIPWindow(_samson_main_window())


def show():
    """Show or raise the MLIP panel inside SAMSON."""
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = _make_window()
    _WINDOW.show()
    _WINDOW.raise_()
    _WINDOW.activateWindow()
    return _WINDOW
