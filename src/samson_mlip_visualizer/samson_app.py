"""Small PySide6 control panel intended to run inside SAMSON."""

from __future__ import annotations

import os
from pathlib import Path

from .calculators import create_calculator
from .compat import assert_model_covers_structure
from .engine import evaluate, relax
from .md import ENSEMBLES, md_warnings, parse_pairs, run_md
from .provenance import collect_provenance
from .samson_bridge import extract_structure, selected_atom_indices, sync_positions
from .ts import dimer_search
from .vibrations import _free_indices, harmonic_frequencies

_WINDOW = None
_TS_DIRECTIONS = ("Softest Hessian mode", "Stretch atom pair", "Random")
SETTINGS_ENV = "SAMSON_MLIP_SETTINGS"
# Panel inputs remembered between sessions; attribute names double as settings keys.
# Structure-specific inputs (atom pairs, trajectory path) and the Python-execution
# opt-in of the remote bridge are deliberately not remembered.
_REMEMBERED = (
    "backend",
    "model_path",
    "device",
    "dtype",
    "min_distance",
    "max_force_std",
    "optimizer",
    "fmax",
    "steps",
    "ensemble",
    "temperature",
    "timestep",
    "md_steps",
    "friction",
    "tdamp",
    "seed",
    "report_interval",
    "max_temperature",
    "ts_direction",
    "ts_displacement",
    "ts_fmax",
    "ts_steps",
    "ts_check",
    "arrow_length",
    "tabs",
)


def settings_path() -> Path:
    """Per-user file where the panel remembers its settings."""
    override = os.environ.get(SETTINGS_ENV)
    if override:
        return Path(override)
    from .remote.protocol import data_dir

    return data_dir() / "panel.ini"


def _default_model_path() -> str:
    """MACE-MP-0 small, if MACE has already downloaded it to its usual cache."""
    candidate = Path.home() / ".cache" / "mace" / "20231210mace128L0_energy_epoch249model"
    return str(candidate) if candidate.is_file() else ""


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

    def spin(decimals, low, high, value, suffix="", tooltip=None):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setValue(value)
        if suffix:
            widget.setSuffix(suffix)
        if tooltip:
            widget.setToolTip(tooltip)
        return widget

    def int_spin(low, high, value, tooltip=None):
        widget = QtWidgets.QSpinBox()
        widget.setRange(low, high)
        widget.setValue(value)
        if tooltip:
            widget.setToolTip(tooltip)
        return widget

    def grid(rows):
        """Rows of (label, field) pairs in two aligned columns.

        A row holding one pair spans the full width; ``None`` leaves a cell empty.
        """
        layout = QtWidgets.QGridLayout()
        layout.setHorizontalSpacing(10)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        for row, pairs in enumerate(rows):
            span = 3 if len(pairs) == 1 else 1
            for column, pair in enumerate(pairs):
                if pair is None:
                    continue
                label, field = pair
                layout.addWidget(QtWidgets.QLabel(label), row, 2 * column)
                if isinstance(field, QtWidgets.QLayout):
                    layout.addLayout(field, row, 2 * column + 1, 1, span)
                else:
                    layout.addWidget(field, row, 2 * column + 1, 1, span)
        return layout

    class MLIPWindow(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("SAMSON MLIP Visualizer")
            self.setMinimumWidth(560)
            self._stop_requested = False
            self._running = False

            # --- model settings shared by every task -------------------------------
            self.backend = QtWidgets.QComboBox()
            self.backend.addItems(["MACE", "DeepMD"])
            self.model_path = QtWidgets.QLineEdit()
            self.model_path.setPlaceholderText(
                "One model file, or several MACE files for an uncertainty committee"
            )
            browse = QtWidgets.QPushButton("Browse…")
            browse.clicked.connect(self._browse)
            model_row = QtWidgets.QHBoxLayout()
            model_row.addWidget(self.model_path, 1)
            model_row.addWidget(browse)

            self.device = QtWidgets.QComboBox()
            self.device.addItems(["cpu", "cuda"])
            self.dtype = QtWidgets.QComboBox()
            self.dtype.addItems(["float64", "float32"])
            self.min_distance = spin(
                2, 0.0, 5.0, 0.5, " Å", "Abort if two atoms come closer than this. 0 disables."
            )
            self.max_force_std = spin(
                3,
                0.0,
                100.0,
                0.0,
                " eV/Å",
                "Abort a committee run when the per-atom force spread exceeds this. 0 disables.",
            )

            settings = grid(
                [
                    [("Backend", self.backend)],
                    [("Model file(s)", model_row)],
                    [("MACE device", self.device), ("MACE dtype", self.dtype)],
                    [
                        ("Min. atom distance", self.min_distance),
                        ("Max committee σ", self.max_force_std),
                    ],
                ]
            )

            self.tabs = QtWidgets.QTabWidget()
            self.tabs.addTab(self._relax_tab(), "Relax")
            self.tabs.addTab(self._md_tab(), "MD")
            self.tabs.addTab(self._ts_tab(), "TS search")

            self.stop_button = QtWidgets.QPushButton("Stop")
            self.stop_button.setEnabled(False)
            self.stop_button.clicked.connect(self._request_stop)

            note = QtWidgets.QLabel(
                "Runs on the selected structural models (or the only one) as one system; "
                "atoms fixed in SAMSON stay fixed."
            )
            note.setToolTip(
                "Operates on complete structural models. When the document holds several, "
                "select every model that belongs to the system in Document View; they are "
                "evaluated together. SAMSON fixed-atom flags become ASE FixAtoms constraints."
            )
            note.setWordWrap(True)
            self.status = QtWidgets.QPlainTextEdit()
            self.status.setReadOnly(True)
            self.status.setMaximumBlockCount(2000)
            self.status.setPlaceholderText("Run output and remote-bridge activity appear here.")
            self.status.setMinimumHeight(8 * self.status.fontMetrics().lineSpacing())

            self.bridge_button = QtWidgets.QPushButton("Start bridge")
            self.bridge_button.setToolTip(
                "Let programs on this computer (scripts, notebooks, samson-remote) read and "
                "edit the document. Loopback only; every request needs a private token."
            )
            self.bridge_button.clicked.connect(self._toggle_bridge)
            self.bridge_exec = QtWidgets.QCheckBox("Allow Python execution")
            self.bridge_exec.setToolTip(
                "Also let connected programs run arbitrary Python inside SAMSON. "
                "Leave off unless you need it."
            )
            self.bridge_status = QtWidgets.QLabel()
            bridge_row = QtWidgets.QHBoxLayout()
            bridge_row.addWidget(QtWidgets.QLabel("Remote bridge"))
            bridge_row.addWidget(self.bridge_button)
            bridge_row.addWidget(self.bridge_exec)
            bridge_row.addWidget(self.bridge_status, 1)

            self._modes = None
            self._mode_nodes = {}
            self._player = None
            self.mode_choice = QtWidgets.QComboBox()
            self.mode_choice.setToolTip(
                "Modes from the last frequency calculation (Frequencies button, or the TS "
                "search's check). Imaginary modes are marked i."
            )
            self.animate_button = QtWidgets.QPushButton("Animate")
            self.animate_button.clicked.connect(self._toggle_mode_animation)
            self.arrow_length = spin(2, 0.1, 5.0, 1.0, " Å", "Length of the longest arrow.")
            self.arrows_button = QtWidgets.QPushButton("Arrows")
            self.arrows_button.setToolTip("Add displacement arrows for the chosen mode.")
            self.arrows_button.clicked.connect(self._add_mode_arrows)
            modes_row = QtWidgets.QHBoxLayout()
            modes_row.addWidget(QtWidgets.QLabel("Normal mode"))
            modes_row.addWidget(self.mode_choice, 1)
            modes_row.addWidget(self.animate_button)
            modes_row.addWidget(self.arrow_length)
            modes_row.addWidget(self.arrows_button)

            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(settings)
            layout.addWidget(self.tabs)
            layout.addLayout(modes_row)
            layout.addWidget(note)
            layout.addLayout(bridge_row)
            layout.addWidget(self.stop_button)
            layout.addWidget(self.status, 1)
            self._refresh_bridge()
            self._refresh_modes()

            self._run_buttons = [
                self.evaluate_button,
                self.relax_button,
                self.frequency_button,
                self.md_button,
                self.ts_button,
            ]
            self.backend.currentTextChanged.connect(self._backend_changed)
            self._backend_changed(self.backend.currentText())

            path = settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._settings = QtCore.QSettings(str(path), QtCore.QSettings.Format.IniFormat)
            self.model_path.setText(_default_model_path())
            self._restore_settings()
            self._remember_settings()
            self.tabs.currentChanged.connect(self._fit_tabs)
            self._fit_tabs(self.tabs.currentIndex())

        # --- tab construction ------------------------------------------------------

        def _relax_tab(self):
            page = QtWidgets.QWidget()
            self.optimizer = QtWidgets.QComboBox()
            self.optimizer.addItems(["FIRE", "LBFGS", "BFGS", "PreconLBFGS"])
            self.fmax = spin(4, 0.0001, 10.0, 0.05, " eV/Å")
            self.steps = int_spin(1, 100000, 250)

            self.evaluate_button = QtWidgets.QPushButton("Single point")
            self.relax_button = QtWidgets.QPushButton("Relax positions")
            self.frequency_button = QtWidgets.QPushButton("Frequencies")
            self.frequency_button.setToolTip(
                "Finite-difference Hessian: 6 force calls per free atom. Classifies the "
                "current geometry as a minimum or a saddle point."
            )
            self.evaluate_button.clicked.connect(self._evaluate)
            self.relax_button.clicked.connect(self._relax)
            self.frequency_button.clicked.connect(self._frequencies)
            actions = QtWidgets.QHBoxLayout()
            actions.addWidget(self.evaluate_button)
            actions.addWidget(self.relax_button)
            actions.addWidget(self.frequency_button)

            layout = QtWidgets.QVBoxLayout(page)
            layout.addLayout(
                grid(
                    [
                        [("Optimizer", self.optimizer)],
                        [("Force threshold", self.fmax), ("Maximum steps", self.steps)],
                    ]
                )
            )
            layout.addLayout(actions)
            return page

        def _md_tab(self):
            page = QtWidgets.QWidget()
            self.ensemble = QtWidgets.QComboBox()
            self.ensemble.addItems(list(ENSEMBLES))
            self.temperature = spin(1, 0.0, 10000.0, 300.0, " K")
            self.timestep = spin(2, 0.05, 10.0, 0.5, " fs", "0.5 fs is safe with hydrogen.")
            self.md_steps = int_spin(1, 10_000_000, 1000)
            self.friction = spin(4, 0.0001, 1.0, 0.01, " 1/fs", "Langevin friction.")
            self.tdamp = spin(
                1, 1.0, 100000.0, 100.0, " fs", "Bussi / Nosé–Hoover thermostat time constant."
            )
            self.seed = int_spin(0, 2_147_483_647, 0, "Random seed; 0 picks a fresh one.")
            self.report_interval = int_spin(
                1, 100000, 10, "Update SAMSON and the log every N steps."
            )
            self.max_temperature = spin(
                0,
                0.0,
                100000.0,
                0.0,
                " K",
                "Abort if the instantaneous temperature exceeds this. 0 disables.",
            )
            self.fixed_distances = QtWidgets.QLineEdit()
            self.fixed_distances.setPlaceholderText("0-based pairs, e.g. 0-3, 5-9:1.20 (Å)")
            self.fixed_distances.setToolTip(
                "Hold these atom-pair distances fixed (RATTLE). ':R' moves the pair to R Å "
                "first. The mean force along each pair is reported for free-energy work."
            )
            add_pair = QtWidgets.QPushButton("Add selected pair")
            add_pair.setToolTip("Select exactly two atoms in SAMSON, then click.")
            add_pair.clicked.connect(self._add_selected_pair)
            pair_row = QtWidgets.QHBoxLayout()
            pair_row.addWidget(self.fixed_distances, 1)
            pair_row.addWidget(add_pair)
            self.trajectory = QtWidgets.QLineEdit()
            self.trajectory.setPlaceholderText("Optional, e.g. C:/runs/md.extxyz")
            browse = QtWidgets.QPushButton("Browse…")
            browse.clicked.connect(self._browse_trajectory)
            trajectory_row = QtWidgets.QHBoxLayout()
            trajectory_row.addWidget(self.trajectory, 1)
            trajectory_row.addWidget(browse)

            settings = grid(
                [
                    [("Ensemble", self.ensemble), ("Temperature", self.temperature)],
                    [("Timestep", self.timestep), ("Steps", self.md_steps)],
                    [("Friction", self.friction), ("Thermostat time", self.tdamp)],
                    [("Seed", self.seed), ("Max temperature", self.max_temperature)],
                    [("Update every", self.report_interval), None],
                    [("Fixed distances", pair_row)],
                    [("Trajectory file", trajectory_row)],
                ]
            )

            self.md_button = QtWidgets.QPushButton("Run MD")
            self.md_button.clicked.connect(self._run_md)
            # One line, no word wrap: a wrapped label inside a tab defeats _fit_tabs.
            hint = QtWidgets.QLabel("Relax first: an unrelaxed start releases its strain as heat.")

            layout = QtWidgets.QVBoxLayout(page)
            layout.addLayout(settings)
            layout.addWidget(hint)
            layout.addWidget(self.md_button)
            self.ensemble.currentTextChanged.connect(self._ensemble_changed)
            self._ensemble_changed(self.ensemble.currentText())
            return page

        def _ts_tab(self):
            page = QtWidgets.QWidget()
            self.ts_direction = QtWidgets.QComboBox()
            self.ts_direction.addItems(list(_TS_DIRECTIONS))
            self.ts_direction.setToolTip(
                "Initial reaction direction. The Hessian mode is most robust for molecules; "
                "a bond that forms or breaks suits large systems."
            )
            self.ts_pair = QtWidgets.QLineEdit()
            self.ts_pair.setPlaceholderText("0-based pair, e.g. 4-7")
            use_pair = QtWidgets.QPushButton("Use selected pair")
            use_pair.clicked.connect(self._use_selected_ts_pair)
            pair_row = QtWidgets.QHBoxLayout()
            pair_row.addWidget(self.ts_pair, 1)
            pair_row.addWidget(use_pair)
            self.ts_displacement = spin(3, 0.001, 1.0, 0.05, " Å")
            self.ts_fmax = spin(4, 0.0001, 10.0, 0.01, " eV/Å")
            self.ts_steps = int_spin(1, 100000, 500)
            self.ts_check = QtWidgets.QCheckBox("Check with frequencies afterwards")
            self.ts_check.setChecked(True)

            settings = grid(
                [
                    [("Initial direction", self.ts_direction)],
                    [("Atom pair", pair_row)],
                    [
                        ("Initial displacement", self.ts_displacement),
                        ("Force threshold", self.ts_fmax),
                    ],
                    [("Maximum steps", self.ts_steps), ("", self.ts_check)],
                ]
            )

            self.ts_button = QtWidgets.QPushButton("Search transition state")
            self.ts_button.clicked.connect(self._search_ts)
            hint = QtWidgets.QLabel("Start near the transition state, not at a minimum.")
            hint.setToolTip(
                "Dimer method: it climbs from the current geometry to the nearest first-order "
                "saddle point. A transition state has exactly one imaginary frequency, which "
                "the frequency check confirms."
            )

            layout = QtWidgets.QVBoxLayout(page)
            layout.addLayout(settings)
            layout.addWidget(hint)
            layout.addWidget(self.ts_button)
            self.ts_direction.currentTextChanged.connect(
                lambda text: self.ts_pair.setEnabled(text == "Stretch atom pair")
            )
            self.ts_pair.setEnabled(False)
            return page

        # --- small helpers ---------------------------------------------------------

        def _backend_changed(self, text):
            is_mace = text.lower() == "mace"
            self.device.setEnabled(is_mace)
            self.dtype.setEnabled(is_mace)

        def _ensemble_changed(self, text):
            self.temperature.setEnabled(text != "NVE")
            self.friction.setEnabled(text == "Langevin")
            self.tdamp.setEnabled(text in ("Bussi", "NoseHooverChain"))

        def _fit_tabs(self, index):
            # A QTabWidget is as tall as its tallest page; size it to the visible one
            # so the log below gets the rest of the height.
            policy = QtWidgets.QSizePolicy.Policy
            for page in range(self.tabs.count()):
                size = policy.Preferred if page == index else policy.Ignored
                self.tabs.widget(page).setSizePolicy(size, size)
            self.tabs.setMaximumHeight(self.tabs.minimumSizeHint().height())

        def _restore_settings(self):
            for name in _REMEMBERED:
                if not self._settings.contains(name):
                    continue
                widget, value = getattr(self, name), self._settings.value(name)
                try:
                    if isinstance(widget, QtWidgets.QComboBox):
                        index = widget.findText(str(value))
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    elif isinstance(widget, QtWidgets.QLineEdit):
                        widget.setText(str(value))
                    elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                        widget.setValue(float(value))
                    elif isinstance(widget, QtWidgets.QSpinBox):
                        widget.setValue(int(value))
                    elif isinstance(widget, QtWidgets.QCheckBox):
                        widget.setChecked(str(value).lower() in ("true", "1"))
                    elif 0 <= int(value) < widget.count():
                        widget.setCurrentIndex(int(value))
                except (TypeError, ValueError):
                    continue  # an unreadable entry keeps the default

        def _save_setting(self, name):
            widget = getattr(self, name)
            if isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QtWidgets.QLineEdit):
                value = widget.text()
            elif isinstance(widget, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
                value = widget.value()
            elif isinstance(widget, QtWidgets.QCheckBox):
                value = widget.isChecked()
            else:
                value = widget.currentIndex()
            self._settings.setValue(name, value)
            self._settings.sync()

        def _remember_settings(self):
            for name in _REMEMBERED:
                widget = getattr(self, name)
                if isinstance(widget, QtWidgets.QComboBox):
                    signal = widget.currentTextChanged
                elif isinstance(widget, QtWidgets.QLineEdit):
                    signal = widget.textChanged
                elif isinstance(widget, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
                    signal = widget.valueChanged
                elif isinstance(widget, QtWidgets.QCheckBox):
                    signal = widget.toggled
                else:
                    signal = widget.currentChanged
                signal.connect(lambda *_, name=name: self._save_setting(name))

        def _browse(self):
            filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose MLIP model(s)")
            if filenames:
                self.model_path.setText(os.pathsep.join(filenames))

        def _browse_trajectory(self):
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "MD trajectory", "md.extxyz", "Extended XYZ (*.extxyz *.xyz);;ASE (*.traj)"
            )
            if filename:
                self.trajectory.setText(filename)

        def _model_files(self):
            entries = [
                part.strip()
                for part in self.model_path.text().split(os.pathsep)
                if part.strip()
            ]
            if not entries:
                raise RuntimeError("Choose at least one model file")
            paths = [str(Path(entry).expanduser()) for entry in entries]
            return paths[0] if len(paths) == 1 else paths

        def _log(self, message):
            self.status.appendPlainText(str(message))
            from samson import SAMSON

            SAMSON.processEvents()

        def _poll_stop(self):
            from samson import SAMSON

            SAMSON.processEvents()
            return self._stop_requested

        def _set_running(self, running):
            self._running = running
            for button in self._run_buttons:
                button.setEnabled(not running)
            self.stop_button.setEnabled(running)

        def _selected_pair(self):
            structure = extract_structure()
            indices = selected_atom_indices(structure)
            if len(indices) != 2:
                raise RuntimeError(
                    f"Select exactly two atoms in SAMSON (found {len(indices)} selected)."
                )
            i, j = indices
            atoms = structure.ase_atoms
            symbols = atoms.get_chemical_symbols()
            distance = atoms.get_distance(i, j, mic=True)
            self._log(f"Selected pair {symbols[i]}{i}–{symbols[j]}{j}: {distance:.4f} Å")
            return f"{i}-{j}"

        def _add_selected_pair(self):
            try:
                pair = self._selected_pair()
                existing = self.fixed_distances.text().strip().rstrip(",")
                self.fixed_distances.setText(f"{existing}, {pair}" if existing else pair)
            except Exception as exc:
                self._show_error(exc)

        def _use_selected_ts_pair(self):
            try:
                self.ts_pair.setText(self._selected_pair())
                self.ts_direction.setCurrentText("Stretch atom pair")
            except Exception as exc:
                self._show_error(exc)

        def _prepare(self):
            model_files = self._model_files()
            backend = self.backend.currentText().lower()
            device = self.device.currentText()
            dtype = self.dtype.currentText()
            calculator = create_calculator(backend, model_files, device=device, dtype=dtype)
            first_model = model_files if isinstance(model_files, str) else model_files[0]
            try:
                provenance = collect_provenance(
                    backend=backend, model_path=first_model, device=device, dtype=dtype
                )
                self._log("Run provenance:\n" + provenance.as_text())
                if not isinstance(model_files, str):
                    self._log(f"Committee of {len(model_files)} models.")
            except OSError as exc:
                self._log(f"Could not hash the model file for provenance: {exc}")
            structure = extract_structure()
            structure.ase_atoms.calc = calculator
            atoms = structure.ase_atoms
            periodic = "".join(axis for axis, flag in zip("xyz", atoms.pbc, strict=True) if flag)
            self._log(
                f"{len(atoms)} atoms from {len(structure.models)} structural model(s); "
                + (f"periodic along {periodic}." if periodic else "no periodic cell.")
            )
            supported = assert_model_covers_structure(calculator, atoms)
            if supported is not None:
                self._log("Model training elements: " + ", ".join(supported))
            else:
                self._log(
                    "Could not read the model's element list. Confirm manually that it "
                    "was trained for every element and environment in this structure."
                )
            return structure

        def _warn_float32(self, task):
            if self.backend.currentText().lower() == "mace" and self.dtype.currentText() == (
                "float32"
            ):
                self._log(
                    f"Warning: MACE recommends float64 for {task}; float32 force noise can "
                    "stall convergence."
                )

        def _log_uncertainty(self, evaluation):
            if evaluation.energy_std_ev is not None:
                self._log(
                    f"Committee energy σ: {evaluation.energy_std_ev:.6f} eV | "
                    f"max force σ: {evaluation.max_force_std_ev_per_angstrom:.6f} eV/Å"
                )

        def _run_task(self, task):
            from .remote import qt_server

            bridge = qt_server.status()
            if bridge is not None and bridge["busy"]:
                self._show_error(RuntimeError(f"SAMSON is busy: {bridge['busy']}."))
                return
            try:
                self._stop_requested = False
                self._stop_mode_animation()  # a job must start from the real geometry
                self._set_running(True)
                # Remote reads stay live during a job; remote edits wait until it ends.
                qt_server.set_busy("an MLIP job is running in the panel")
                task()
            except Exception as exc:
                self._show_error(exc)
            finally:
                qt_server.set_busy(None)
                self._set_running(False)

        def _toggle_bridge(self):
            from .remote import qt_server

            try:
                if qt_server.status() is None:
                    qt_server.serve(
                        allow_exec=self.bridge_exec.isChecked(),
                        log=self.status.appendPlainText,
                    )
                else:
                    qt_server.stop()
                    self._log("Remote bridge stopped.")
            except Exception as exc:
                self._show_error(exc)
            self._refresh_bridge()

        def _refresh_bridge(self):
            from .remote import qt_server

            state = qt_server.status()
            self.bridge_button.setText("Start bridge" if state is None else "Stop bridge")
            self.bridge_exec.setEnabled(state is None)
            if state is None:
                self.bridge_status.setText("stopped")
            else:
                self.bridge_status.setText(
                    f"listening on 127.0.0.1:{state['port']}"
                    + (" · Python execution ON" if state["allow_exec"] else "")
                )

        def _report_frequencies(self, structure):
            atoms = structure.ase_atoms
            free = len(_free_indices(atoms))
            self._log(f"Frequencies: {6 * free} force calls…")
            step = max(1, 3 * free // 10)

            def progress(done, total):
                if done % step == 0 or done == total:
                    self._log(f"  Hessian column {done}/{total}")

            result = harmonic_frequencies(atoms, on_progress=progress)
            values = " ".join(f"{value:.1f}" for value in result.wavenumbers_cm)
            removed = result.rigid_body_modes_removed
            self._log(
                "Wavenumbers (cm⁻¹, negative = imaginary"
                + (f"; {removed} rigid-body modes projected out" if removed else "")
                + f"):\n{values}"
            )
            self._log(f"Stationary point: {result.classification()}")
            hint = result.soft_mode_hint()
            if hint:
                self._log(f"Note: {hint}")
            self._set_modes(structure, result)
            return result

        # --- normal-mode display -----------------------------------------------------

        def _set_modes(self, structure, frequencies):
            self._stop_mode_animation()
            self._modes = (structure, frequencies)
            self._mode_nodes = {}
            self.mode_choice.clear()
            for index, value in enumerate(frequencies.wavenumbers_cm):
                shown = f"{abs(value):.1f}i" if value < 0 else f"{value:.1f}"
                self.mode_choice.addItem(f"Mode {index + 1}: {shown} cm⁻¹")
            self._refresh_modes()

        def _refresh_modes(self):
            available = self._modes is not None and self.mode_choice.count() > 0
            for widget in (self.mode_choice, self.animate_button, self.arrows_button):
                widget.setEnabled(available)
            playing = self._player is not None and self._player.playing
            self.animate_button.setText("Stop animation" if playing else "Animate")

        def _mode_label(self, kind):
            return f"{self.mode_choice.currentText()} {kind}"

        def _chosen_mode(self):
            from .samson_modes import full_mode

            structure, frequencies = self._modes
            return structure, full_mode(structure, frequencies, self.mode_choice.currentIndex())

        def _toggle_mode_animation(self):
            try:
                if self._player is not None and self._player.playing:
                    self._stop_mode_animation()
                    return
                from .samson_modes import PathPlayer, add_mode_path

                key = ("path", self.mode_choice.currentIndex())
                if key not in self._mode_nodes:
                    structure, mode = self._chosen_mode()
                    self._mode_nodes[key] = add_mode_path(
                        structure, mode, name=self._mode_label("animation")
                    )
                if self._player is None:
                    self._player = PathPlayer()
                self._player.play(self._mode_nodes[key])
                self._log(
                    f"Animating {self.mode_choice.currentText()}. The path is in Document "
                    "View; stopping returns the atoms to the computed geometry."
                )
            except Exception as exc:
                self._show_error(exc)
            self._refresh_modes()

        def _stop_mode_animation(self):
            if self._player is not None:
                self._player.stop()
            if hasattr(self, "animate_button"):
                self._refresh_modes()

        def _add_mode_arrows(self):
            try:
                from .samson_modes import add_mode_arrows

                structure, mode = self._chosen_mode()
                add_mode_arrows(
                    structure,
                    mode,
                    name=self._mode_label("arrows"),
                    max_length=self.arrow_length.value(),
                )
                self._log(
                    f"Added arrows for {self.mode_choice.currentText()} (longest "
                    f"{self.arrow_length.value():.2f} Å). Delete them in Document View."
                )
            except Exception as exc:
                self._show_error(exc)

        # --- tasks -----------------------------------------------------------------

        def _evaluate(self):
            def task():
                structure = self._prepare()
                result = evaluate(structure.ase_atoms)
                self._log(
                    f"Energy: {result.energy_ev:.10f} eV\n"
                    f"Maximum constrained force: {result.max_force_ev_per_angstrom:.6f} eV/Å"
                )
                self._log_uncertainty(result)

            self._run_task(task)

        def _relax(self):
            def task():
                structure = self._prepare()
                optimizer = self.optimizer.currentText()
                self._warn_float32("geometry optimization")
                self._log(f"Starting position-only {optimizer} relaxation…")

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
                        optimizer=optimizer,
                        min_distance=self.min_distance.value() or None,
                        max_force_std=self.max_force_std.value() or None,
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
                self._log_uncertainty(result.evaluation)

            self._run_task(task)

        def _frequencies(self):
            def task():
                structure = self._prepare()
                self._warn_float32("finite-difference frequencies")
                self._report_frequencies(structure)

            self._run_task(task)

        def _run_md(self):
            def task():
                structure = self._prepare()
                atoms = structure.ase_atoms
                ensemble = self.ensemble.currentText()
                for message in md_warnings(
                    atoms,
                    timestep_fs=self.timestep.value(),
                    ensemble=ensemble,
                    dtype=self.dtype.currentText(),
                ):
                    self._log(f"Warning: {message}")
                constraints = parse_pairs(self.fixed_distances.text())
                trajectory = self.trajectory.text().strip() or None
                thermostat = "" if ensemble == "NVE" else f" at {self.temperature.value():g} K"
                self._log(
                    f"Starting {ensemble} MD{thermostat}: {self.md_steps.value()} steps × "
                    f"{self.timestep.value():g} fs"
                    + (f", {len(constraints)} fixed distance(s)" if constraints else "")
                )

                def progress(frame):
                    sync_positions(structure, frame.positions)
                    forces = "".join(
                        f" | f{pair.i}-{pair.j} {value:+.3f}"
                        for pair, value in zip(constraints, frame.constraint_forces, strict=True)
                    )
                    self._log(
                        f"Step {frame.step:6d} | {frame.time_fs:8.1f} fs | "
                        f"Epot {frame.potential_ev:.5f} | Etot {frame.total_ev:.5f} eV | "
                        f"T {frame.temperature_k:7.1f} K{forces}"
                    )

                from samson import SAMSON

                with SAMSON.holding("MLIP molecular dynamics"):
                    result = run_md(
                        atoms,
                        ensemble=ensemble,
                        temperature_k=self.temperature.value(),
                        timestep_fs=self.timestep.value(),
                        steps=self.md_steps.value(),
                        friction_per_fs=self.friction.value(),
                        tdamp_fs=self.tdamp.value(),
                        seed=self.seed.value() or None,
                        distance_constraints=constraints,
                        report_interval=self.report_interval.value(),
                        min_distance=self.min_distance.value() or None,
                        max_force_std=self.max_force_std.value() or None,
                        max_temperature_k=self.max_temperature.value() or None,
                        trajectory=trajectory,
                        trajectory_interval=self.report_interval.value(),
                        on_progress=progress,
                        should_stop=self._poll_stop,
                    )
                    sync_positions(structure, atoms.get_positions())
                self._log(
                    f"MD {'stopped' if result.stopped else 'finished'} after {result.steps} "
                    f"steps ({result.time_fs:g} fs); mean T {result.mean_temperature_k:.1f} K"
                )
                if result.energy_drift_mev_per_atom_ps is not None:
                    self._log(
                        f"NVE energy drift: {result.energy_drift_mev_per_atom_ps:+.3f} "
                        "meV/atom/ps"
                    )
                for summary in result.constraint_forces:
                    self._log(
                        f"Constraint {summary.i}-{summary.j} at {summary.distance:.4f} Å: "
                        f"mean force {summary.mean_force_ev_per_angstrom:+.5f} eV/Å "
                        f"(σ {summary.std_ev_per_angstrom:.5f}, {summary.samples} samples; "
                        "positive pushes the atoms apart)"
                    )
                if trajectory:
                    self._log(f"Trajectory written to {trajectory}")

            self._run_task(task)

        def _search_ts(self):
            def task():
                structure = self._prepare()
                atoms = structure.ase_atoms
                self._warn_float32("transition-state searches")
                direction = self.ts_direction.currentText()
                pair = None
                if direction == "Stretch atom pair":
                    parsed = parse_pairs(self.ts_pair.text())
                    if len(parsed) != 1:
                        raise RuntimeError("Enter one atom pair, e.g. 4-7, or use the selection")
                    pair = (parsed[0].i, parsed[0].j)
                use_hessian = direction == "Softest Hessian mode"
                if use_hessian:
                    self._log(
                        f"Computing the Hessian for the initial direction "
                        f"({6 * len(atoms)} force calls)…"
                    )
                self._log(f"Starting dimer search ({direction.lower()})…")

                def progress(step, energy, max_force, curvature, positions):
                    sync_positions(structure, positions)
                    self._log(
                        f"Step {step:4d} | E {energy:.10f} eV | Fmax {max_force:.6f} eV/Å | "
                        f"curvature {curvature:+.4f} eV/Å²"
                    )

                from samson import SAMSON

                with SAMSON.holding("MLIP transition-state search"):
                    result = dimer_search(
                        atoms,
                        fmax=self.ts_fmax.value(),
                        max_steps=self.ts_steps.value(),
                        pair=pair,
                        use_hessian=use_hessian,
                        displacement=self.ts_displacement.value(),
                        min_distance=self.min_distance.value() or None,
                        on_progress=progress,
                        should_stop=self._poll_stop,
                    )
                    sync_positions(structure, atoms.get_positions())
                if result.stopped:
                    state = "stopped"
                elif result.converged:
                    state = "converged"
                else:
                    state = "not converged"
                self._log(
                    f"Dimer search finished ({state}) after {result.steps} steps; "
                    f"E = {result.evaluation.energy_ev:.10f} eV, "
                    f"Fmax = {result.evaluation.max_force_ev_per_angstrom:.6f} eV/Å, "
                    f"curvature {result.curvature:+.4f} eV/Å²"
                )
                if result.converged and self.ts_check.isChecked():
                    frequencies = self._report_frequencies(structure)
                    if frequencies.n_imaginary != 1:
                        self._log(
                            "Warning: not a first-order saddle point. Try another starting "
                            "geometry or initial direction."
                        )

            self._run_task(task)

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

