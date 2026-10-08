"""Settings dialog — surface CLI knobs the GUI uses.

Reads / writes :class:`bidsmgr.gui.app_settings.AppSettings` via
``QSettings``. All changes are applied on **Save** (no live binding) so
the user can experiment with values and cancel without commit.

Tabs: Display / System / Scan / Convert + post-convert. The Convert tab
lays the post-convert chain out as an indented hierarchy (parent step +
its sub-options), and a "Restore defaults" button resets every widget to
the :class:`AppSettings` field defaults.
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import agent_service
from .. import schema
from ..classifier import sequence_dict
from ..deface import engines as deface_engines
from ..deface import run as deface_run
from ..classifier import user_rules
from ..util.system_info import SystemInfo, get_system_info
from .app_settings import AI_DEVICE_MAPS, AI_QUANTIZATIONS, AppSettings


def _indented(child: QWidget, *, indent: int = 22) -> QWidget:
    """Wrap ``child`` in a left-indented container so it reads as a
    sub-option nested under the checkbox above it (the post-convert
    hierarchy tree)."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(indent, 0, 0, 0)
    lay.setSpacing(6)
    lay.addWidget(child)
    lay.addStretch(1)
    return box


def _bind_children(parent_cb: QCheckBox, *children: QWidget) -> None:
    """Enable ``children`` only while ``parent_cb`` is checked, and sync
    immediately so the initial state is correct."""
    def _sync(checked: bool) -> None:
        for c in children:
            c.setEnabled(checked)
    parent_cb.toggled.connect(_sync)
    _sync(parent_cb.isChecked())


# Human wording for the agent's constrained vocabularies. The combo's data
# role always carries the machine value; only the text is friendly, and
# app_settings is what decides which values are legal at all.
_DEVICE_LABELS = {
    "auto": "Automatic (let PyTorch decide)",
    "cpu": "CPU only",
    "cuda": "NVIDIA GPU (CUDA)",
    "mps": "Apple GPU (MPS)",
}
_QUANT_LABELS = {
    "none": "Full precision (the model as published)",
    "4bit": "4-bit (about a quarter of the memory)",
    "8bit": "8-bit (about half the memory)",
}


class SettingsDialog(QDialog):
    """Settings dialog: Display / System / Scan / Convert + post-convert.

    Theme + post-convert chain live under their natural homes. The
    inspector column visibility is NOT here — it's controlled via the
    table header's right-click menu, and that menu writes through to
    the same QSettings namespace.
    """

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS-Manager — Settings")
        self.resize(560, 600)
        self._settings = settings
        # Detected once: the worker-count spinboxes are capped at the host's
        # logical thread count so the user can never ask for more workers
        # than the machine has threads.
        self._sys: SystemInfo = get_system_info()

        v = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_bids_version_tab(), "BIDS version")
        tabs.addTab(self._build_display_tab(), "Display")
        tabs.addTab(self._build_system_tab(), "System")
        tabs.addTab(self._build_scan_tab(), "Scan")
        tabs.addTab(self._build_scan_rules_tab(), "Scan rules")
        tabs.addTab(self._build_convert_tab(), "Convert + post-convert")
        tabs.addTab(self._build_validation_tab(), "Validation")
        tabs.addTab(self._build_ai_tab(), "AI Agent")
        v.addWidget(tabs, 1)
        # Connected once, after every tab exists: the AI tab asks the
        # agent what models it offers the first time it is opened, and a
        # connection made while tabs were still being added would fire
        # for the initial selection.
        self._tabs = tabs
        tabs.currentChanged.connect(self._on_tab_changed)

        # Save / Cancel / Restore defaults.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(self._on_restore_defaults)
        v.addWidget(buttons)

        # Populate every widget from the current settings.
        self._load_into_widgets(self._settings)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    # Font scale presets shown in the Display tab. The combo stores the
    # human-readable label; the float multiplier is the second element.
    _FONT_SCALE_PRESETS: list[tuple[str, float]] = [
        ("Compact (0.85x)",        0.85),
        ("Normal (1.00x)",         1.00),
        ("Comfortable (1.15x)",    1.15),
        ("Large (1.30x)",          1.30),
        ("Extra large (1.50x)",    1.50),
    ]

    # Combo presets for the header brand mark.
    _HEADER_LOGO_PRESETS: list[tuple[str, str]] = [
        ("Default (monochrome mark)", "default"),
        ("App icon (full color)",     "app_icon"),
    ]

    def _build_display_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        form.addRow("Theme:", self._theme_combo)

        # Font scale: multiplies every font-size (QSS + delegate paints +
        # inline stylesheets + icon sizes) so the user can comfortably
        # nudge the whole UI up or down. Persisted under ``ui/font_scale``.
        self._font_scale_combo = QComboBox()
        for label, _value in self._FONT_SCALE_PRESETS:
            self._font_scale_combo.addItem(label)
        form.addRow("Font scale:", self._font_scale_combo)

        # Header brand artwork.
        self._header_logo_combo = QComboBox()
        for label, _value in self._HEADER_LOGO_PRESETS:
            self._header_logo_combo.addItem(label)
        form.addRow("Header logo:", self._header_logo_combo)

        # Editor tree: dotfiles and the machinery folders. Off by default,
        # because a dataset carries .bidsmgr/, .git/ and .bidsignore and none
        # of them are the data. On, they are shown dimmed.
        self._editor_show_hidden = QCheckBox(
            "Show hidden files and folders in the Editor tree"
        )
        self._editor_show_hidden.setToolTip(
            "Dotfiles and dot-folders (.bidsignore, .bidsmgr, .git) are "
            "hidden by default. Shown, they are dimmed so they do not "
            "compete with the dataset. Needed to open .bidsignore."
        )
        form.addRow("Editor tree:", self._editor_show_hidden)

        # Save as you go. Safe because every editor write goes through the
        # operation log, so an edit made without being asked for can still be
        # undone after the pane has moved on.
        self._editor_autosave = QCheckBox(
            "Save a sidecar edit as soon as the field is committed"
        )
        self._editor_autosave.setToolTip(
            "Off by default: edits wait for the Save button, and the toolbar "
            "says there are unsaved changes from the first keystroke either "
            "way.\n\nOn, a field commits when it loses focus or you press "
            "Enter and is written after a short pause, so a burst of typing "
            "is one write. Every write is reversible, so this cannot lose "
            "what was there before."
        )
        form.addRow("Editor saving:", self._editor_autosave)

        hint = QLabel(
            "Theme can also be toggled live via the sun / moon button "
            "in the top header. Font scale and header logo apply on Save."
        )
        hint.setStyleSheet("color: #8b949e;")
        hint.setWordWrap(True)
        form.addRow("", hint)

        return w

    def _build_system_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        info = QGroupBox("System info (detected)")
        form = QFormLayout(info)

        threads = self._sys.logical_threads
        cores = self._sys.physical_cores
        cpu_txt = f"{threads} logical threads"
        if cores:
            cpu_txt += f"  /  {cores} physical cores"
        form.addRow("CPU:", QLabel(cpu_txt))

        ram_gib = self._sys.total_ram_gib
        form.addRow(
            "Memory:",
            QLabel(f"{ram_gib:.1f} GiB total" if ram_gib is not None else "unknown"),
        )

        v.addWidget(info)

        note = QLabel(
            "Parallel-worker counts (Scan and Convert) are capped at the "
            f"detected thread count ({threads}). Asking for more workers than "
            "the machine has threads only adds scheduling overhead, so the "
            "spinboxes will not go higher."
        )
        note.setStyleSheet("color: #8b949e;")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _build_scan_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        defaults = QGroupBox("Scan defaults")
        form = QFormLayout(defaults)

        self._scan_jobs = QSpinBox()
        self._scan_jobs.setRange(1, self._sys.logical_threads)
        self._scan_jobs.setToolTip(
            f"Capped at the detected thread count ({self._sys.logical_threads})."
        )
        form.addRow("Parallel workers (-j):", self._scan_jobs)

        # EEG/MEG line frequency + montage are no longer scan settings: they
        # are recording metadata, edited as dropdowns per recording in the
        # inspection table (montage / line_freq columns) and dataset-wide in
        # the "Recording metadata" editor. Free-text fields here would have
        # been a second, inconsistent way to set the same values.

        self._scan_probe = QCheckBox(
            "Enable --probe-convert (run dcm2niix per series to enrich "
            "naming with the actual file count + extensions)"
        )
        form.addRow("Probe:", self._scan_probe)

        self._scan_preview = QCheckBox(
            "Record what the conversion fills in by itself"
        )
        self._scan_preview.setToolTip(
            "Note, per kind of file, the values dcm2niix and mne-bids produce, "
            "so the metadata form can show them instead of asking for a field "
            "nobody has to answer.\n\nCosts nothing extra: it reads what the "
            "scan, and any probe conversion, already produced. With Probe on "
            "it covers MRI and PET as well as EEG and MEG."
        )
        form.addRow("Converter fields:", self._scan_preview)

        self._scan_skip_bids_guess = QCheckBox(
            "Skip dcm2niix BidsGuess classifier (use only the legacy "
            "regex fallback layer)"
        )
        form.addRow("Classifier:", self._scan_skip_bids_guess)

        # Index widths.
        #
        # A group rather than a row of spin boxes: the first version was six
        # unlabelled numbers after the words "Index width", which says what
        # the control IS and nothing about what it does or why anyone would
        # touch it. The entities come from the schema, so a BIDS version that
        # adds one is offered it without an edit here.
        from ..editor.values import index_entities

        widths_box = QGroupBox("Index width in proposed names")
        widths_outer = QVBoxLayout(widths_box)
        widths_outer.setContentsMargins(12, 10, 12, 10)
        widths_outer.setSpacing(8)

        explain = QLabel(
            "An <b>index</b> entity is one whose value is a number: run, "
            "echo, and the others below. BIDS accepts <code>run-1</code> and "
            "<code>run-01</code> equally, so this is a house style rather "
            "than a correction.<br><br>"
            "Setting a width here makes the inspection table propose that "
            "width from the moment a scan finishes, so you never have to go "
            "and repad the dataset afterwards. <b>As found</b> keeps whatever "
            "the source gives, which is what every earlier version did and "
            "what stays out of your way."
        )
        explain.setWordWrap(True)
        explain.setObjectName("dlg-hint")
        widths_outer.addWidget(explain)

        self._index_widths: dict[str, QSpinBox] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        for i, entity in enumerate(index_entities()):
            try:
                info = schema.entity_key_info(entity)
                display, why = info.display_name, info.description.strip()
            except KeyError:
                display, why = entity, ""

            box = QSpinBox()
            box.setObjectName("ent-input")
            box.setRange(0, 6)
            box.setSpecialValueText("as found")
            box.setSuffix(" digits")
            box.setToolTip(why[:300] if why else f"The {entity} entity.")
            self._index_widths[entity] = box

            name = QLabel(f"<code>{entity}-</code>")
            name.setToolTip(display)
            sample = QLabel("")
            sample.setObjectName("dlg-hint")
            # Live example, because "2 digits" is abstract and
            # "run-1 becomes run-01" is not.
            def _sync(value, entity=entity, sample=sample):
                sample.setText(
                    f"{entity}-7 stays {entity}-7" if not value
                    else f"{entity}-7 becomes {entity}-{str(7).zfill(value)}"
                )
            box.valueChanged.connect(_sync)
            _sync(box.value())

            row, col = divmod(i, 2)
            grid.addWidget(name, row, col * 3)
            grid.addWidget(box, row, col * 3 + 1)
            grid.addWidget(sample, row, col * 3 + 2)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(5, 1)
        widths_outer.addLayout(grid)
        form.addRow("", widths_box)

        v.addWidget(defaults)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------
    # Scan rules tab (exclusions + user hints + read-only built-ins)
    # ------------------------------------------------------------------

    def _build_scan_rules_tab(self) -> QWidget:
        # Valid BIDS datatypes for the hint dropdowns (derivatives excluded -
        # user hints route to raw datatypes only).
        self._valid_datatypes = sorted(
            d for d in schema.list_datatypes() if d != "derivatives"
        )

        content = QWidget()
        v = QVBoxLayout(content)

        intro = QLabel(
            "These rules apply to MRI / DICOM series, which are classified "
            "from their SeriesDescription. EEG / MEG recordings are classified "
            "by a different built-in method (mne channel types) and are NOT "
            "affected by custom sequence hints. Path-based exclusions can still "
            "skip any modality."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8b949e;")
        v.addWidget(intro)

        # Exclusions.
        excl_box = QGroupBox("Scan exclusions (skip matching series)")
        ebl = QVBoxLayout(excl_box)
        self._excl_table = QTableWidget(0, 3)
        self._excl_table.setHorizontalHeaderLabels(["Pattern", "Match against", "Mode"])
        self._excl_table.verticalHeader().setVisible(False)
        self._excl_table.setMinimumHeight(130)
        self._excl_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        ebl.addWidget(self._excl_table)
        ebl.addLayout(self._rule_row_buttons(self._add_exclusion_row, self._excl_table))
        v.addWidget(excl_box)

        # User hints (MRI / DICOM only).
        hint_box = QGroupBox("Custom sequence hints (MRI / DICOM only)")
        hbl = QVBoxLayout(hint_box)
        self._hint_table = QTableWidget(0, 6)
        self._hint_table.setHorizontalHeaderLabels(
            ["Patterns (comma-separated)", "Datatype", "Suffix", "Task", "Mode", "Force"]
        )
        self._hint_table.verticalHeader().setVisible(False)
        self._hint_table.setMinimumHeight(150)
        self._hint_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        hbl.addWidget(self._hint_table)
        force_hint = QLabel(
            "Datatype + suffix are chosen from the BIDS schema (no free text). "
            "'Force' overrides even the dcm2niix classifier; otherwise a hint "
            "only beats the built-in regex layer. 'Task' is the optional "
            "task-<label> for func rows."
        )
        force_hint.setWordWrap(True)
        force_hint.setStyleSheet("color: #8b949e;")
        hbl.addWidget(force_hint)
        hbl.addLayout(self._rule_row_buttons(self._add_hint_row, self._hint_table))
        v.addWidget(hint_box)

        # Read-only built-in criteria.
        builtin_box = QGroupBox("Built-in MRI classifier criteria (read-only)")
        bbl = QVBoxLayout(builtin_box)
        builtin_note = QLabel(
            "What the MRI classifier already matches. EEG / MEG do not use this "
            "table - their datatype comes from mne channel types."
        )
        builtin_note.setWordWrap(True)
        builtin_note.setStyleSheet("color: #8b949e;")
        bbl.addWidget(builtin_note)
        builtin = QTableWidget(0, 4)
        builtin.setHorizontalHeaderLabels(
            ["Label / group", "Datatype", "Suffix", "Match patterns"]
        )
        builtin.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        builtin.verticalHeader().setVisible(False)
        builtin.setMinimumHeight(240)
        builtin.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self._populate_builtin_criteria(builtin)
        bbl.addWidget(builtin)
        v.addWidget(builtin_box)
        v.addStretch(1)

        # Whole tab scrolls (not just the built-in table).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _rule_row_buttons(self, add_cb, table: QTableWidget) -> QHBoxLayout:
        bar = QHBoxLayout()
        add = QPushButton("Add row")
        add.clicked.connect(lambda: add_cb())
        rem = QPushButton("Delete selected")
        rem.clicked.connect(lambda: self._delete_selected_rows(table))
        bar.addStretch(1)
        bar.addWidget(add)
        bar.addWidget(rem)
        return bar

    @staticmethod
    def _delete_selected_rows(table: QTableWidget) -> None:
        for r in sorted({i.row() for i in table.selectedIndexes()}, reverse=True):
            table.removeRow(r)

    def _add_exclusion_row(self, pattern: str = "", target: str = "sequence",
                           mode: str = "substring") -> None:
        t = self._excl_table
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(pattern))
        target_cb = QComboBox()
        target_cb.addItems(list(user_rules.EXCLUSION_TARGETS))
        target_cb.setCurrentText(target if target in user_rules.EXCLUSION_TARGETS else "sequence")
        t.setCellWidget(r, 1, target_cb)
        mode_cb = QComboBox()
        mode_cb.addItems(list(user_rules.MATCH_MODES))
        mode_cb.setCurrentText(mode if mode in user_rules.MATCH_MODES else "substring")
        t.setCellWidget(r, 2, mode_cb)

    def _add_hint_row(self, patterns: str = "", datatype: str = "", suffix: str = "",
                      task: str = "", mode: str = "substring", force: bool = False) -> None:
        t = self._hint_table
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(patterns))

        # Datatype + suffix are constrained dropdowns (no hand-typed labels).
        # The suffix list depends on the chosen datatype, so it re-fills
        # whenever the datatype changes.
        dt_cb = QComboBox()
        dt_cb.addItems(self._valid_datatypes)
        suffix_cb = QComboBox()

        def _refill_suffixes(dt: str) -> None:
            suffix_cb.blockSignals(True)
            suffix_cb.clear()
            try:
                suffix_cb.addItems(sorted(schema.list_suffixes(dt)))
            except Exception:
                pass
            suffix_cb.blockSignals(False)

        dt_cb.currentTextChanged.connect(_refill_suffixes)
        if datatype in self._valid_datatypes:
            dt_cb.setCurrentText(datatype)
        _refill_suffixes(dt_cb.currentText())   # seed for the initial datatype
        if suffix:
            idx = suffix_cb.findText(suffix)
            if idx >= 0:
                suffix_cb.setCurrentIndex(idx)
        t.setCellWidget(r, 1, dt_cb)
        t.setCellWidget(r, 2, suffix_cb)

        t.setItem(r, 3, QTableWidgetItem(task))
        mode_cb = QComboBox()
        mode_cb.addItems(list(user_rules.MATCH_MODES))
        mode_cb.setCurrentText(mode if mode in user_rules.MATCH_MODES else "substring")
        t.setCellWidget(r, 4, mode_cb)
        force_item = QTableWidgetItem()
        force_item.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
        )
        force_item.setCheckState(Qt.CheckState.Checked if force else Qt.CheckState.Unchecked)
        t.setItem(r, 5, force_item)

    @staticmethod
    def _populate_builtin_criteria(table: QTableWidget) -> None:
        rows: list[tuple[str, str, str, str]] = []
        for label, hint in sequence_dict.SEQUENCE_HINTS.items():
            dt = hint.container_override or (hint.datatype or "")
            rows.append((label, dt, hint.suffix or "", ", ".join(hint.patterns)))
        for rgx, suffix, dt in sequence_dict._DWI_DERIVATIVE_PATTERNS:
            rows.append(("dwi-derivative", dt, suffix, rgx))
        for task_label, pats in sequence_dict.TASK_HINT_PATTERNS.items():
            rows.append((f"task:{task_label}", "func", "(task entity)", ", ".join(pats)))
        table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for col, val in enumerate(cells):
                it = QTableWidgetItem(val)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                table.setItem(r, col, it)

    def _read_scan_rules(self) -> tuple[list[dict], list[dict], Optional[str]]:
        """Read both editable tables into list[dict]. Returns
        ``(hints, exclusions, error)`` - ``error`` non-None means a hint /
        regex was invalid and the dialog must not save."""
        # Exclusions.
        exclusions: list[dict] = []
        for r in range(self._excl_table.rowCount()):
            item = self._excl_table.item(r, 0)
            pattern = item.text().strip() if item else ""
            if not pattern:
                continue
            mode = self._excl_table.cellWidget(r, 2).currentText()
            if mode == "regex":
                err = user_rules.validate_regex(pattern)
                if err:
                    return [], [], f"Exclusion regex {pattern!r} is invalid: {err}"
            exclusions.append({
                "pattern": pattern,
                "target": self._excl_table.cellWidget(r, 1).currentText(),
                "match_mode": mode,
            })

        # Hints.
        hints: list[dict] = []
        valid_datatypes = schema.list_datatypes()
        for r in range(self._hint_table.rowCount()):
            pat_item = self._hint_table.item(r, 0)
            patterns = [p.strip() for p in (pat_item.text() if pat_item else "").split(",") if p.strip()]
            if not patterns:
                continue
            # Datatype + suffix come from constrained dropdowns.
            datatype = self._hint_table.cellWidget(r, 1).currentText().strip()
            suffix = self._hint_table.cellWidget(r, 2).currentText().strip()
            task = (self._hint_table.item(r, 3).text().strip() if self._hint_table.item(r, 3) else "")
            mode = self._hint_table.cellWidget(r, 4).currentText()
            force_item = self._hint_table.item(r, 5)
            force = bool(force_item and force_item.checkState() == Qt.CheckState.Checked)

            if not datatype or not suffix:
                return [], [], f"Hint for {patterns!r} needs both a datatype and a suffix."
            if datatype == "derivatives" or datatype not in valid_datatypes:
                return [], [], (
                    f"Hint datatype {datatype!r} is not a valid BIDS datatype. "
                    f"Choose one of: {', '.join(sorted(valid_datatypes))}."
                )
            if suffix not in schema.list_suffixes(datatype):
                return [], [], (
                    f"Suffix {suffix!r} is not valid for datatype {datatype!r}."
                )
            if mode == "regex":
                for p in patterns:
                    err = user_rules.validate_regex(p)
                    if err:
                        return [], [], f"Hint regex {p!r} is invalid: {err}"
            hints.append({
                "patterns": patterns,
                "datatype": datatype,
                "suffix": suffix,
                "task": task,
                "entities": {},
                "match_mode": mode,
                "force": force,
            })
        return hints, exclusions, None

    def _build_convert_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        convert = QGroupBox("Convert defaults")
        form = QFormLayout(convert)

        self._convert_jobs = QSpinBox()
        self._convert_jobs.setRange(1, self._sys.logical_threads)
        self._convert_jobs.setToolTip(
            f"Capped at the detected thread count ({self._sys.logical_threads})."
        )
        form.addRow("Parallel workers (-j):", self._convert_jobs)

        # Policy for a subject that already exists in the dataset (incremental
        # conversion). New sessions/datatypes always merge in; this governs
        # files that collide with existing ones. Replaces the old overwrite
        # checkbox (which mapped only to "replace").
        self._convert_on_existing = QComboBox()
        for value, label in (
            ("skip",    "Skip: keep existing files, add only new (safe default)"),
            ("update",  "Update: replace only files whose content changed"),
            ("replace", "Replace: back up and replace colliding files"),
            ("error",   "Error: abort a subject if any file would be overwritten"),
        ):
            self._convert_on_existing.addItem(label, userData=value)
        self._convert_on_existing.setToolTip(
            "What to do when a subject already exists in the dataset. Adding a "
            "new session or datatype always merges in; this only governs files "
            "that collide with existing ones. Existing data is never lost on the "
            "default (Skip)."
        )
        form.addRow("Existing subjects:", self._convert_on_existing)

        # Somebody curates a sidecar in the Editor, then re-converts that
        # subject. Deciding at file level throws the curation away; deciding at
        # field level keeps what a person stated and still takes what the fresh
        # pass newly knows.
        self._convert_preserve_curation = QCheckBox(
            "Keep curated metadata (merge sidecars field by field instead of "
            "overwriting them)"
        )
        self._convert_preserve_curation.setToolTip(
            "When a subject you already curated in the Editor is converted "
            "again, merge its JSON sidecars and _scans.tsv field by field: a "
            "value you stated is kept, a TODO placeholder is replaced, and "
            "anything the fresh conversion newly knows is added. Turn it off "
            "to let the fresh conversion win outright. Only has an effect "
            "with Update or Replace above. Recommended: on."
        )
        form.addRow("Curated metadata:", self._convert_preserve_curation)

        self._convert_skip_residuals = QCheckBox(
            "Skip residual volumes (drop dcm2niix secondary duplicates such "
            "as ..._bolda / _Eq_ / _ROI that are not real images)"
        )
        self._convert_skip_residuals.setToolTip(
            "dcm2niix splits a single input series into the real image plus "
            "derived single-volume duplicates it names ..._bolda, ..._Eq_1, "
            "etc. These have no valid BIDS suffix. Recommended: on."
        )
        form.addRow("Residuals:", self._convert_skip_residuals)

        self._convert_force_edf = QCheckBox(
            "Force EDF for EEG (re-encode recordings to EDF on convert)"
        )
        self._convert_force_edf.setToolTip(
            "Re-encode EEG recordings to EDF instead of keeping the "
            "source format. Harmonises a study to one BIDS-native format, and "
            "makes a non-BIDS-native but mne-readable source (GDF, EGI, ...) "
            "convertible. MEG / NIRS are unaffected."
        )
        form.addRow("Force EDF:", self._convert_force_edf)

        # Defacing, with its engine beside it. Off by default: it is
        # destructive, so it has to be chosen rather than discovered.
        self._convert_deface = QCheckBox(
            "Remove faces from anatomical and PET images"
        )
        self._convert_deface.setToolTip(
            "Blank the face before the subject is committed, so the "
            "identifiable image never enters the dataset at all. Off by "
            "default because it cannot be undone from the conversion: the "
            "original stays in your raw data, not in the BIDS tree. Needs "
            "niimath, which ships with BIDS Manager."
        )
        form.addRow("Deface:", self._convert_deface)

        self._convert_deface_engine = QComboBox()
        self._convert_deface_engine.setObjectName("ent-input")
        for eng in deface_engines.ENGINES:
            self._convert_deface_engine.addItem(eng.label, eng.id)
        self._convert_deface_engine.setToolTip(
            "\n\n".join(f"{e.label}: {e.description}" for e in deface_engines.ENGINES)
        )
        form.addRow("Deface engine:", self._convert_deface_engine)

        reason = deface_run.unavailable_reason()
        if reason:
            self._convert_deface.setEnabled(False)
            self._convert_deface.setChecked(False)
            self._convert_deface_engine.setEnabled(False)
            # Shown, not hidden: a missing row reads as "this tool cannot do
            # that", which is how somebody ships a dataset with faces in it.
            self._convert_deface.setToolTip(reason)
        else:
            self._convert_deface.toggled.connect(
                self._convert_deface_engine.setEnabled
            )
            self._convert_deface_engine.setEnabled(
                self._convert_deface.isChecked()
            )

        v.addWidget(convert)

        # Post-convert chain laid out as an indented hierarchy: each step is
        # a parent checkbox; its sub-options sit indented beneath and are
        # enabled only while the parent is on.
        post = QGroupBox("Post-convert chain (run after every conversion)")
        pv = QVBoxLayout(post)
        pv.setSpacing(4)

        self._post_run_metadata = QCheckBox(
            "Generate metadata (dataset_description, participants.tsv, "
            "*_scans.tsv, sidecar audit)"
        )
        pv.addWidget(self._post_run_metadata)
        self._post_metadata_fill_todos = QCheckBox(
            "Mark missing metadata with a placeholder"
        )
        self._post_metadata_fill_todos.setToolTip(
            "Writes a placeholder into every declared field the file does "
            "not carry, so the gap is visible in the file and reported by "
            "validation instead of being an absence nobody notices. Existing "
            "values are never overwritten."
        )
        pv.addWidget(_indented(self._post_metadata_fill_todos))

        # How much to mark. Separate from whether, because "mark the required
        # fields" and "mark everything the standard declares" are different
        # amounts of work and different amounts of noise.
        scope_row = QHBoxLayout()
        scope_row.setSpacing(8)
        scope_label = QLabel("Mark which fields:")
        self._metadata_fill_scope = QComboBox()
        for value, label in (
            ("required", "Required only"),
            ("recommended", "Required and recommended (default)"),
            ("optional", "Everything declared, including optional"),
        ):
            self._metadata_fill_scope.addItem(label, userData=value)
        self._metadata_fill_scope.setToolTip(
            "The scopes nest. A field whose type admits no honest marker (a "
            "number, a boolean, a controlled vocabulary) is left absent and "
            "reported rather than given a value nobody stated, so a wider "
            "scope never introduces a validation error.\n\n"
            "Used by the post-convert chain and by the Editor's Fix ups, so "
            "both do the same thing."
        )
        scope_row.addWidget(scope_label)
        scope_row.addWidget(self._metadata_fill_scope, 1)
        scope_holder = QWidget()
        scope_holder.setLayout(scope_row)
        pv.addWidget(_indented(scope_holder, indent=40))
        self._post_metadata_fill_todos.toggled.connect(
            self._metadata_fill_scope.setEnabled
        )

        # Dataset repairs. They run between metadata and validation, and they
        # are the same code the Editor's Fix ups button runs, so a dataset
        # gets the same result whichever moment the user chooses. Both are off
        # by default: one adds files and the other moves fields between them.
        self._post_fixup_companions = QCheckBox(
            "Generate missing companion files (events.tsv, channels.tsv, "
            "JSON sidecars)"
        )
        self._post_fixup_companions.setToolTip(
            "What can be read from a recording is read from it, so a "
            "channels table is real content. The rest is a stub carrying "
            "TODO rows.\n\nA generated events table is deliberately INVALID "
            "until you fill it in: TODO is not a valid onset, so validation "
            "reports an error for each one. That is the point. An empty but "
            "valid events table would be indistinguishable from a recording "
            "that genuinely had no events, and would pass quietly forever."
        )
        pv.addWidget(_indented(self._post_fixup_companions))
        self._post_fixup_citation = QCheckBox(
            "Write CITATION.cff from the dataset description"
        )
        self._post_fixup_citation.setToolTip(
            "Writes CITATION.cff from what dataset_description.json already "
            "says. This runs without asking, so here is exactly what it "
            "changes.\n\n"
            "MOVED: Authors. It is taken OUT of dataset_description.json, "
            "because stating authorship in both files is an error "
            "(AUTHORS_AND_CITATION_FILE_MUTUALLY_EXCLUSIVE).\n\n"
            "COPIED and KEPT: License, HowToAcknowledge and "
            "ReferencesAndLinks. They are written into the citation file and "
            "left where they are, so a value you typed does not disappear "
            "from the file you typed it into. The validator would rather "
            "each lived in one place only and says so as a warning "
            "(SINGLE_SOURCE_CITATION_FIELDS). That warning is the cost of "
            "not deleting your answer.\n\n"
            "An existing CITATION.cff is never overwritten."
        )
        # Said on the face of the setting too, not only on hover. This one
        # runs unattended at the end of a conversion, and a fix up that
        # removes a field a user typed cannot announce itself in a tooltip.
        pv.addWidget(_indented(self._post_fixup_citation))
        citation_note = QLabel(
            "Moves <b>Authors</b> out of dataset_description.json (stating it "
            "in both is an error). License, HowToAcknowledge and "
            "ReferencesAndLinks are copied and kept."
        )
        citation_note.setObjectName("dlg-hint")
        citation_note.setWordWrap(True)
        # Indented by margin rather than by ``_indented``, whose trailing
        # stretch would stop a wrapping label from using the width.
        citation_note.setContentsMargins(44, 0, 0, 4)
        pv.addWidget(citation_note)

        self._post_run_validate = QCheckBox(
            "Validate dataset (bidsval schema-driven validation)"
        )
        pv.addWidget(self._post_run_validate)
        self._post_validate_strict = QCheckBox(
            "Deep checks: read NIfTI headers and file contents (slower)"
        )
        self._post_validate_strict.setToolTip(
            "When on, validation reads NIfTI headers and file contents in "
            "addition to the structural checks. More thorough, slower on "
            "large trees. Maps to the validator's read-headers mode."
        )
        self._post_validate_html = QCheckBox(
            "Write a self-contained validation_report.html (--html)"
        )
        pv.addWidget(_indented(self._post_validate_strict))
        pv.addWidget(_indented(self._post_validate_html))

        _bind_children(
            self._post_run_metadata,
            self._post_metadata_fill_todos,
            self._post_fixup_companions,
            self._post_fixup_citation,
        )
        _bind_children(
            self._post_run_validate,
            self._post_validate_strict,
            self._post_validate_html,
        )

        v.addWidget(post)
        v.addStretch(1)
        return w

    def _build_bids_version_tab(self) -> QWidget:
        """Which version of the standard this session works to.

        Its own tab, and the first one, because it is not a validation
        preference. It decides which fields every metadata form asks for, which
        entities a filename may carry, what gets stamped into
        dataset_description.json and what validation reports. It used to sit
        under Validation, which is where it reached when validation was the only
        thing that read it.
        """
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("BIDS version")
        form = QFormLayout(box)

        self._validate_schema = QComboBox()
        self._validate_schema.addItem("Newest available (recommended)", userData="")
        for ver in schema.available_versions():
            self._validate_schema.addItem(f"BIDS {ver}", userData=ver)
        self._validate_schema.setToolTip(
            "The version of BIDS this session works to. Several ship with "
            "BIDS Manager.\n\nChoose an older one to work to a dataset that "
            "was built against it, so the forms ask for that version's fields "
            "and validation judges it by that version's rules."
        )
        self._validate_schema.currentIndexChanged.connect(self._describe_bids_version)
        form.addRow("Work to:", self._validate_schema)

        self._version_summary = QLabel()
        self._version_summary.setWordWrap(True)
        self._version_summary.setStyleSheet("color: #8b949e;")
        form.addRow("", self._version_summary)
        v.addWidget(box)

        note = QLabel(
            "Applies to the whole pipeline: what the metadata forms ask for, "
            "which entities a filename may carry, the BIDSVersion written into "
            "dataset_description.json, and what validation reports.\n\n"
            "The command line takes the same choice per run, as --schema."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8b949e;")
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _describe_bids_version(self) -> None:
        """Say what the highlighted version actually is, in its own terms.

        A version number alone does not tell anyone what changes. The counts do,
        and they come from the schema rather than from a note that would go
        stale.
        """
        chosen = self._validate_schema.currentData() or None
        try:
            namespace = schema.get_schema(chosen)
            self._version_summary.setText(
                f"BIDS {namespace.bids_version}: "
                f"{len(namespace.objects.datatypes)} datatypes, "
                f"{len(namespace.rules.entities)} entities, "
                f"{len(namespace.objects.metadata)} metadata fields."
            )
        except Exception:
            self._version_summary.setText("")

    def _build_validation_tab(self) -> QWidget:
        """Validation-engine (bidsval) knobs.

        The Editor's "Deep checks" toggle (read NIfTI headers / file contents)
        lives in the Editor toolbar, not here, because it is a per-run choice.
        These two are dataset-wide preferences worth persisting.
        """
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("Validation engine (bidsval)")
        form = QFormLayout(box)

        self._validate_max_rows = QSpinBox()
        self._validate_max_rows.setRange(1, 10_000_000)
        self._validate_max_rows.setSingleStep(1000)
        self._validate_max_rows.setToolTip(
            "Maximum number of rows scanned per TSV during column / value "
            "validation. Large tables are bounded to keep validation fast; "
            "raise this to scan more of very long tables."
        )
        form.addRow("Max TSV rows scanned:", self._validate_max_rows)

        # Which severities the Editor's Validation pane lists. The tree badges
        # and chips always reflect the full picture; this only filters the
        # findings list so you can focus on errors (or warnings).
        self._validate_show = QComboBox()
        for label, value in (
            ("Errors and warnings", "error_warning"),
            ("Errors only",         "error"),
            ("Warnings only",       "warning"),
        ):
            self._validate_show.addItem(label, userData=value)
        self._validate_show.setToolTip(
            "Filter which findings the Editor's Validation pane lists. The "
            "tree badges and the error / warning counters always show the "
            "full picture; this only narrows the list so you can focus."
        )
        form.addRow("Show findings:", self._validate_show)

        self._validate_flag_todos = QCheckBox(
            "Flag 'TODO' placeholder values as warnings"
        )
        self._validate_flag_todos.setToolTip(
            "A BIDS Manager convention: the metadata engine writes the literal "
            "string 'TODO' into missing recommended fields so you can find and "
            "fill them. On by default. Turn it off for exact parity with the "
            "standalone bidsval engine (which does not know this convention)."
        )
        form.addRow("TODO placeholders:", self._validate_flag_todos)

        v.addWidget(box)

        note = QLabel(
            "The Editor's \"Deep checks\" toggle (read NIfTI headers and file "
            "contents) lives in the Editor toolbar, since it is a per-run "
            "choice. Validation results are written into the project's "
            "<bids_root>/.bidsmgr/ folder, never into the BIDS tree."
        )
        note.setStyleSheet("color: #8b949e;")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _build_ai_tab(self) -> QWidget:
        """Whether to offer the AI agent, and what it should run with.

        The agent is BIDS-Manager's to start, so there is no URL here:
        whichever process spawns a service already knows its address, and
        a field that only disagrees when something else grabs port 8000
        is worse than no field at all. What is left is the state of the
        service, the interpreter to run it with, and the model knobs a
        normal user can act on.
        """
        w = QWidget()
        v = QVBoxLayout(w)

        box = QGroupBox("AI Agent")
        form = QFormLayout(box)

        self._ai_enabled = QCheckBox(
            "Show an “Ask AI” button on errors and warnings"
        )
        self._ai_enabled.setToolTip(
            "Adds a small button beside every error and warning the app "
            "shows. Clicking it sends that finding to the AI agent and "
            "opens a plain-language explanation of what is wrong, why, "
            "and how to fix it.\n\n"
            "On by default. While it is on, BIDS-Manager starts the AI "
            "agent itself when the app opens, and stops it when you turn "
            "this off."
        )
        form.addRow("", self._ai_enabled)

        # -- is it up? ------------------------------------------------
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self._ai_status = QLabel("")
        self._ai_status.setWordWrap(True)
        self._ai_start_btn = QPushButton("Start")
        self._ai_start_btn.setToolTip(
            "Bring the AI agent up now, or stop the one BIDS-Manager "
            "started.\n\n"
            "It is a separate process: importing torch and friends takes "
            "a few seconds, and the first answer also has to load a model."
        )
        self._ai_start_btn.clicked.connect(self._on_agent_start_stop)
        status_row.addWidget(self._ai_status, 1)
        status_row.addWidget(self._ai_start_btn, 0)
        form.addRow("Agent:", status_row)

        # -- how to run it --------------------------------------------
        python_row = QHBoxLayout()
        python_row.setSpacing(8)
        self._ai_python = QLineEdit()
        self._ai_python.setPlaceholderText("Automatic")
        self._ai_python.setToolTip(
            "Which Python launches the AI agent.\n\n"
            "Leave this empty to detect it: BIDS-Manager asks each "
            "candidate whether it can see flask, torch, transformers and "
            "laya, and takes the first one that can. The interpreter "
            "BIDS-Manager itself runs from usually cannot, because it "
            "does not carry a model runtime.\n\n"
            "Set it only if detection picks the wrong one."
        )
        python_btn = QPushButton("...")
        python_btn.setFixedWidth(28)
        python_btn.setToolTip("Browse for a Python interpreter.")
        python_btn.clicked.connect(self._browse_agent_python)
        python_row.addWidget(self._ai_python, 1)
        python_row.addWidget(python_btn, 0)
        form.addRow("Python:", python_row)

        # -- which model ----------------------------------------------
        self._ai_model = QComboBox()
        self._ai_model.setEditable(False)
        self._ai_model.setToolTip(
            "The local model that writes the answers.\n\n"
            "Populated from the agent's own model list; \"as configured "
            "on the agent\" means BIDS-Manager has no preference and the "
            "agent's config.json decides.\n\n"
            "The list is fetched when you open this tab and after "
            "<b>Test connection</b>. Downloading a model is the agent's "
            "job and can take a while the first time."
        )
        self._ai_model.addItem("— as configured on the agent —", "")
        form.addRow("Model:", self._ai_model)

        # -- how much it is allowed to write ---------------------------
        self._ai_tokens = QSpinBox()
        self._ai_tokens.setRange(16, 8192)
        self._ai_tokens.setSingleStep(32)
        self._ai_tokens.setToolTip(
            "The longest answer, in tokens (roughly 3/4 of a word each).\n\n"
            "250 is enough for a plain-language explanation. Higher is "
            "slower, and on CPU the difference is very noticeable; a "
            "follow-up question may need more room than the first."
        )
        form.addRow("Answer length:", self._ai_tokens)

        self._ai_quant = QComboBox()
        for value in AI_QUANTIZATIONS:
            self._ai_quant.addItem(_QUANT_LABELS[value], value)
        self._ai_quant.setToolTip(
            "How the model's weights are held while it runs.\n\n"
            "Full precision is exactly the model as published. 4-bit uses "
            "about a quarter of the memory and 8-bit about half, at some "
            "cost in answer quality. Both need bitsandbytes; without it "
            "the agent warns and loads full precision instead.\n\n"
            "Choose this to fit a smaller GPU (or to stop the model being "
            "moved to swap)."
        )
        form.addRow("Memory:", self._ai_quant)

        self._ai_device = QComboBox()
        for value in AI_DEVICE_MAPS:
            self._ai_device.addItem(_DEVICE_LABELS[value], value)
        self._ai_device.setToolTip(
            "Where the model runs.\n\n"
            "Automatic lets PyTorch pick. CPU only is the safe choice on "
            "a machine with no supported GPU, and CUDA is an NVIDIA card. "
            "Changing this forces the model to be reloaded on the next "
            "answer."
        )
        form.addRow("Device:", self._ai_device)

        self._ai_think = QCheckBox(
            "Think step by step before answering"
        )
        self._ai_think.setToolTip(
            "Ask the model to reason through the finding first when the "
            "model supports it (Qwen3 does; older ones ignore it).\n\n"
            "Better answers on involved problems, noticeably slower ones "
            "on simple ones."
        )
        form.addRow("", self._ai_think)

        self._ai_temperature = QDoubleSpinBox()
        self._ai_temperature.setRange(0.0, 2.0)
        self._ai_temperature.setSingleStep(0.05)
        self._ai_temperature.setDecimals(2)
        self._ai_temperature.setToolTip(
            "How varied the answers are.\n\n"
            "Lower is focused and the same every time; higher is more "
            "wide-ranging and occasionally less reliable. 0 means always "
            "give the single most likely answer.\n\n"
            "This is applied immediately — the agent re-reads it on the "
            "next question."
        )
        form.addRow("Answer variety:", self._ai_temperature)

        self._ai_timeout = QSpinBox()
        self._ai_timeout.setRange(5, 3600)
        self._ai_timeout.setSingleStep(15)
        self._ai_timeout.setSuffix(" s")
        self._ai_timeout.setToolTip(
            "How long to wait for an answer before giving up.\n\n"
            "The FIRST request loads the local model, which can take a "
            "minute or two on CPU. Raise this if answers time out and "
            "the agent's own console shows it is still working."
        )
        form.addRow("Request timeout:", self._ai_timeout)

        test_row = QHBoxLayout()
        test_row.setSpacing(8)
        test_btn = QPushButton("Test connection")
        test_btn.setToolTip(
            "Ask the agent for its model list and current configuration. "
            "Neither call loads the model, so this is quick even while "
            "the LLM is still cold."
        )
        test_btn.clicked.connect(self._test_agent_connection)
        self._ai_test_result = QLabel("")
        self._ai_test_result.setWordWrap(True)
        self._ai_test_result.setStyleSheet("color: #8b949e;")
        test_row.addWidget(test_btn)
        test_row.addWidget(self._ai_test_result, 1)
        form.addRow("Connection:", test_row)

        v.addWidget(box)

        note = QLabel(
            "BIDS-Manager starts the agent for you when it opens, on the "
            "Python detected above (or the one you chose). It is a "
            "separate local process: it loads a small model onto your "
            "machine, so the first answer of a session takes a moment.\n\n"
            "Asking sends the finding &mdash; rule id, field, message and "
            "file path &mdash; to that service; nothing leaves your "
            "machine. Its own README lists what it needs installed."
        )
        note.setStyleSheet("color: #8b949e;")
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)

        # The service lives on after this dialog is built, so its state
        # is polled rather than pushed: a status that only refreshed on
        # an event would go stale the moment a background start finished.
        self._ai_models_fetched = False
        self._ai_timer = QTimer(self)
        self._ai_timer.setInterval(400)
        self._ai_timer.timeout.connect(self._update_agent_status)
        self._ai_timer.start()
        self.finished.connect(self._ai_timer.stop)
        self._update_agent_status()
        return w

    # -- the agent process -------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        """Ask the agent for its model list the first time it is shown.

        Deliberately not done in ``__init__``: building Settings must
        cost nothing when the tab is never opened, and by the time it
        *is* opened the agent has usually had a few more seconds to come
        up than it had when the dialog was constructed.
        """
        if self._tabs.tabText(index) != "AI Agent":
            return
        # Retried until it works: opening the tab while the agent is
        # still importing torch is the likeliest moment for this to
        # fail, and a one-shot flag would leave the user staring at a
        # dropdown with nothing in it for the rest of the session.
        if self._ai_models_fetched:
            return
        self._refresh_agent_models()

    def _update_agent_status(self) -> None:
        """One poll of ``bidsmgr.agent_service`` into label + button."""
        svc = agent_service.service()
        state = svc.state
        if state == "running":
            owned = svc.owned
            self._ai_status.setText(
                "Running (started by BIDS-Manager)."
                if owned else "Running (started outside BIDS-Manager)."
            )
            self._ai_status.setStyleSheet("color: #3fb950;")
            self._ai_start_btn.setText("Stop")
            self._ai_start_btn.setEnabled(owned)
        elif state == "starting":
            self._ai_status.setText("Starting...")
            self._ai_status.setStyleSheet("color: #d29922;")
            self._ai_start_btn.setText("Start")
            self._ai_start_btn.setEnabled(False)
        elif state == "failed":
            # The first line only: the full message ends in a log tail
            # that belongs in the tooltip, not in a form row.
            first = (svc.detail or "Did not start.").splitlines()[0]
            self._ai_status.setText(first)
            self._ai_status.setStyleSheet("color: #f85149;")
            self._ai_start_btn.setText("Start")
            self._ai_start_btn.setEnabled(True)
        else:
            self._ai_status.setText("Not running.")
            self._ai_status.setStyleSheet("color: #8b949e;")
            self._ai_start_btn.setText("Start")
            self._ai_start_btn.setEnabled(True)
        self._ai_status.setToolTip(svc.detail)

    def _on_agent_start_stop(self) -> None:
        svc = agent_service.service()
        if svc.state == "running":
            svc.stop()
            self._update_agent_status()
            return
        # Read the interpreter straight from the box rather than from
        # settings: the user may have typed it two seconds ago and not
        # saved yet, and this is the button that acts on what they see.
        svc.start(python=self._ai_python.text().strip())
        self._update_agent_status()

    def _browse_agent_python(self) -> None:
        start = str(self._ai_python.text().strip() or os.path.expanduser("~"))
        filters = (
            "Python interpreter (*.exe);;All files (*)"
            if os.name == "nt"
            else "All files (*)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the Python that runs the AI agent",
            start if os.path.isfile(start) else "", filters,
        )
        if path:
            self._ai_python.setText(path)

    def _refresh_agent_models(self) -> None:
        """Fill the Model dropdown from the agent, and say where it is.

        Blocking and local-only: ``/models`` and ``/config`` load no
        model, and a refused connection on loopback is instant. Runs when
        the tab is opened and again after Test connection, so the list
        cannot drift away from what the agent actually offers.
        """
        from .ai_explainer import AgentClient, AgentError, agent_base_url

        url = agent_base_url()
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            client = AgentClient(base_url=url, timeout=8.0)
            info = client.ping()
        except AgentError as exc:
            self._ai_models_fetched = False
            self._ai_test_result.setStyleSheet("color: #f85149;")
            self._ai_test_result.setText(
                str(exc).splitlines()[0] or "Connection failed."
            )
            self._ai_test_result.setToolTip(str(exc))
            return
        except Exception as exc:                        # pragma: no cover
            self._ai_models_fetched = False
            self._ai_test_result.setStyleSheet("color: #f85149;")
            self._ai_test_result.setText(str(exc))
            self._ai_test_result.setToolTip(repr(exc))
            return
        finally:
            self.unsetCursor()

        self._ai_models_fetched = True
        models = info.get("models") if isinstance(info, dict) else None
        models = [m for m in (models or []) if isinstance(m, dict) and m.get("name")]

        # What the agent is actually running, which is also what we
        # should have selected when the user never chose anything.
        current = ""
        llm: dict = {}
        try:
            cfg = client.config()
            llm = cfg.get("llm") if isinstance(cfg, dict) else {}
            if not isinstance(llm, dict):
                llm = {}
            current = str(llm.get("model_name") or "").strip()
        except (AgentError, AttributeError, TypeError):
            current = ""

        # Show the knobs as they are rather than as they were stored. In
        # the ordinary flow this changes nothing — BIDS-Manager pushed
        # these values when it started the agent — but it keeps the tab
        # honest if somebody edited the agent's own config meanwhile,
        # instead of letting a save overwrite it with a guess.
        if llm:
            self._sync_agent_knobs(llm)

        chosen = self._ai_model.currentData() or ""
        self._ai_model.clear()
        self._ai_model.addItem("— as configured on the agent —", "")
        for entry in models:
            name = str(entry["name"])
            label = f"{name}  ·  {entry.get('description', '')}"
            if entry.get("size"):
                label += f"  ·  {entry['size']}"
            if entry.get("vram_gb"):
                label += f"  ·  {entry['vram_gb']}"
            self._ai_model.addItem(label, name)
        self._set_agent_model(chosen or current)

        count = len(models)
        self._ai_test_result.setStyleSheet("color: #3fb950;")
        self._ai_test_result.setText(
            f"Connected to {client.base_url}"
            + (f"  ·  {count} models" if count else "")
            + (f"  ·  {current}" if current else "")
            + "."
        )
        self._ai_test_result.setToolTip(
            f"Model running on the agent: {current}" if current else ""
        )

    def _sync_agent_knobs(self, llm: dict) -> None:
        """Show the knobs the way the agent has them, not the way we stored them.

        In the ordinary flow this changes nothing — BIDS-Manager pushes
        these same values when it starts the agent, so store and reality
        have just converged. It matters only when they have drifted: a
        hand-edited ``config.json``, or a value somebody else set while
        the app was open. Without this, opening Settings, pressing Save
        and never looking would quietly overwrite somebody else's edit
        with a guess.

        Silently tolerant: one malformed field out of five should leave
        the other four right, and a missing one should leave the widget
        as the user set it.
        """
        try:
            tokens = int(llm["max_new_tokens"])
        except (KeyError, TypeError, ValueError):
            tokens = -1
        if 16 <= tokens <= 8192:
            self._ai_tokens.setValue(tokens)

        try:
            temperature = float(llm["temperature"])
        except (KeyError, TypeError, ValueError):
            temperature = -1.0
        if 0.0 <= temperature <= 2.0:
            self._ai_temperature.setValue(temperature)

        for widget, key, vocabulary in (
            (self._ai_device, "device_map", AI_DEVICE_MAPS),
            (self._ai_quant, "quantization", AI_QUANTIZATIONS),
        ):
            value = str(llm.get(key) or "").strip().lower()
            # The agent's own config may predate quantization and only
            # carry the old boolean, which is still what it acts on.
            if key == "quantization" and not value and llm.get("load_in_4bit"):
                value = "4bit"
            if value in vocabulary:
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)

        if isinstance(llm.get("enable_thinking"), bool):
            self._ai_think.setChecked(llm["enable_thinking"])

    def _set_agent_model(self, name: str) -> None:
        """Select ``name``, adding it if the agent's list omits it."""
        name = (name or "").strip()
        index = self._ai_model.findData(name)
        if index < 0 and name:
            self._ai_model.addItem(name, name)
            index = self._ai_model.findData(name)
        self._ai_model.setCurrentIndex(index if index >= 0 else 0)

    def _test_agent_connection(self) -> None:
        """Ping the agent and reload the model list from it.

        Blocking, and capped at eight seconds: a refused connection on
        localhost is instant, and a wrong address is the case the cap
        exists for. A spinner for one button press would cost more code
        than the wait costs patience.
        """
        self._refresh_agent_models()

    # ------------------------------------------------------------------
    # Widget <-> settings
    # ------------------------------------------------------------------

    @classmethod
    def _header_logo_index(cls, value: str) -> int:
        for i, (_label, key) in enumerate(cls._HEADER_LOGO_PRESETS):
            if key == value:
                return i
        return 0

    @classmethod
    def _closest_font_scale_index(cls, value: float) -> int:
        """Return the preset index whose multiplier is nearest *value*."""
        try:
            return min(
                range(len(cls._FONT_SCALE_PRESETS)),
                key=lambda i: abs(cls._FONT_SCALE_PRESETS[i][1] - value),
            )
        except Exception:
            return 1  # "Normal"

    def _load_into_widgets(self, s: AppSettings) -> None:
        """Push every value from ``s`` into the dialog's widgets.

        Used both on open (with the live settings) and by "Restore
        defaults" (with a fresh ``AppSettings()``), so the two paths can
        never drift. Worker counts are clamped to the detected thread cap.
        """
        cap = self._sys.logical_threads

        self._theme_combo.setCurrentText(s.theme)
        self._editor_show_hidden.setChecked(s.editor_show_hidden)
        self._editor_autosave.setChecked(s.editor_autosave)
        self._font_scale_combo.setCurrentIndex(
            self._closest_font_scale_index(s.font_scale)
        )
        self._header_logo_combo.setCurrentIndex(
            self._header_logo_index(s.header_logo)
        )

        self._scan_jobs.setValue(max(1, min(s.scan_n_jobs, cap)))
        self._scan_probe.setChecked(s.scan_probe_convert)
        self._scan_preview.setChecked(s.scan_converter_preview)
        self._scan_skip_bids_guess.setChecked(s.scan_skip_bids_guess)
        for entity, box in self._index_widths.items():
            box.setValue(int(s.scan_index_widths.get(entity, 0) or 0))

        self._convert_jobs.setValue(max(1, min(s.convert_n_jobs, cap)))
        idx = self._convert_on_existing.findData(s.convert_on_existing)
        self._convert_on_existing.setCurrentIndex(idx if idx >= 0 else 0)
        self._convert_skip_residuals.setChecked(s.convert_skip_residuals)
        self._convert_preserve_curation.setChecked(
            s.convert_preserve_curation
        )
        self._convert_force_edf.setChecked(s.convert_force_edf)
        if self._convert_deface.isEnabled():
            self._convert_deface.setChecked(s.convert_deface)
        idx = self._convert_deface_engine.findData(s.convert_deface_engine)
        if idx >= 0:
            self._convert_deface_engine.setCurrentIndex(idx)
        self._convert_deface_engine.setEnabled(
            self._convert_deface.isChecked()
            and self._convert_deface.isEnabled()
        )

        self._post_run_metadata.setChecked(s.post_run_metadata)
        self._post_metadata_fill_todos.setChecked(s.post_metadata_fill_todos)
        idx = self._metadata_fill_scope.findData(s.metadata_fill_scope)
        self._metadata_fill_scope.setCurrentIndex(idx if idx >= 0 else 1)
        self._metadata_fill_scope.setEnabled(s.post_metadata_fill_todos)
        self._post_fixup_companions.setChecked(s.post_fixup_companions)
        self._post_fixup_citation.setChecked(s.post_fixup_citation)
        self._post_run_validate.setChecked(s.post_run_validate)
        self._post_validate_strict.setChecked(s.post_validate_strict)
        self._post_validate_html.setChecked(s.post_validate_html)

        # BIDS version, then the validation engine's own knobs.
        sidx = self._validate_schema.findData(s.validate_schema_version)
        self._validate_schema.setCurrentIndex(sidx if sidx >= 0 else 0)
        self._describe_bids_version()
        self._validate_max_rows.setValue(max(1, int(s.validate_max_rows)))
        shidx = self._validate_show.findData(s.validate_show)
        self._validate_show.setCurrentIndex(shidx if shidx >= 0 else 0)
        self._validate_flag_todos.setChecked(s.validate_flag_todos)

        # AI agent. The result label is cleared rather than reloaded:
        # a "Connected" from the last open says nothing about now.
        self._ai_enabled.setChecked(s.ai_enabled)
        self._ai_timeout.setValue(max(5, int(s.ai_timeout)))
        self._ai_python.setText(s.ai_python)
        self._set_agent_model(s.ai_model)
        self._ai_tokens.setValue(max(16, int(s.ai_max_new_tokens)))
        self._ai_device.setCurrentIndex(
            max(0, self._ai_device.findData(s.ai_device_map))
        )
        self._ai_quant.setCurrentIndex(
            max(0, self._ai_quant.findData(s.ai_quantization))
        )
        self._ai_think.setChecked(s.ai_enable_thinking)
        self._ai_temperature.setValue(float(s.ai_temperature))
        self._ai_test_result.setText("")
        self._ai_test_result.setStyleSheet("color: #8b949e;")
        self._ai_test_result.setToolTip("")

        # Scan rules: rebuild both editable tables from the persisted lists
        # (clear first so Restore-defaults empties them).
        self._excl_table.setRowCount(0)
        for e in s.scan_exclusions:
            self._add_exclusion_row(
                pattern=str(e.get("pattern", "")),
                target=str(e.get("target", "sequence")),
                mode=str(e.get("match_mode", "substring")),
            )
        self._hint_table.setRowCount(0)
        for h in s.user_hints:
            pats = h.get("patterns", [])
            if isinstance(pats, str):
                pats = [pats]
            self._add_hint_row(
                patterns=", ".join(str(p) for p in pats),
                datatype=str(h.get("datatype", "")),
                suffix=str(h.get("suffix", "")),
                task=str(h.get("task", "") or ""),
                mode=str(h.get("match_mode", "substring")),
                force=bool(h.get("force", False)),
            )

    def _on_restore_defaults(self) -> None:
        """Reset all widgets to the AppSettings field defaults (not saved
        until the user clicks Save)."""
        self._load_into_widgets(AppSettings())

    def _on_save(self) -> None:
        # Validate the scan rules first so an invalid hint blocks the save
        # without losing the user's other edits.
        hints, exclusions, error = self._read_scan_rules()
        if error:
            QMessageBox.warning(self, "Invalid scan rule", error)
            return

        s = self._settings
        s.user_hints = hints
        s.scan_exclusions = exclusions
        s.theme = self._theme_combo.currentText()
        s.editor_show_hidden = self._editor_show_hidden.isChecked()
        s.editor_autosave = self._editor_autosave.isChecked()
        s.font_scale = self._FONT_SCALE_PRESETS[
            self._font_scale_combo.currentIndex()
        ][1]
        s.header_logo = self._HEADER_LOGO_PRESETS[
            self._header_logo_combo.currentIndex()
        ][1]

        s.scan_n_jobs = self._scan_jobs.value()
        s.scan_probe_convert = self._scan_probe.isChecked()
        s.scan_converter_preview = self._scan_preview.isChecked()
        s.scan_skip_bids_guess = self._scan_skip_bids_guess.isChecked()
        # Zero means "as found", so it is absent rather than stored as 0:
        # the scan pass treats an empty map as nothing to do.
        s.scan_index_widths = {
            entity: box.value()
            for entity, box in self._index_widths.items() if box.value()
        }

        s.convert_n_jobs = self._convert_jobs.value()
        s.convert_on_existing = self._convert_on_existing.currentData() or "skip"
        # Keep the legacy flag in sync for any old reader.
        s.convert_overwrite = (s.convert_on_existing == "replace")
        s.convert_skip_residuals = self._convert_skip_residuals.isChecked()
        s.convert_preserve_curation = (
            self._convert_preserve_curation.isChecked()
        )
        s.convert_force_edf = self._convert_force_edf.isChecked()
        s.convert_deface = self._convert_deface.isChecked()
        s.convert_deface_engine = (
            self._convert_deface_engine.currentData()
            or s.convert_deface_engine
        )

        s.post_run_metadata = self._post_run_metadata.isChecked()
        s.post_metadata_fill_todos = self._post_metadata_fill_todos.isChecked()
        s.metadata_fill_scope = (
            self._metadata_fill_scope.currentData() or "recommended"
        )
        s.post_fixup_companions = self._post_fixup_companions.isChecked()
        s.post_fixup_citation = self._post_fixup_citation.isChecked()
        s.post_run_validate = self._post_run_validate.isChecked()
        s.post_validate_strict = self._post_validate_strict.isChecked()
        s.post_validate_html = self._post_validate_html.isChecked()

        s.validate_schema_version = self._validate_schema.currentData() or ""
        s.validate_max_rows = self._validate_max_rows.value()
        s.validate_show = self._validate_show.currentData() or "error_warning"
        s.validate_flag_todos = self._validate_flag_todos.isChecked()

        s.ai_enabled = self._ai_enabled.isChecked()
        s.ai_timeout = self._ai_timeout.value()
        # ai_base_url deliberately has no widget: BIDS-Manager starts the
        # agent itself, so the only address worth trusting is the one it
        # just bound, and a saved URL is a stale one by construction.
        s.ai_python = self._ai_python.text().strip()
        s.ai_model = (self._ai_model.currentData() or "").strip()
        s.ai_max_new_tokens = int(self._ai_tokens.value())
        s.ai_device_map = self._ai_device.currentData() or "auto"
        s.ai_quantization = self._ai_quant.currentData() or "none"
        s.ai_enable_thinking = self._ai_think.isChecked()
        s.ai_temperature = float(self._ai_temperature.value())

        s.save()
        # Bring the service in line with what was just saved. Both calls
        # are quick and local: start() returns immediately, and the push
        # is skipped unless an agent is actually up to receive it (if it
        # is down, main.py pushes the same payload when it next starts).
        svc = agent_service.service()
        if s.ai_enabled and svc.state in ("stopped", "failed"):
            svc.start(python=s.ai_python)
        elif not s.ai_enabled and svc.owned:
            svc.stop()
        if svc.state == "running":
            from .ai_explainer import push_llm_config
            push_llm_config(s)

        # Adopt the chosen version now rather than at the next launch: every
        # schema answer in the process is memoised, so this also drops the
        # answers about the old one.
        schema.set_active_version(s.validate_schema_version)
        self.accept()


__all__ = ["SettingsDialog"]
