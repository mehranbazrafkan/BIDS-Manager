"""Compress or decompress a dataset's NIfTI images, and what names them.

Over :mod:`bidsmgr.editor.compression`. Two questions and the answer
(GUI_LOOK_AND_FEEL 4.34, 4.37): which way (``.nii`` to ``.nii.gz`` or back)
and where (what was picked in the tree, the dataset, a subject, a session, a
datatype); then every image that would change, nested as the dataset is, with
the files that name them under "Follows automatically" and the ones left as
they are with why. Headline tiles say how many images, their size now and
after, and the free space the new ones need while the old ones are kept for
Undo.

The plan is made on a thread, again after every choice; the images are
re-encoded on another, with Stop, and nothing in the dataset changes until all
of them are ready. The change is one step of the Editor's history.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..editor import compression as engine
from .dialog_chrome import build_footer_with, build_header, card, hint, stat
from .fs_watch import watchers_released
from .widgets.move_preview import MovePreviewTree
from .widgets.primitives import Segmented
from .widgets.scope_bar import ScopeBar
from .widgets.spinner import BusySpinner
from .widgets.tree_click import toggle_on_click
from .window_flags import give_window_buttons

log = logging.getLogger(__name__)

#: How long the choices settle before the dataset is read again.
_SETTLE_MS = 250

_DIRECTIONS = (
    ("decompress", "Decompress to .nii",
     "Write each .nii.gz image as .nii: the same image, uncompressed. It opens "
     "faster and some tools (SPM, older pipelines) read only .nii; it takes "
     "about twice the space or more."),
    ("compress", "Compress to .nii.gz",
     "Write each .nii image as .nii.gz: the same image, gzip-compressed. It "
     "usually takes less than half the space; reading it costs a moment of "
     "decompression."),
)

_WHAT = {
    "scans.tsv filename": "the filename column follows",
    "saved scene": "the saved view follows",
    ".bidsignore": "the line follows",
}


def human(size: Optional[int]) -> str:
    """``1536`` -> ``1.5 KB``: sizes the way a file manager says them."""
    if size is None:
        return "…"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _count(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def plan_for(root: Path, compress: Optional[bool], targets: list[Path]):
    """The plan; with ``compress`` ``None``, in the direction most of the
    images in ``targets`` can go (decompress on a tie)."""
    if compress is None:
        images = engine.find_images(root, targets)
        gz = sum(1 for p in images if p.name.endswith(engine.GZ))
        compress = gz < len(images) - gz
    return engine.plan_compression(root, compress, targets)


class CompressionDialog(QDialog):
    """Plan, show, then compress or decompress the ticked images."""

    def __init__(self, root: Path, targets: Optional[list] = None,
                 parent: Optional[QWidget] = None, *,
                 compress: Optional[bool] = None) -> None:
        super().__init__(parent)
        give_window_buttons(self)
        self._root = Path(root)
        self._plan: Optional[engine.CompressionPlan] = None
        self._outcome: Optional[engine.CompressionOutcome] = None
        self._generation = 0
        self._planning = False
        self._worker = None
        self._watchers = None
        #: The direction to plan in; ``None`` until the first plan chose it.
        self._compress = compress
        picked = [Path(t) for t in (targets or [])]
        from .viz.bridge import JobRunner

        self._jobs = JobRunner(self)
        self._jobs.done.connect(self._on_planned)
        self._jobs.failed.connect(self._on_plan_failed)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._replan)

        self.setWindowTitle("Compress or decompress images")
        self.setModal(True)
        self.resize(1040, 800)
        self.setMinimumSize(640, 480)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Compress or decompress images",
            "A NIfTI image is stored compressed (<b>.nii.gz</b>) or not (<b>.nii</b>): "
            "the same image either way, and both are valid BIDS. This writes the images "
            "here in the other form, and every file that names them follows: the scans "
            "tables, IntendedFor and the other sidecar fields that point at a file, "
            "saved viewer scenes and .bidsignore. <b>Nothing changes until you press "
            "the button</b>, and the change is one step of the Editor's history.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)
        bl.addWidget(self._build_choice(picked))
        bl.addWidget(self._build_result(), 1)

        # Progress, and Stop, only while a run owns the dataset (an empty bar
        # reads as something broken).
        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self._progress = QProgressBar()
        self._progress.setObjectName("run-progress")
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        run_row.addWidget(self._progress, 1)
        self._stop = QPushButton("Stop")
        self._stop.setObjectName("tb-btn")
        self._stop.setVisible(False)
        self._stop.setToolTip(
            "Stop after the images being written. Nothing in the dataset has changed "
            "until every image is ready, so stopping leaves it as it was.")
        self._stop.clicked.connect(self._on_stop)
        run_row.addWidget(self._stop)
        bl.addLayout(run_row)
        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Ok)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setObjectName("tb-btn-primary")
        self._ok.setText("Decompress" if not compress else "Compress")
        self._ok.setEnabled(False)
        buttons.accepted.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer_with(self._status, buttons))

        self._replan()

    # -- building -----------------------------------------------------------

    def _build_choice(self, picked: list[Path]) -> QWidget:
        frame, layout = card("Change")
        row = QHBoxLayout()
        row.setSpacing(12)
        self._direction = Segmented(
            _DIRECTIONS, current="compress" if self._compress else "decompress")
        self._direction.changed.connect(self._on_direction)
        row.addWidget(self._direction)
        self._direction_hint = QLabel("")
        self._direction_hint.setObjectName("dlg-hint")
        self._direction_hint.setWordWrap(True)
        row.addWidget(self._direction_hint, 1)
        layout.addLayout(row)

        where = QLabel("Where")
        where.setObjectName("dlg-section-title")
        layout.addSpacing(2)
        layout.addWidget(where)
        self._scope_bar = ScopeBar(self._root, picked, show_summary=False, parent=frame)
        self._scope_bar.changed.connect(self._schedule)
        layout.addWidget(self._scope_bar)
        layout.addWidget(hint(
            "A folder stands for every image in it. derivatives/ and sourcedata/ are "
            "left as their pipelines wrote them."))
        self._show_direction_hint()
        return frame

    def _build_result(self) -> QWidget:
        frame, layout = card("What would change")
        stats = QHBoxLayout()
        stats.setSpacing(28)
        self._stat_images = stat("images", "…")
        self._stat_now = stat("now", "…")
        self._stat_after = stat("after", "…")
        self._stat_refs = stat("files naming them", "…")
        self._stat_free = stat("free on the disk", "…")
        for tile in (self._stat_images, self._stat_now, self._stat_after,
                     self._stat_refs, self._stat_free):
            stats.addWidget(tile)
        stats.addStretch(1)
        layout.addLayout(stats)

        # Where the images will appear while they are found (rule 4.14),
        # hidden once they are, so it leaves no gap.
        self._busy = QWidget()
        busy = QHBoxLayout(self._busy)
        busy.setContentsMargins(0, 0, 0, 0)
        busy.setSpacing(8)
        self._spinner = BusySpinner()
        busy.addWidget(self._spinner)
        self._busy_label = QLabel("")
        self._busy_label.setObjectName("dlg-hint")
        busy.addWidget(self._busy_label, 1)
        layout.addWidget(self._busy)

        self._tree = MovePreviewTree()
        self._tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._tree.itemChanged.connect(lambda *_a: self._update_status())
        toggle_on_click(self._tree)
        layout.addWidget(self._tree, 1)

        picks = QHBoxLayout()
        picks.setSpacing(6)
        for text, state in (("All", Qt.CheckState.Checked), ("None", Qt.CheckState.Unchecked)):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.clicked.connect(lambda _c=False, s=state: self._set_all(s))
            picks.addWidget(btn)
        picks.addSpacing(8)
        picks.addWidget(hint(
            "The images replaced are moved, not copied, into the dataset's edit history "
            "(.bidsmgr/editor/), where Undo finds them."), 1)
        layout.addLayout(picks)
        return frame

    # -- choosing -------------------------------------------------------------

    def _show_direction_hint(self) -> None:
        key = self._direction.current()
        self._direction_hint.setText(next(tip for k, _t, tip in _DIRECTIONS if k == key))

    def _on_direction(self, key: str) -> None:
        self._compress = key == "compress"
        self._show_direction_hint()
        self._schedule()

    def _schedule(self) -> None:
        # The plan on screen no longer answers the choice: nothing may be
        # applied from it while the new one settles.
        self._ok.setEnabled(False)
        self._settle.start()

    def _set_all(self, state: Qt.CheckState) -> None:
        self._tree.set_all(state)
        self._update_status()

    # -- planning ---------------------------------------------------------------

    def planning(self) -> bool:
        """Whether a plan is being made or is about to be (a choice settling)."""
        return self._planning or self._settle.isActive()

    def _replan(self) -> None:
        self._settle.stop()
        self._generation += 1
        self._planning = True
        self._plan = None
        self._tree.clear()
        self._tree.setEnabled(False)
        self._ok.setEnabled(False)
        for tile in (self._stat_images, self._stat_now, self._stat_after,
                     self._stat_refs, self._stat_free):
            tile.number.setText("…")
        self._set_tone(self._stat_free, "text")
        self._busy.setVisible(True)
        self._spinner.set_busy(True, message="")
        self._busy_label.setText("Reading the images and what names them…")
        self._status.setText("")
        self._jobs.start("plan", self._generation, plan_for, self._root, self._compress,
                         self._scope_bar.targets())

    def _on_plan_failed(self, _tag: str, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._planning = False
        self._spinner.set_busy(False, message="")
        self._busy.setVisible(False)
        log.warning("compression plan failed: %s", message)
        self._status.setText(
            f"Could not read the dataset: {message.strip().splitlines()[-1]}")

    def _on_planned(self, _tag: str, generation: int, plan) -> None:
        if generation != self._generation:
            return
        self._planning = False
        self._spinner.set_busy(False, message="")
        self._busy.setVisible(False)
        self._plan = plan
        if self._compress is None or self._compress != plan.compress:
            # The first plan chose the direction: show the choice it made.
            self._compress = plan.compress
            self._direction.set_current("compress" if plan.compress else "decompress")
            self._show_direction_hint()
        self._fill()

    def _fill(self) -> None:
        plan = self._plan
        assert plan is not None
        form = ".nii.gz" if plan.compress else ".nii"
        moves = [(c.rel, c.source, c.target) for c in plan.changes]
        extras = []
        for ref in plan.references:
            n = len(ref.images)
            what = _WHAT.get(ref.what, f"{ref.what} follows")
            extras.append((ref.rel, f"{what} ({_count(n, 'image')})"))
        left = list(plan.skipped)
        if plan.already:
            left.append((_count(len(plan.already), "image"), f"already {form}"))
        self._tree.show_moves(self._root, moves, extras=extras, left=left)
        self._tree.setEnabled(True)
        self._update_status()

    # -- reading the choice back ----------------------------------------------

    def _chosen(self) -> list[str]:
        if self._plan is None:
            return []
        keys = self._tree.selected_keys()
        return [c.rel for c in self._plan.changes if c.rel in keys]

    @staticmethod
    def _set_tone(tile: QWidget, tone: str) -> None:
        name = f"stat-value-{tone}"
        if tile.number.objectName() != name:
            tile.number.setObjectName(name)
            tile.number.style().unpolish(tile.number)
            tile.number.style().polish(tile.number)

    def _update_status(self) -> None:
        if self.planning() or self._plan is None:
            return
        plan = self._plan
        chosen = self._chosen()
        verb = "Compress" if plan.compress else "Decompress"
        self._stat_images.number.setText(f"{len(chosen):,}")
        self._stat_now.number.setText(human(plan.bytes_now(chosen)))
        self._stat_after.number.setText(
            "smaller" if plan.compress and chosen else human(plan.bytes_needed(chosen)))
        self._stat_refs.number.setText(f"{len(plan.references_for(chosen)):,}")
        self._stat_free.number.setText(human(plan.free_bytes))
        short = plan.short_of_space(chosen)
        self._set_tone(self._stat_free, "err" if short else "text")

        if not plan.changes:
            form = ".nii.gz" if plan.compress else ".nii"
            self._status.setText(
                f"Every image here is already {form}." if plan.already and not plan.skipped
                else "No image here can be changed; the reasons are listed above."
                if plan.skipped else "There is no NIfTI image here.")
            self._ok.setEnabled(False)
            self._ok.setText(verb)
            return
        if not chosen:
            self._status.setText("No image ticked.")
            self._ok.setEnabled(False)
            self._ok.setText(verb)
            return
        refs = plan.references_for(chosen)
        if short:
            self._status.setText(
                f"Not enough free space: the new images need {human(plan.bytes_needed(chosen))}"
                f" while the old ones are kept for Undo, and the disk has "
                f"{human(plan.free_bytes)} free.")
            self._ok.setEnabled(False)
        else:
            size = (f"{human(plan.bytes_now(chosen))} becomes less"
                    if plan.compress else
                    f"{human(plan.bytes_now(chosen))} becomes "
                    f"{human(plan.bytes_needed(chosen))}")
            follow = (f"; {_count(len(refs), 'file')} naming them "
                      f"follow{'s' if len(refs) == 1 else ''}" if refs else "")
            self._status.setText(f"{_count(len(chosen), 'image')}: {size}{follow}.")
            self._ok.setEnabled(True)
        self._ok.setText(f"{verb} {_count(len(chosen), 'image')}")

    # -- running it, off the GUI thread ------------------------------------------

    def _on_apply(self) -> None:
        chosen = self._chosen()
        if not chosen or self.planning() or self._plan is None or self._worker is not None:
            return
        from ..workers import CompressionWorker

        self._set_running(True, len(chosen))
        # Images are replaced: the watchers let go first (on Windows a watched
        # file cannot be moved). Held for the run, released when it ends.
        self._watchers = watchers_released()
        self._watchers.__enter__()
        worker = CompressionWorker(self._root, self._plan, only=chosen, parent=self)
        worker.progress.connect(self._on_progress)
        worker.finished_with_result.connect(self._on_done)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _set_running(self, running: bool, total: int = 0) -> None:
        self._progress.setVisible(running)
        self._stop.setVisible(running)
        self._stop.setEnabled(running)
        if running:
            self._progress.setRange(0, max(total, 1))
            self._progress.setValue(0)
            verb = "Compressing" if self._compress else "Decompressing"
            self._status.setText(f"{verb} {_count(total, 'image')}…")
        self._ok.setEnabled(not running)
        for widget in (self._tree, self._direction, self._scope_bar):
            widget.setEnabled(not running)

    def _on_progress(self, done: int, total: int, rel: str) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        if rel:
            verb = "Compressing" if self._compress else "Decompressing"
            self._status.setText(f"{verb}: {done:,} of {total:,} ready ({Path(rel).name})…")
        else:
            self._status.setText("Updating the files that name them…")

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._stop.setEnabled(False)
            self._status.setText("Stopping…")
            self._worker.cancel()

    def _release(self) -> None:
        watchers, self._watchers = self._watchers, None
        if watchers is not None:
            watchers.__exit__(None, None, None)

    def _on_worker_failed(self, tb: str) -> None:
        self._release()
        self._worker = None
        self._set_running(False)
        self._update_status()
        QMessageBox.warning(
            self, "Could not change the images",
            "It stopped on an unexpected error. Nothing was changed.\n\n"
            f"{tb.strip().splitlines()[-1]}")

    def _on_done(self, outcome) -> None:
        self._release()
        self._worker = None
        self._set_running(False)
        self._outcome = outcome
        if outcome.cancelled:
            self._update_status()
            self._status.setText("Stopped. Nothing was changed.")
            return
        if outcome.failed:
            where, why = outcome.failed[0]
            self._update_status()
            QMessageBox.warning(
                self, "Could not change the images",
                f"{where}:\n\n{why}\n\nNothing was changed: the whole step was put back.")
            return
        self.accept()

    def _stop_worker(self) -> None:
        """Stop a run and wait for its thread before anything is destroyed (a
        running QThread destroyed with its parent aborts the process)."""
        worker, self._worker = self._worker, None
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(60_000)
        self._release()

    def done(self, result: int) -> None:  # noqa: D102 - Qt signature
        self._stop_worker()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        self._stop_worker()
        super().closeEvent(event)

    def outcome(self) -> Optional[engine.CompressionOutcome]:
        """What the run did, once accepted. ``None`` before it has run."""
        return self._outcome

    def renamed(self) -> dict[Path, Path]:
        """Each changed image's old path and new one, once accepted."""
        if self._outcome is None or self._plan is None:
            return {}
        done = set(self._outcome.changed)
        return {c.source: c.target for c in self._plan.changes if c.rel in done}


__all__ = ["CompressionDialog", "human", "plan_for"]
