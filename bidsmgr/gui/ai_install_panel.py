"""The first-run ask: install the AI Agent module?

The agent is a separate program with its own heavy dependencies (torch,
transformers, laya), so a fresh BIDS-Manager does not include it. When
the AI Agent page is opened and the module is missing, this panel asks
whether to install it, and — on yes — clones the agent's repository,
installs its dependencies into BIDS-Manager's own Python environment,
showing every step as the installer produces it. The actual work lives
in :mod:`bidsmgr.agent_installer`; this panel only asks, points a worker
thread at it, and reports.

Installation takes minutes (torch is gigabytes), so it runs on a
worker thread and the page stays responsive; the spinner and the step
line say where it has got to. When it finishes, the panel hands the
interpreter the agent now runs on to the dialog, which starts the agent
with it.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import agent_installer
from .dialog_chrome import WrapLabel, card
from .widgets.spinner import BusySpinner

#: Installer threads that outlive the dialog that started them. A closed
#: dialog must not leave a worker signalling into a dead page — the
#: pattern the Ask AI dialog uses for its own workers.
_ORPHANS: set[QThread] = set()


def _retire(thread: QThread) -> None:
    _ORPHANS.discard(thread)
    thread.deleteLater()


class _AgentInstallWorker(QThread):
    """The installer on a worker thread; one step per signal."""

    progressed = pyqtSignal(str)
    failed = pyqtSignal(str)
    succeeded = pyqtSignal(str)          # the interpreter to run it with

    def run(self) -> None:
        try:
            python = agent_installer.full_install(emit=self.progressed.emit)
        except agent_installer.InstallError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:          # pragma: no cover
            self.failed.emit(f"Could not install the AI Agent module: {exc}")
        else:
            self.succeeded.emit(str(python))


class AiInstallPanel(QWidget):
    """The question the AI Agent page asks when the module is missing.

    Visible whenever the module is not installed — which on a first run
    is exactly the state of the page — with Install and Not now. "Not
    now" collapses it to one line so the page is not dominated by a
    question the user has already declined; the install button stays.
    """

    #: The interpreter the installer built, ready to start the agent with.
    installed = pyqtSignal(str)
    failed = pyqtSignal(str)
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        frame, lay = card("Install the AI Agent module")

        self._question = WrapLabel(
            "The AI Agent module is not part of this copy of BIDS-Manager — "
            "it lives in its own repository and is not installed yet.\n\n"
            "Install it now? BIDS-Manager will clone the agent's GitHub "
            "repository and install its dependencies (PyTorch and friends, "
            "several gigabytes) into the Python that runs BIDS-Manager. "
            "This runs once and needs an internet connection; models are "
            "downloaded separately, only when you first use one."
        )
        lay.addWidget(self._question)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self._install_btn = QPushButton("Install AI Agent module")
        self._install_btn.setToolTip(
            "Clone the agent's repository and install its dependencies into "
            "BIDS-Manager's Python environment. This can take several minutes."
        )
        self._install_btn.clicked.connect(self._on_install)
        self._later_btn = QPushButton("Not now")
        self._later_btn.setToolTip(
            "Leave the module uninstalled. The AI Agent page keeps a "
            "one-line install button; the agent cannot start until the "
            "module is installed."
        )
        self._later_btn.clicked.connect(self._on_later)
        buttons.addWidget(self._install_btn)
        buttons.addWidget(self._later_btn)
        buttons.addStretch(1)
        lay.addLayout(buttons)

        # Progress: the spinner says it is working, the step line says
        # where it has got to, and the installer's own output lands in
        # the tooltip (a form row cannot hold a pip log).
        self._spinner = BusySpinner()
        self._spinner.set_busy(False)
        self._step = WrapLabel("")
        self._step.setStyleSheet("color: #8b949e;")
        self._step.hide()
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)
        progress_row.addWidget(self._spinner, 0)
        progress_row.addWidget(self._step, 1)
        lay.addLayout(progress_row)

        self._error = WrapLabel("")
        self._error.setStyleSheet("color: #f85149;")
        self._error.hide()
        lay.addWidget(self._error)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        self._worker: _AgentInstallWorker | None = None

    # -- behaviour ---------------------------------------------------

    def is_running(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        try:
            return worker.isRunning()
        except RuntimeError:
            # Retired between the signal and this check: the C++ object
            # is gone, and a thread that no longer exists is not running.
            return False

    def _on_install(self) -> None:
        if self.is_running():
            return
        self._error.hide()
        self._install_btn.setEnabled(False)
        self._later_btn.setEnabled(False)
        self._spinner.set_busy(True, message="Installing...")
        self._step.setText("Preparing the installation ...")
        self._step.show()
        worker = _AgentInstallWorker()
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_success)
        worker.failed.connect(self._on_failed)
        # Bound to the thread itself, not to self: the panel may be
        # deleted before the install ends, and the retire still has to
        # run when the signal fires.
        worker.finished.connect(lambda w=worker: _retire(w))
        # This one is a bound method, so Qt drops it if the panel is
        # deleted first — and while the panel lives it forgets the
        # retired thread, which the next click would otherwise reach.
        worker.finished.connect(self._forget_worker)
        _ORPHANS.add(worker)
        self._worker = worker
        worker.start()

    def _forget_worker(self) -> None:
        """The install ended: drop the reference the next click checks."""
        self._worker = None

    def _on_later(self) -> None:
        # Collapse to one line: the question has been answered, and the
        # page should not nag — but the agent cannot start without the
        # module, so the install button stays reachable.
        self._question.hide()
        self._later_btn.hide()
        self.dismissed.emit()

    def _on_progress(self, text: str) -> None:
        self._step.setText(text)
        self._step.setToolTip(text)

    def _on_success(self, python: str) -> None:
        self._spinner.set_busy(False)
        self._step.setText(f"Installed. The agent runs on {python}")
        self.installed.emit(python)

    def _on_failed(self, message: str) -> None:
        self._spinner.set_busy(False)
        self._step.hide()
        self._error.setText(message)
        self._error.show()
        self._install_btn.setEnabled(True)
        self._later_btn.setEnabled(True)
        self.failed.emit(message)


__all__ = ["AiInstallPanel"]
