"""The first-run install panel: asks, runs off-thread, reports.

The installer takes minutes, so what matters for the GUI is that it does
not live on a thread the dialog owns (closing Settings mid-install must
not destroy a running ``QThread``), that its steps reach the page, and
that a failure leaves the buttons usable instead of a dead spinner.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from bidsmgr import agent_installer  # noqa: E402
from bidsmgr.gui.ai_install_panel import AiInstallPanel  # noqa: E402

pytestmark = pytest.mark.gui


def _panel(qtbot) -> AiInstallPanel:
    panel = AiInstallPanel()
    qtbot.addWidget(panel)
    panel.show()
    return panel


def test_a_missing_module_is_asked_about_with_two_buttons(qtbot):
    panel = _panel(qtbot)

    assert not panel._question.isHidden()
    assert not panel._install_btn.isHidden()
    assert not panel._later_btn.isHidden()
    assert panel._error.isHidden()
    assert not panel.is_running()


def test_not_now_collapses_the_question_but_keeps_install(qtbot):
    panel = _panel(qtbot)

    with qtbot.waitSignal(panel.dismissed, timeout=1000):
        panel._later_btn.click()

    assert panel._question.isHidden()
    assert panel._later_btn.isHidden()
    assert not panel._install_btn.isHidden()


def test_the_install_worker_does_not_belong_to_the_panel(qtbot, monkeypatch):
    """A closed Settings dialog must not delete a QThread that is running."""
    release = threading.Event()

    def slow_install(emit=None):
        if emit is not None:
            emit("Installing torch ...")
        release.wait(10)
        return Path("C:/agent/.venv/python.exe")

    monkeypatch.setattr(agent_installer, "full_install", slow_install)
    panel = _panel(qtbot)

    panel._install_btn.click()
    qtbot.waitUntil(lambda: panel.is_running(), timeout=5000)

    worker = panel._worker
    assert worker is not None
    assert worker.parent() is None                   # outlives the page
    assert not panel._install_btn.isEnabled()

    release.set()
    qtbot.waitUntil(lambda: not panel.is_running(), timeout=5000)


def test_a_finished_install_hands_over_the_interpreter(qtbot, monkeypatch):
    want = Path("C:/agent/.venv/python.exe")
    seen: list[str] = []

    def install(emit=None):
        if emit is not None:
            emit("Creating the environment ...")
            emit("Installing dependencies ...")
        return want

    monkeypatch.setattr(agent_installer, "full_install", install)
    panel = _panel(qtbot)
    panel.installed.connect(seen.append)

    panel._install_btn.click()
    qtbot.waitUntil(lambda: bool(seen), timeout=5000)

    assert seen == [str(want)]
    assert want.name in panel._step.text() or str(want) in panel._step.text()
    assert not panel._spinner.is_busy()
    assert not panel._install_btn.isEnabled()        # the page takes over


def test_a_failed_install_says_why_and_leaves_retry_available(
    qtbot, monkeypatch,
):
    def broken(emit=None):
        raise agent_installer.InstallError(
            "Could not clone the AI Agent repository:\nfatal: no network"
        )

    monkeypatch.setattr(agent_installer, "full_install", broken)
    panel = _panel(qtbot)

    panel._install_btn.click()
    qtbot.waitUntil(lambda: not panel._error.isHidden(), timeout=5000)

    assert "no network" in panel._error.text()
    assert panel._install_btn.isEnabled()
    assert panel._later_btn.isEnabled()
    assert not panel.is_running()
