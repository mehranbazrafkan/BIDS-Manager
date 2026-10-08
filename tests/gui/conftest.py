"""Shared fixtures for the GUI test suite.

Defines :func:`isolated_settings` — sandbox ``QSettings`` per-test so
the GUI's persistence layer doesn't leak the real user's
preferences into tests (or vice versa).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from PyQt6.QtCore import QCoreApplication, QSettings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path) -> Iterator[None]:
    """Redirect ``QSettings`` into a per-test INI file.

    Forces the IniFormat default (macOS otherwise uses native plist
    and ignores ``setPath``) and points it at ``tmp_path``. The
    org/app names also get swapped so a leaked value from outside
    the sandbox cannot poison the test.

    ``autouse`` because opting in is a decision every test author has to
    remember and 17 of 37 files did not, so those wrote to the developer's
    REAL settings store. Two consequences, both seen: a test could pass or fail
    depending on what an earlier run had left behind, which made a genuine
    regression indistinguishable from noise, and running the suite quietly
    rewrote the preferences of whoever ran it.
    """
    orig_org = QCoreApplication.organizationName()
    orig_app = QCoreApplication.applicationName()
    orig_default = QSettings.defaultFormat()

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    QCoreApplication.setOrganizationName("bidsmgr-tests")
    QCoreApplication.setApplicationName("bidsmgr-tests")
    # Ensure an empty starting state.
    QSettings().clear()
    QSettings().sync()
    try:
        yield
    finally:
        QSettings().clear()
        QSettings().sync()
        QCoreApplication.setOrganizationName(orig_org)
        QCoreApplication.setApplicationName(orig_app)
        QSettings.setDefaultFormat(orig_default)


class FakeAgentService:
    """Stand-in for ``bidsmgr.agent_service``'s process-wide singleton.

    The real one starts a *separate Python process* whose first act is
    importing torch — tens of seconds and a few GB, on a machine the
    tests do not get to choose. Since Settings → Save and the Ask AI
    button both reach for it, leaving the real singleton in place would
    make the suite spawn agents in the background. So the singleton is
    swapped out for every GUI test, and this object is what a test that
    cares about the wiring can inspect.

    It has the same surface as :class:`bidsmgr.agent_service.AgentService`
    minus the plumbing: state, detail, url, owned, start, stop,
    is_running and wait_running.
    """

    def __init__(self) -> None:
        self.state = "stopped"
        self.detail = "Not running."
        self.url = ""
        self.owned = False
        self.started_with: list[str] = []
        self.stop_calls = 0

    # -- the surface AgentService has --------------------------------
    def start(self, *, python: str = "", on_ready=None) -> None:
        self.started_with.append(python)
        self.state = "starting"
        self.detail = "Starting the AI agent..."
        self.url = ""
        self.owned = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.state = "stopped"
        self.detail = "Not running."
        self.url = ""
        self.owned = False

    def is_running(self) -> bool:
        return self.state == "running"

    def wait_running(self, timeout: float = 60.0) -> bool:
        return self.state == "running"

    # -- what a test needs to say happened --------------------------
    def pretend_running(self, url: str = "", owned: bool = True) -> None:
        self.state = "running"
        self.detail = "Running (started by BIDS-Manager)."
        self.url = url
        self.owned = owned

    def pretend_failed(self, detail: str = "No Python found.") -> None:
        self.state = "failed"
        self.detail = detail
        self.url = ""
        self.owned = False


@pytest.fixture(autouse=True)
def fake_agent_service(monkeypatch) -> Iterator[FakeAgentService]:
    """Never let a test start the real agent."""
    from bidsmgr import agent_service as module

    fake = FakeAgentService()
    monkeypatch.setattr(module, "_SERVICE", fake)
    yield fake


def open_every_folder(tree) -> None:
    """Draw every row in a lazy tree, then put the folds back as they were.

    Both file trees draw a folder's contents when the folder is OPENED, so a
    test that wants to see a deep row has to open its way down, as a user
    does. ``QTreeWidget.expandAll`` is not enough on its own: it expands the
    rows that exist when it is called, and the rows it creates by doing so
    are left folded.

    The fold state is restored because several tests assert that what the
    user had open survived a refresh, and a helper that left the tree
    expanded would be answering its own question.
    """
    was_open: list = []

    def visit(item) -> None:
        if item.childCount():
            if not item.isExpanded():
                was_open.append(item)
            item.setExpanded(True)
        for i in range(item.childCount()):
            visit(item.child(i))

    for i in range(tree.topLevelItemCount()):
        visit(tree.topLevelItem(i))
    for item in was_open:
        item.setExpanded(False)
