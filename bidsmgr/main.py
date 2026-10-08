"""GUI entry point for the ``bidsmgr`` console script.

Usage::

    bidsmgr [--theme dark|light] [--project PATH]

* ``--theme``  selects the initial palette (defaults to ``dark``).
* ``--project`` opens (or creates / adopts) a BIDS dataset project at the
  given directory and lands in the Converter bound to it - the same
  project-first flow as the Welcome tab's Open / Create. The output is locked
  to the dataset and the header project switcher appears.

The CLI side of the workflow stays available — ``bidsmgr-scan``,
``-rebuild``, ``-convert``, ``-metadata``, ``-validate`` are unchanged.
The GUI is a convenience layer over the same engine.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import sys
from pathlib import Path
from typing import Optional


def _quietly(fn) -> None:
    """Run ``fn``, logging instead of raising.

    Used for best-effort work that follows a service coming up: failing
    to apply a preference must not take the whole app down with it.
    """
    try:
        fn()
    except Exception:
        logging.getLogger(__name__).debug(
            "background follow-up failed", exc_info=True,
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr",
        description="Schema-driven BIDS converter / curator (GUI).",
    )
    parser.add_argument(
        "--theme", choices=("dark", "light"), default=None,
        help=(
            "Initial color theme. If omitted, the last theme the user "
            "selected in-app is restored (default: dark on first run)."
        ),
    )
    parser.add_argument(
        "--project", type=Path, default=None,
        help=(
            "Open (or create / adopt) a BIDS dataset project at this directory "
            "and land in the Converter bound to it (same as the Welcome tab's "
            "Open / Create). The output is locked to the dataset."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    args = parser.parse_args(argv)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    # Import Qt + GUI lazily so the ``--help`` path doesn't require
    # PyQt to be available. On Linux, make sure libxcb-cursor0 is
    # reachable (Qt 6.5+ refuses to load the xcb plugin without it).
    from .util.qt_platform import prepare as _prepare_qt_platform
    _prepare_qt_platform()

    from PyQt6.QtWidgets import QApplication

    # Register an OpenGL 3.3 core default surface format *before* the
    # QApplication is constructed so the NIfTI viewer's GPU raycaster
    # (bidsmgr.gui.widgets.nifti_gl_view) gets a context its #version 330
    # shaders can compile against — on macOS the compatibility profile is
    # stuck at GL 2.1. Harmless for the rest of the (raster) GUI.
    from PyQt6.QtCore import Qt
    # Share GL contexts across the app's QOpenGLWidgets. Required before the
    # QApplication is built; it lets the raycaster survive the detachable
    # Viewer being re-docked without losing its context.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    from .gui.widgets.nifti_gl_view import request_gl_format
    request_gl_format()

    from .gui.main_window import MainWindow
    from .gui.theme_manager import ThemeManager

    # ``--project`` is a BIDS dataset directory (the project-first model). Open
    # or create/adopt the project bundle nested at <dir>/.bidsmgr/project, then
    # bind it through the same flow the Welcome tab uses (set below, after the
    # window exists).
    project = None
    bids_root = None
    if args.project is not None:
        from .cli.create import open_or_create_workspace
        bids_root = Path(args.project)
        try:
            project = open_or_create_workspace(bids_root)
        except Exception as exc:
            print(f"could not open project {bids_root}: {exc}", file=sys.stderr)
            return 2

    app = QApplication(sys.argv)
    # QSettings keys these to find the right per-user config file on
    # macOS / Linux / Windows. Setting them once here means every
    # ``QSettings()`` constructed in the GUI picks the same INI / plist
    # / registry location.
    app.setOrganizationName("bidsmgr")
    app.setApplicationName("bidsmgr")
    app.setStyle("Fusion")
    # The app font's pixel size is set by ``ThemeManager.apply`` below
    # so it picks up the user's persisted "Font scale" preference.

    # Brand icon for the title bar / taskbar / alt-tab on Linux and
    # Windows. macOS reads its Dock and Spotlight icons from the
    # ``.app`` bundle the installer builds; the call here is still
    # safe (Qt no-ops where a native bundle already supplies an icon).
    from .gui.app_icon import set_app_icon
    set_app_icon(app)

    # Warm the schema on a background thread while the user is still
    # choosing a folder. Answering "which sidecar fields apply here" is a
    # walk of the standard's rule tree, cached for the life of the process,
    # and the first walk was being paid on the GUI thread the moment a scan
    # finished: 393 ms of dead window with the spinner already stopped.
    from PyQt6.QtCore import QThreadPool

    from . import schema as _schema

    QThreadPool.globalInstance().start(_schema.warm_caches)

    # Honor the persisted theme + font-scale if the user didn't pass
    # ``--theme``.
    from .gui.app_settings import AppSettings
    persisted = AppSettings.load()
    initial_theme = args.theme or persisted.theme

    # Which BIDS version this session speaks. Set before any window exists, so
    # the first form built already asks the right questions. Until this, the
    # setting reached the validator alone: a dataset could be checked against
    # one version while being filled in against another.
    from .schema import set_active_version
    set_active_version(persisted.validate_schema_version)

    # Bring the AI agent up in the background while the window is still
    # being built, so by the time anybody can click Ask AI it is usually
    # already serving. It is a separate process with its own interpreter
    # (see bidsmgr.agent_service); nothing here blocks on it, and a
    # machine that cannot run it ends up in a state Settings -> AI Agent
    # can explain rather than failing here.
    from . import agent_service as _agent_service
    agent = _agent_service.service()
    # Both: aboutToQuit covers the ordinary exit, atexit the paths that
    # never reach the event loop. stop() is idempotent and refuses to
    # touch a process we did not start, so the double call costs nothing.
    app.aboutToQuit.connect(agent.stop)
    atexit.register(agent.stop)
    if persisted.ai_enabled:
        from .gui.ai_explainer import push_llm_config

        # Pushed only once the agent is up: our stored LLM preferences
        # are what it should run with, not whatever its config.json says.
        agent.start(
            python=persisted.ai_python,
            on_ready=lambda: _quietly(
                lambda: push_llm_config(persisted)
            ),
        )

    theme = ThemeManager(app, font_scale=persisted.font_scale)
    theme.apply(initial_theme)

    # Round every QComboBox dropdown (frameless + translucent popup window),
    # matching the header project menu. Safe no-op if it ever fails.
    from .gui.combo_popup import install as install_combo_popup_rounder
    install_combo_popup_rounder(app)

    win = MainWindow(theme)
    # Bind the --project dataset through the standard open-project flow so the
    # Converter is set_project'd (output locked), the Editor points at the root,
    # and the header project switcher appears - identical to a Welcome open.
    if project is not None and bids_root is not None:
        win._on_project_opened(project, bids_root)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
