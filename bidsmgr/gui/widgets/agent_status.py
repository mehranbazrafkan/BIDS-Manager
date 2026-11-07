"""The status line under the AI conversation: what is happening, in words.

The footer of the AI window used to say "Working…": true, and useless.
A reader watching a first answer wants to know *what* the machine is
doing — reading a checkpoint from disk, writing tokens, fetching
weights — *why* it takes that long, and *how long* it has been going.
This widget answers all three: an animated glyph, an action line, a
sentence of context under it, and the seconds once generation starts.

The glyph is the same ∞-comet the rest of the app spins while it works
(:class:`~bidsmgr.gui.widgets.spinner.BusySpinner`); one-off states use
a text glyph in the same fixed column, so the two lines of text never
move between states.

Public surface: :meth:`AgentStatus.show_state` for the named phases,
:meth:`set_seconds` while the answer is being written, and
:meth:`clear` when the request is over. Nothing here decides when a
request starts or ends — the dialog does that.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..dialog_chrome import WrapLabel
from .spinner import BusySpinner


#: Width of the glyph column. The comet is 44px wide, the spacing
#: inside the spinner is 8, and the column is fixed so a text glyph and
#: the animation share it: the lines beside it never shift.
GLYPH_COLUMN = 52

#: The states the AI window moves through. ``title`` is the action line,
#: ``detail`` the sentence under it (empty when there is nothing to add),
#: and ``glyph`` is empty for the animated comet and a character for the
#: one-off states.
_STATES: dict[str, tuple[str, str, str]] = {
    "waiting": (
        "Preparing the answer…",
        "The agent runs on this machine. The first answer also loads the "
        "model.",
        "",
    ),
    "loading": (
        "Loading the model…",
        "Reading the checkpoint from disk — once per session, before the "
        "clock starts.",
        "",
    ),
    "generating": (
        "Writing the answer…",
        "The model is generating the text here; nothing leaves this "
        "machine.",
        "",
    ),
    "downloading": (
        "Downloading the model…",
        "Fetching the weights once; later questions use the local copy.",
        "↓",
    ),
    "restored": (
        "Earlier conversation restored.",
        "Nothing was re-generated — these are the messages from before.",
        "↺",
    ),
    "unreachable": (
        "Could not reach the agent.",
        "Start it in Settings → AI Agent, then ask again.",
        "!",
    ),
    "copied": (
        "Conversation copied to the clipboard.",
        "",
        "✓",
    ),
}

#: Colour role per state, matched by the ``tone`` property in the QSS.
_TONES: dict[str, str] = {
    "waiting": "busy",
    "loading": "busy",
    "generating": "busy",
    "downloading": "busy",
    "restored": "info",
    "unreachable": "error",
    "copied": "success",
}


class AgentStatus(QWidget):
    """Two lines of progress and a glyph, in the dialog footer.

    Hidden until something is said — a request in flight, a restored
    conversation, a failure — so an idle window carries no empty strip.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ai-status")
        self._state = ""
        self._base_title = ""
        self._seconds: Optional[int] = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        # One fixed column for both kinds of glyph. The comet and a
        # character cannot be told to the same size otherwise, and the
        # text would jump a few pixels sideways whenever the state does.
        self._glyph_box = QWidget(self)
        self._glyph_box.setObjectName("ai-status-glyph-box")
        self._glyph_box.setFixedWidth(GLYPH_COLUMN)
        glyph_lay = QVBoxLayout(self._glyph_box)
        glyph_lay.setContentsMargins(0, 0, 0, 0)
        glyph_lay.setSpacing(0)
        self._spinner = BusySpinner(self._glyph_box)
        self._icon = QLabel("", self._glyph_box)
        self._icon.setObjectName("ai-status-glyph")
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        glyph_lay.addWidget(
            self._spinner, 0, Qt.AlignmentFlag.AlignVCenter,
        )
        glyph_lay.addWidget(
            self._icon, 0, Qt.AlignmentFlag.AlignVCenter,
        )
        row.addWidget(self._glyph_box, 0)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        self._title = WrapLabel("")
        self._title.setObjectName("ai-status-title")
        self._detail = WrapLabel("")
        self._detail.setObjectName("ai-status-detail")
        self._detail.setVisible(False)
        text.addWidget(self._title)
        text.addWidget(self._detail)
        row.addLayout(text, 1)

        self._icon.setVisible(False)
        self.setVisible(False)

    # -- saying something --------------------------------------------

    def show_state(self, state: str) -> None:
        """Show one of the named states (see ``_STATES``)."""
        if state == self._state and self.isVisible():
            return
        title, detail, glyph = _STATES[state]
        self._state = state
        self._base_title = title
        self._seconds = None
        self._title.setText(title)
        self._detail.setText(detail)
        self._detail.setVisible(bool(detail))
        self._set_tone(_TONES[state])
        if glyph:
            self._spinner.set_busy(False)
            self._icon.setText(glyph)
            self._icon.setVisible(True)
        else:
            self._icon.setVisible(False)
            self._spinner.set_busy(True, message="")
        self.setVisible(True)

    def set_seconds(self, seconds: Optional[int]) -> None:
        """Put the elapsed seconds beside the action line, or drop them.

        Only generation has a clock: loading a checkpoint is not the
        model answering, and counting it as if it were was the point of
        moving the clock here.
        """
        self._seconds = None if seconds is None else max(0, int(seconds))
        self._write_title()

    def clear(self) -> None:
        """Nothing to report: hide the line and stop the glyph."""
        self._state = ""
        self._base_title = ""
        self._seconds = None
        self._title.setText("")
        self._detail.setText("")
        self._icon.setText("")
        self._icon.setVisible(False)
        self._spinner.set_busy(False)
        self.setVisible(False)

    # -- reading -----------------------------------------------------

    def text(self) -> str:
        """The action line, for tests and screen readers."""
        return self._title.text()

    def detail(self) -> str:
        """The context sentence under the action line, if any."""
        return self._detail.text()

    # -- painting ----------------------------------------------------

    def _write_title(self) -> None:
        text = self._base_title
        if self._seconds is not None:
            text = f"{text} ({self._seconds}s)"
        self._title.setText(text)

    def _set_tone(self, tone: str) -> None:
        """Recolour the glyph and the title through the QSS ``tone`` rule."""
        for widget in (self._icon, self._title):
            if widget.property("tone") == tone:
                continue
            widget.setProperty("tone", tone)
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)


__all__ = ["AgentStatus"]
