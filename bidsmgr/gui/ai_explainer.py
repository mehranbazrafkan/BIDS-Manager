"""Ask the bundled AI agent to explain a finding, in plain language.

BIDS Manager ships an agent next to the app in ``bidsmgr/ai_agent/``.
It is a **service**, not a library, and this module never imports it:

* its modules import each other by bare name (``from agent import
  Agent``), so they cannot live inside the ``bidsmgr`` package;
* it loads a local LLM, which a desktop app must not pull in on
  startup — the model is downloaded and resident on *its* schedule, not
  ours.

So everything crosses HTTP, against the contract documented in
``bidsmgr/ai_agent/README.md``::

    POST /predict   {"user_input": str, "context": {...}}
     -> {"response": str, "intent": "explain"|"fix", "tool_used": str?}
    GET  /models    cheapest endpoint there is; never loads the model
    GET  /config    current knobs — which model is running
    PUT  /config    apply this app's stored LLM preferences

The agent process itself belongs to :mod:`bidsmgr.agent_service`, which
starts it at launch and decides where it listens; this module only ever
talks to it.

The whole GUI side needs one call — :func:`ask` — plus a finding as a
plain dict, which :func:`build_payload` makes from the fields every
``ValMessage`` already holds. Nothing here blocks: :class:`_PredictWorker`
owns the request so the first answer (which loads the model) cannot
freeze the window.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ..agent_service import service
from .app_settings import AppSettings, KEYS
from .dialog_chrome import (
    WrapLabel,
    build_footer_with,
    build_header,
    card,
    scrollable_body,
)
from .widgets.spinner import BusySpinner

log = logging.getLogger(__name__)

# The agent's own ``python main.py`` default (ai_agent/main.py line 245).
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# Generous: the first /predict loads a local LLM onto the device.
DEFAULT_TIMEOUT = 120
# Sent when the user has not typed anything. Deliberately the word
# "Explain" — ``Planner._decided_intent`` routes on the prompt, and the
# button is there to explain, not to reach for a fix tool.
DEFAULT_PROMPT = "Explain this BIDS validation finding to me."

# Workers the user closed the dialog on. Held HERE, never as a child of
# the dialog: a QThread destroyed while ``run()`` is in flight takes the
# process with it, and "close while the model is still answering" is the
# first thing anybody does. See :func:`_retire`.
_ORPHANS: set[QThread] = set()


# =====================================================================
# Settings
# =====================================================================

def _qsettings():
    return AppSettings._settings()


def _read(key: str, default: Any) -> Any:
    """Read one key straight from ``QSettings``.

    Deliberately *not* :meth:`AppSettings.load`: that walks every setting
    in the app (and pulls in the defacing engines) and this runs on a
    button click. The keys live in one namespace either way —
    ``app_settings.KEYS`` is the schema, this only fetches from it, and
    it goes through ``AppSettings._settings`` so the test suite's
    sandboxed store is honoured like everywhere else.
    """
    return _qsettings().value(KEYS[key], default)


def ai_enabled() -> bool:
    """Whether the "Ask AI" button is drawn at all."""
    raw = _read("ai_enabled", "1")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes")


def agent_base_url() -> str:
    """Where the agent is.

    The service knows first: if 8000 was already taken, BIDS-Manager
    moved the agent and that address lives only in
    :mod:`bidsmgr.agent_service`. The stored ``ai_base_url`` is the
    fallback for an agent somebody started by hand.
    """
    running = service().url
    if running:
        return running.rstrip("/")
    raw = _read("ai_base_url", DEFAULT_BASE_URL)
    url = str(raw or "").strip()
    if not url:
        return DEFAULT_BASE_URL
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def agent_timeout() -> float:
    raw = _read("ai_timeout", DEFAULT_TIMEOUT)
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_TIMEOUT)
    return max(5.0, seconds)


# =====================================================================
# Payload
# =====================================================================

_TAG_RE = re.compile(r"<[^>]+>")
_BREAK_RE = re.compile(r"(?i)<br\s*/?>|</p\s*>|</div\s*>")


def plain_text(rich: Any) -> str:
    """Reduce validator rich text to what an LLM should read.

    Finding messages carry ``<code>...</code>`` around literals — the
    GUI renders them, but sent raw they become angle-bracket noise in a
    prompt, and a message that happens to contain ``RepetitionTime<3``
    would otherwise be half-eaten as a tag.
    """
    text = str(rich or "")
    if "<" not in text and "&" not in text:
        return text.strip()
    text = _BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    return html.unescape(text).strip()


def build_payload(
    *,
    severity: str,
    message: str,
    rule_id: str = "",
    field: Optional[str] = None,
    line: Optional[int] = None,
    lines: Optional[list] = None,
    fix_label: Optional[str] = None,
    fix_action: Optional[str] = None,
    mirrored: bool = False,
    extra: Optional[dict] = None,
) -> dict:
    """Build the ``context`` object ``POST /predict`` expects.

    The documented shape is ``editor.types.Issue.model_dump()`` — the
    agent's README example is that object field for field. Empty fields
    are omitted rather than sent as ``null``: ``utils.compact_json``
    renders Python's ``None`` as the literal word "None", which is a
    worse thing to hand a language model than an absent key.

    ``extra`` is where the caller adds what the agent's retriever keys
    on but the core Issue shape has no room for — ``path``, ``datatype``
    and ``suffix`` are read by ``retriever._issue_structured`` and they
    are what makes retrieval land on the right BIDS rule.
    """
    ctx: dict[str, Any] = {
        "severity": str(severity or ""),
        "rule_id": str(rule_id or ""),
        "message": plain_text(message),
        "mirrored": bool(mirrored),
    }
    if field:
        ctx["field"] = str(field)
    if line:
        try:
            ctx["line"] = int(line)
        except (TypeError, ValueError):
            pass
    if lines:
        ctx["lines"] = [int(x) for x in lines if str(x).isdigit()]
    if fix_label:
        ctx["fix_label"] = str(fix_label)
    if fix_action:
        ctx["fix_action"] = str(fix_action)
    for key, value in (extra or {}).items():
        if value in (None, "", [], {}):
            continue
        # ``setdefault``: the extras carry what the Issue shape has no
        # room for — path, datatype, suffix — and must never shadow the
        # finding's own fields if a caller passes both.
        ctx.setdefault(str(key), value)
    return ctx


# =====================================================================
# Transport
# =====================================================================

class AgentError(RuntimeError):
    """Transport or protocol failure talking to the agent service.

    Carries a message written for a *user*, not a log: every one of
    these surfaces in the dialog, and the common case (the service is
    not running) has to say how to start it.
    """


class AgentClient:
    """Blocking client for the agent's read endpoints and ``/predict``.

    Blocking on purpose — every caller wraps it in
    :class:`_PredictWorker`. ``base_url`` / ``timeout`` default to the
    stored settings so a test can pin both.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or agent_base_url()).rstrip("/")
        self.timeout = float(timeout or agent_timeout())

    # -- transport ---------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        *,
        timeout: Optional[float] = None,
    ) -> dict:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            url, data=data, headers=headers, method=method,
        )
        try:
            with urllib.request.urlopen(
                req, timeout=timeout or self.timeout,
            ) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise AgentError(_http_error(url, exc)) from exc
        except urllib.error.URLError as exc:
            raise AgentError(_unreachable(url, getattr(exc, "reason", exc))) from exc
        except (TimeoutError, OSError) as exc:
            # ``socket.timeout`` is an OSError on every Python we support,
            # and urlopen raises it bare rather than wrapped in URLError
            # when the read itself times out.
            raise AgentError(_timed_out(url, exc)) from exc

        if not raw.strip():
            return {}
        try:
            out = json.loads(raw)
        except ValueError as exc:
            raise AgentError(
                f"The AI agent at {url} answered with something that was "
                f"not JSON:\n\n{raw[:400]}"
            ) from exc
        if not isinstance(out, dict):
            raise AgentError(
                f"The AI agent at {url} answered with a JSON "
                f"{type(out).__name__}, expected an object."
            )
        return out

    # -- endpoints ---------------------------------------------------

    def ping(self) -> dict:
        """``GET /models`` — a liveness probe that loads no model.

        Only ``/predict`` constructs the agent singleton, so this stays
        fast even while the LLM is still cold.
        """
        return self._request("GET", "/models", timeout=8.0)

    def config(self) -> dict:
        """``GET /config`` — what the agent is running (model, retrieval).

        Shown by Settings so "Connected" also says *what* it connected
        to, and read to preselect the model in the dropdown.
        """
        return self._request("GET", "/config")

    def set_config(self, **fields: Any) -> dict:
        """``PUT /config`` — apply this app's stored LLM preferences.

        The only endpoint here that changes state, called twice: once
        when the agent has come up (see ``bidsmgr.main``) and again
        whenever Settings is saved. The caller treats every failure as
        best-effort — a preference that could not be pushed is not worth
        interrupting anyone for, and the next save tries again.
        """
        return self._request("PUT", "/config", dict(fields))

    def predict(self, user_input: str, context: dict) -> dict:
        """Run the agent. Returns ``{response, intent, tool_used}``."""
        out = self._request(
            "POST",
            "/predict",
            {"user_input": str(user_input), "context": dict(context)},
        )
        out.setdefault("response", "")
        out.setdefault("intent", "")
        if not str(out["response"]).strip():
            raise AgentError(
                "The AI agent answered, but its response was empty."
            )
        return out


def llm_config_payload(settings: Any) -> dict:
    """The ``PUT /config`` body built from stored LLM preferences.

    An empty ``ai_model`` means "leave the agent's own config.json choice
    alone", so ``model_name`` is omitted rather than sent blank — a blank
    would travel all the way to ``from_pretrained``.
    """
    payload: dict[str, Any] = {
        "max_new_tokens": int(settings.ai_max_new_tokens),
        "temperature": float(settings.ai_temperature),
        "enable_thinking": bool(settings.ai_enable_thinking),
        "quantization": str(settings.ai_quantization),
        "device_map": str(settings.ai_device_map),
    }
    model = str(getattr(settings, "ai_model", "") or "").strip()
    if model:
        payload["model_name"] = model
    return payload


def push_llm_config(settings: Any) -> Optional[str]:
    """Send those preferences to the agent. Returns an error, or ``None``.

    Deliberately best-effort and quiet: the agent may be down because the
    user stopped it, and that is not something to interrupt anybody for.
    The next save, or the next launch, tries again.
    """
    try:
        AgentClient(timeout=15.0).set_config(**llm_config_payload(settings))
    except AgentError as exc:
        log.info("could not apply AI agent settings: %s", exc)
        return str(exc)
    return None


def _unreachable(url: str, reason: Any) -> str:
    return (
        f"Could not reach the BIDS AI agent at {url}.\n\n"
        "BIDS-Manager starts it for you, so it is either still starting "
        "up or it could not start at all. Open Settings -> AI Agent and "
        f"press Start to see why.\n\n({reason})"
    )


def _timed_out(url: str, exc: Any) -> str:
    return (
        f"The BIDS AI agent at {url} did not answer in time.\n\n"
        "The first request loads the local model, which can take a while "
        "on CPU. Raise the timeout in Settings -> AI Agent, then click "
        f"\"Ask again\".\n\n({exc})"
    )


def _http_error(url: str, exc: urllib.error.HTTPError) -> str:
    body = ""
    try:
        body = (exc.read() or b"").decode("utf-8", "replace").strip()
    except Exception:                                    # noqa: BLE001
        pass

    # The agent reports its own failures as {"error": "..."} — that is the
    # useful half of a non-2xx answer and is worth leading with. The HTML
    # Flask produces when an error escapes every handler is not: its text
    # ("The server encountered an internal error...") tells a reader
    # nothing they can act on, and the desktop app used to show it whole.
    reason = ""
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("error"):
        reason = str(parsed["error"]).strip()
    # Starts with "<" rather than "contains <html>": Flask's own page
    # opens with "<!doctype html>", which has no "<html" in it at all.
    markup = body.lstrip().lower().startswith("<")
    if reason:
        detail = f"\n\n{reason}"
    elif body and not markup:
        detail = f"\n\n{body[:400]}"
    else:
        detail = ""

    if exc.code == 404:
        return (
            f"{url} answered 404. The service at that address is probably "
            f"not the BIDS AI agent.\n\nOpen Settings -> AI Agent and "
            f"press Start so BIDS-Manager brings up its own.{detail}"
        )
    if exc.code == 400:
        return (
            "The BIDS AI agent rejected the request as malformed. "
            f"Check that it is running the version shipped in "
            f"bidsmgr/ai_agent.{detail}"
        )
    if reason:
        return f"The BIDS AI agent could not answer.\n\n{reason}"
    if markup:
        return (
            f"The BIDS AI agent answered HTTP {exc.code} with an error page "
            "instead of an explanation.\n\n"
            "It raised while handling the request. Restart it from "
            "Settings -> AI Agent — its log holds the traceback — and "
            "choose a different model there if it keeps happening."
        )
    return f"The BIDS AI agent answered HTTP {exc.code}.{detail}"


# =====================================================================
# Worker
# =====================================================================

class _PredictWorker(QThread):
    """One ``/predict`` round-trip off the GUI thread.

    Signals rather than a return value, because the whole point is that
    ``run()`` outlives the click. ``done`` / ``failed`` land on the
    dialog via a queued connection (it lives in the main thread) and
    simply vanish if the dialog has been closed — which is why the
    worker is never its child: see :func:`_retire`.
    """

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self, user_input: str, context: dict, parent=None,
    ) -> None:
        super().__init__(parent)
        self._user_input = user_input
        self._context = dict(context)

    def run(self) -> None:                               # noqa: D102
        try:
            # The first click of a session can land while torch is still
            # importing on the agent's side. Without this, a refused
            # connection during that window would be reported to the user
            # as "the agent is not running" when it is very much on its way.
            service().wait_running(timeout=60.0)
            result = AgentClient().predict(self._user_input, self._context)
        except AgentError as exc:
            log.info("AI agent request failed: %s", exc)
            self.failed.emit(str(exc))
        except Exception:                                # noqa: BLE001
            log.exception("AI agent request crashed")
            self.failed.emit(traceback.format_exc())
        else:
            self.done.emit(result)


def _retire(worker: QThread) -> None:
    """Drop the hold on *worker* once its thread has finished.

    Connected to ``QThread.finished``, which Qt emits after ``run()``
    returns — so ``deleteLater`` here can never race the thread it is
    deleting. The set lives at module scope so a dialog the user closed
    mid-answer cannot take the thread down with it.
    """
    _ORPHANS.discard(worker)
    worker.deleteLater()


def _spawn(
    user_input: str,
    context: dict,
    *,
    on_done,
    on_failed,
) -> _PredictWorker:
    """Start a request, hold the thread until it is genuinely done.

    Both result slots are connected *before* ``start()``. Getting that
    backwards is not a theoretical race: a refused connection fails
    inside a few milliseconds, and a slot connected a moment too late
    never fires — which leaves the dialog stuck on "Working…" forever.
    """
    worker = _PredictWorker(user_input, context)
    worker.done.connect(on_done)
    worker.failed.connect(on_failed)
    # Parentless by construction (``parent=None`` above): never a child
    # of the dialog, whose lifetime the user controls.
    _ORPHANS.add(worker)
    worker.finished.connect(lambda w=worker: _retire(w))
    worker.start()
    return worker


# =====================================================================
# Dialog
# =====================================================================

class AiExplainDialog(QDialog):
    """One finding, and what the agent makes of it.

    Modeless — findings come in groups and closing the window to read
    the next one would be a chore. The reported side never changes; the
    answer area refills on every ask, so a follow-up question replaces
    the previous answer while keeping the finding on screen.
    """

    def __init__(self, payload: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS-Manager — AI explanation")
        self.resize(660, 640)
        self.setMinimumWidth(440)

        self._payload: dict[str, Any] = dict(payload)
        # A flag, not the thread. Holding the QThread here is what makes
        # "user closes the dialog while the model is answering" unsafe.
        self._busy = False
        self._last_prompt = DEFAULT_PROMPT

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(build_header("AI explanation", self._subtitle()))

        scroll, body = scrollable_body()
        body.setSpacing(12)
        self._build_reported(body)
        self._build_answer(body)
        self._build_followup(body)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        self._retry_btn = buttons.addButton(
            "Ask again", QDialogButtonBox.ButtonRole.ActionRole,
        )
        self._copy_btn = buttons.addButton(
            "Copy answer", QDialogButtonBox.ButtonRole.ActionRole,
        )
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        outer.addWidget(build_footer_with(self._status, buttons))

        buttons.rejected.connect(self.reject)
        self._retry_btn.clicked.connect(lambda: self.start(self._last_prompt))
        self._copy_btn.clicked.connect(self._copy)

        self._set_busy(False)

    # -- construction ------------------------------------------------

    def _subtitle(self) -> str:
        rule = str(self._payload.get("rule_id") or "").strip()
        sev = str(self._payload.get("severity") or "").strip()
        bits = [b for b in (rule, sev) if b]
        return "  ·  ".join(bits) or "A finding from BIDS Manager"

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("val-caption")
        return label

    @staticmethod
    def _fact(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("ai-fact")
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        label.setWordWrap(True)
        return label

    def _build_reported(self, body: QVBoxLayout) -> None:
        frame, lay = card("What BIDS Manager reported")

        facts = self._facts()
        if facts:
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(4)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            for name, value in facts:
                form.addRow(self._caption(name), self._fact(value))
            lay.addLayout(form)

        message = str(self._payload.get("message") or "").strip()
        if message:
            body_l = WrapLabel(message)
            body_l.setObjectName("val-body")
            body_l.setTextFormat(Qt.TextFormat.RichText)
            lay.addWidget(body_l)

        if not facts and not message:
            lay.addWidget(WrapLabel("(no detail was recorded for this finding)"))
        body.addWidget(frame)

    def _facts(self) -> list[tuple[str, str]]:
        """Label/value pairs for the reported side, empty ones dropped."""
        p = self._payload
        out: list[tuple[str, str]] = []

        def add(name: str, value: Any) -> None:
            text = str(value or "").strip()
            if text and text.lower() != "none":
                out.append((name, text))

        add("Rule", p.get("rule_id"))
        add("Field", p.get("field"))
        add("File", p.get("path"))
        datatype, suffix = p.get("datatype"), p.get("suffix")
        if datatype or suffix:
            add("Type", "/".join(x for x in (datatype, suffix) if x))
        add("Severity", p.get("severity"))
        add("Row", p.get("row"))
        return out

    def _build_answer(self, body: QVBoxLayout) -> None:
        frame, lay = card("What the AI agent says")

        self._spinner = BusySpinner()
        lay.addWidget(self._spinner)

        self._error = WrapLabel("")
        self._error.setObjectName("ai-error")
        self._error.setTextFormat(Qt.TextFormat.PlainText)
        self._error.setVisible(False)
        lay.addWidget(self._error)

        self._answer = QTextEdit()
        self._answer.setObjectName("ai-answer")
        self._answer.setReadOnly(True)
        self._answer.setAcceptRichText(False)
        self._answer.setMinimumHeight(150)
        self._answer.setPlaceholderText(
            "The agent's explanation will appear here."
        )
        lay.addWidget(self._answer, 1)

        self._meta = QLabel("")
        self._meta.setObjectName("ai-meta")
        self._meta.setWordWrap(True)
        lay.addWidget(self._meta)
        body.addWidget(frame)

    def _build_followup(self, body: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        self._followup = QLineEdit()
        self._followup.setPlaceholderText(
            "Ask a follow-up about this finding  (Enter to send)"
        )
        self._followup.setClearButtonEnabled(True)
        self._followup.returnPressed.connect(self._send_followup)
        send = QPushButton("Ask")
        send.setObjectName("val-ai")
        send.setToolTip("Send this question to the BIDS AI agent.")
        send.clicked.connect(self._send_followup)
        row.addWidget(self._followup, 1)
        row.addWidget(send, 0)
        body.addLayout(row)

    # -- behaviour ---------------------------------------------------

    def is_busy(self) -> bool:
        """True while a request is in flight.

        Public because "did it answer yet?" is what every caller wants
        to know — including tests, which otherwise reach for the
        private flag and pin this dialog to its own internals.
        """
        return self._busy

    def start(self, prompt: str = DEFAULT_PROMPT) -> None:
        """Ask *prompt* about the finding shown in this dialog."""
        if self._busy:
            return
        prompt = str(prompt or "").strip() or DEFAULT_PROMPT
        self._last_prompt = prompt
        self._set_busy(True)
        self._started = time.time()
        # The thread is owned by ``_ORPHANS``, not by this dialog; only
        # its signals are wired here, so closing the window mid-answer
        # just leaves the request to finish into the void.
        _spawn(
            prompt,
            self._payload,
            on_done=self._on_done,
            on_failed=self._on_failed,
        )

    def _send_followup(self) -> None:
        text = self._followup.text().strip()
        if not text or self._busy:
            return
        self._followup.clear()
        self.start(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._spinner.set_busy(
            busy,
            message=(
                "Asking the agent… first answer loads the local model "
                "and can take a minute."
            ),
        )
        self._retry_btn.setEnabled(not busy)
        self._followup.setEnabled(not busy)
        if busy:
            self._error.setVisible(False)
            self._status.setText("Working…")
        else:
            self._status.setText(
                f"Agent at {agent_base_url()}"
            )

    def _on_done(self, result: dict) -> None:
        elapsed = time.time() - getattr(self, "_started", time.time())
        self._set_busy(False)
        self._error.setVisible(False)
        self._answer.setPlainText(str(result.get("response") or ""))
        intent = str(result.get("intent") or "").strip() or "—"
        tool = str(result.get("tool_used") or "").strip()
        bits = [f"intent: {intent}"]
        if tool:
            bits.append(f"tool: {tool}")
        bits.append(f"{elapsed:.1f}s")
        self._meta.setText("  ·  ".join(bits))
        self._followup.setFocus()

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        # A WrapLabel, not a QTextEdit: the answer area is a text edit
        # for copyability, but this is a short paragraph of prose and it
        # has to wrap to the card's width.
        self._error.setText(message)
        self._error.setVisible(True)
        self._meta.setText("")
        self._status.setText("Could not reach the agent.")

    def _copy(self) -> None:
        text = self._answer.toPlainText()
        if not text:
            return
        QApplication.clipboard().setText(text)
        self._status.setText("Answer copied to the clipboard.")


# =====================================================================
# Entry point
# =====================================================================

def ask(payload: dict, parent=None) -> Optional[AiExplainDialog]:
    """Open an explanation window for *payload* and start the request.

    Returns the dialog (so a caller could, say, test it) or ``None``
    when the feature is switched off in Settings. Modeless and
    ``WA_DeleteOnClose``: findings are clicked one after another and a
    window that outlives its purpose is a leak.
    """
    if not ai_enabled():
        return None
    dlg = AiExplainDialog(payload, parent=parent)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.start()
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


__all__ = [
    "AgentClient",
    "AgentError",
    "AiExplainDialog",
    "DEFAULT_BASE_URL",
    "DEFAULT_PROMPT",
    "ai_enabled",
    "agent_base_url",
    "agent_timeout",
    "ask",
    "build_payload",
    "plain_text",
]
