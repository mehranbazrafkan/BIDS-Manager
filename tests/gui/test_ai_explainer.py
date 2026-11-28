"""The "Ask AI" button and the explanation window it opens.

Covers the two halves of the feature that a user will judge it on: the
button is where a finding is, and it says something useful — including
the very first thing that will happen to most people, which is that the
agent service is not running yet.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator, NamedTuple

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton

pytest.importorskip("PyQt6")

from bidsmgr.gui import ai_explainer                     # noqa: E402
from bidsmgr.gui.ai_explainer import (                   # noqa: E402
    AiExplainDialog,
    ask,
    build_payload,
)
from bidsmgr.gui.app_settings import KEYS, AppSettings    # noqa: E402
from bidsmgr.gui.widgets.val_message import ValMessage   # noqa: E402

pytestmark = pytest.mark.gui


# =====================================================================
# Helpers
# =====================================================================


def _ai_buttons(msg: ValMessage) -> list[QPushButton]:
    return [
        b for b in msg.findChildren(QPushButton)
        if b.objectName() == "val-ai"
    ]


def _set(**changes) -> None:
    """Persist settings through the same object the app uses."""
    s = AppSettings.load()
    for key, value in changes.items():
        setattr(s, key, value)
    s.save()


def _message(qtbot, severity: str, **kwargs) -> ValMessage:
    msg = ValMessage(
        severity,
        kwargs.pop("rule", "entity.required"),
        kwargs.pop("message", "<code>RepetitionTime</code> is required."),
        **kwargs,
    )
    qtbot.addWidget(msg)
    msg.show()
    return msg


def _dataset(root):
    """A tiny dataset: root metadata, a table, one file with a finding."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        '{"Name": "Probe", "BIDSVersion": "1.9.0"}', encoding="utf-8")
    (root / "participants.tsv").write_text(
        "participant_id\tage\nsub-01\t24\n", encoding="utf-8")
    func = root / "sub-01/func"
    func.mkdir(parents=True, exist_ok=True)
    (func / "sub-01_task-rest_bold.json").write_text(
        '{"RepetitionTime": 2.0, "TaskName": "rest"}', encoding="utf-8")
    return root


class _StubAgent(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:               # noqa: D102
        pass

    def do_GET(self) -> None:                           # noqa: N802
        if self.path == "/models":
            payload = getattr(self.server, "models_payload", None)
            if payload is None:
                payload = {"models": [
                    {
                        "name": "stub-7b-instruct",
                        "description": "A stub for the wire",
                        "size": "7B",
                        "vram_gb": "~14 GB (fp16)",
                    },
                    {
                        "name": "other-model",
                        "description": "Also a stub",
                        "size": "1B",
                        "vram_gb": "~2 GB (fp16)",
                    },
                ]}
            body = json.dumps(payload).encode()
        elif self.path == "/config":
            body = json.dumps({
                "llm": {
                    "model_name": "stub-7b-instruct",
                    "max_new_tokens": 123,
                    "temperature": 0.5,
                    "device_map": "cpu",
                    "quantization": "4bit",
                    "enable_thinking": True,
                },
                "hardware": {
                    "backend": "cuda",
                    "device_name": "Stub GPU",
                    "cuda_version": "12.4",
                    "memory_gb": 8.0,
                    "device_count": 1,
                    "torch": "2.5.1",
                    "python": "3.11.9",
                    "cpu_cores": 12,
                },
            }).encode()
        elif self.path == "/models/download/status":
            body = json.dumps(
                self.server.download_status            # type: ignore[attr-defined]
            ).encode()
        else:
            body = b"{}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self) -> None:                           # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        self.server.config = json.loads(raw or b"{}")   # type: ignore[attr-defined]
        body = json.dumps({}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:                          # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        received = json.loads(raw or b"{}")
        if self.path == "/models/download":
            self.server.download_started = received  # type: ignore[attr-defined]
            body = json.dumps({
                "name": received.get("name"), "status": "downloading",
                "downloaded_bytes": 0, "total_bytes": None, "error": "",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.server.received = received                 # type: ignore[attr-defined]
        self.server.requests += 1                       # type: ignore[attr-defined]
        body = json.dumps({
            "response": "RepetitionTime is the TR of your scan.",
            "intent": "explain",
            "tool_used": None,
            "model": "stub/stub-7b-instruct",
            "prompt_tokens": 214,
            "generated_tokens": 42,
            "generation_seconds": 1.25,
            "model_load_seconds": 4.5,
            "log": {
                "system_prompt": "SYSTEM PROMPT",
                "user_prompt": "What is TR?",
                "output": "RepetitionTime is the TR of your scan.",
                "model": "stub/stub-7b-instruct",
                "prompt_tokens": 214,
                "generated_tokens": 42,
                "seconds": 1.25,
                "model_load_seconds": 4.5,
            },
        }).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Stub(NamedTuple):
    """The running stub plus the URL to point the dialog at."""

    url: str
    server: HTTPServer

    @property
    def received(self) -> dict:
        """The body of the last ``/predict``, as crossed the wire."""
        return self.server.received                   # type: ignore[attr-defined]

    @property
    def requests(self) -> int:
        """How many ``/predict`` calls have arrived — the difference
        between "it showed me the conversation" and "it generated it
        again", which is otherwise invisible from outside."""
        return self.server.requests                   # type: ignore[attr-defined]

    @property
    def config(self) -> dict:
        """The body of the last ``PUT /config``, as crossed the wire."""
        return self.server.config                     # type: ignore[attr-defined]


@pytest.fixture
def stub_agent() -> Iterator[_Stub]:
    server = HTTPServer(("127.0.0.1", 0), _StubAgent)
    server.received = {}                                # type: ignore[attr-defined]
    server.requests = 0                                 # type: ignore[attr-defined]
    server.config = {}                                  # type: ignore[attr-defined]
    server.download_started = {}                        # type: ignore[attr-defined]
    server.download_status = {                          # type: ignore[attr-defined]
        "name": "", "status": "idle", "downloaded_bytes": 0,
        "total_bytes": None, "error": "",
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield _Stub(f"http://{host}:{port}", server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def dead_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def agent_enabled():
    """Every test here starts from the shipped default: the button is on."""
    _set(
        ai_enabled=True,
        ai_base_url="http://127.0.0.1:8000",
        ai_timeout=30,
        ai_include_dataset_structure=True,
        ai_include_file_content=True,
    )
    yield
    _set(
        ai_enabled=True,
        ai_base_url="http://127.0.0.1:8000",
        ai_timeout=120,
        ai_include_dataset_structure=True,
        ai_include_file_content=True,
    )


# =====================================================================
# The button
# =====================================================================


class TestTheButton:
    def test_a_warning_offers_it(self, qtbot):
        assert len(_ai_buttons(_message(qtbot, "warn"))) == 1

    def test_an_error_offers_it(self, qtbot):
        assert len(_ai_buttons(_message(qtbot, "err"))) == 1

    def test_a_skipped_row_offers_it(self, qtbot):
        """"We did not read this file" is exactly what people ask about."""
        assert len(_ai_buttons(_message(qtbot, "skip"))) == 1

    def test_a_clean_file_does_not(self, qtbot):
        """A button that explains why nothing is wrong is noise, and it
        makes the row read as if it had a problem."""
        msg = _message(
            qtbot, "ok", rule="SCHEMA · func/bold", message="Entity set is valid.",
        )
        assert _ai_buttons(msg) == []

    def test_it_disappears_when_the_feature_is_off(self, qtbot):
        _set(ai_enabled=False)
        assert _ai_buttons(_message(qtbot, "err")) == []

    def test_the_tooltip_says_the_app_starts_it_for_you(self, qtbot):
        # Hold the row: it is a top-level (no parent), so nothing but
        # this reference keeps it alive once _message returns.
        msg = _message(qtbot, "err")
        btn = _ai_buttons(msg)[0]
        assert "bidsmgr/ai_agent" in btn.toolTip()
        assert "python main.py" not in btn.toolTip()
        assert "starts that agent for you" in btn.toolTip()

    def test_it_sits_beside_the_fix_button_without_replacing_it(
        self, qtbot,
    ):
        msg = _message(qtbot, "err", fix_label="Add the missing key")
        names = {b.objectName() for b in msg.findChildren(QPushButton)}
        assert {"val-fix", "val-ai"} <= names


# =====================================================================
# What gets sent
# =====================================================================


class TestWhatGetsSent:
    def test_clicking_sends_the_finding_and_the_file(self, qtbot, monkeypatch):
        seen = {}

        def fake_ask(payload, parent=None, root=None):
            seen["payload"] = payload
            seen["parent"] = parent
            seen["root"] = root

        monkeypatch.setattr(ai_explainer, "ask", fake_ask)

        msg = _message(
            qtbot,
            "err",
            rule="entity.required",
            field="RepetitionTime",
            context={"path": "sub-01/func", "datatype": "func"},
        )
        btn = _ai_buttons(msg)[0]
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)

        payload = seen["payload"]
        assert payload["rule_id"] == "entity.required"
        assert payload["field"] == "RepetitionTime"
        assert payload["severity"] == "err"
        # The <code> markup is the GUI's business, not the model's.
        assert payload["message"] == "RepetitionTime is required."
        assert payload["path"] == "sub-01/func"
        assert payload["datatype"] == "func"
        assert seen["parent"] is not None
        # No dataset was handed to this message, so none is claimed.
        assert seen["root"] is None

    def test_the_dataset_root_travels_to_the_window(
        self, qtbot, monkeypatch, tmp_path,
    ):
        """The window walks the dataset for its shape and reads the
        file — and it cannot find that dataset itself, because
        ``context["path"]`` is deliberately relative to the root."""
        seen = {}

        def fake_ask(payload, parent=None, root=None):
            seen["root"] = root

        monkeypatch.setattr(ai_explainer, "ask", fake_ask)

        msg = _message(
            qtbot, "err", context={"path": "sub-01/func"}, root=tmp_path,
        )
        qtbot.mouseClick(_ai_buttons(msg)[0], Qt.MouseButton.LeftButton)

        assert seen["root"] == tmp_path

    def test_the_window_sends_the_tree_and_the_file(
        self, qtbot, stub_agent, tmp_path,
    ):
        """The two evidence fields, as they leave the dialog.

        Both are what the agent cannot work out for itself: it never
        sees the disk. The tree describes the dataset the finding sits
        in and the digest says what is in the file — and neither
        carries the folder the dataset was built from, which is the
        reason ``context["path"]`` is relative in the first place.
        """
        _set(ai_base_url=stub_agent.url)
        root = _dataset(tmp_path / "my_mri_dataset")
        payload = build_payload(
            severity="err",
            message="RepetitionTime is required.",
            rule_id="evidence.test.tree",
            extra={"path": "sub-01/func/sub-01_task-rest_bold.json"},
        )

        dialog = AiExplainDialog(payload, root=root)
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        sent = stub_agent.received                       # type: ignore[attr-defined]
        assert sent["dataset_structure"].startswith("my_mri_dataset/")
        assert "participants.tsv" in sent["dataset_structure"]
        assert "func/" in sent["dataset_structure"]
        assert "RepetitionTime: 2.0" in sent["file_content"]
        assert str(tmp_path) not in sent["dataset_structure"]
        assert str(tmp_path) not in sent["file_content"]

    def test_the_dataset_tree_can_be_turned_off(
        self, qtbot, stub_agent, tmp_path,
    ):
        """The Settings checkbox drops the tree, not the finding's own file.

        Walking a large dataset is wasted work when the project layout is
        not what the question is about, so the switch skips it entirely.
        """
        _set(
            ai_base_url=stub_agent.url,
            ai_include_dataset_structure=False,
        )
        root = _dataset(tmp_path / "my_mri_dataset")
        payload = build_payload(
            severity="err",
            message="RepetitionTime is required.",
            rule_id="evidence.test.no_tree",
            extra={"path": "sub-01/func/sub-01_task-rest_bold.json"},
        )

        dialog = AiExplainDialog(payload, root=root)
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        sent = stub_agent.received                       # type: ignore[attr-defined]
        assert "dataset_structure" not in sent
        assert "RepetitionTime: 2.0" in sent["file_content"]

    def test_the_file_content_can_be_turned_off(
        self, qtbot, stub_agent, tmp_path,
    ):
        """The two evidence checkboxes are independent: dropping the file
        digest keeps the dataset tree, and vice versa."""
        _set(
            ai_base_url=stub_agent.url,
            ai_include_file_content=False,
        )
        root = _dataset(tmp_path / "my_mri_dataset")
        payload = build_payload(
            severity="err",
            message="RepetitionTime is required.",
            rule_id="evidence.test.no_file_content",
            extra={"path": "sub-01/func/sub-01_task-rest_bold.json"},
        )

        dialog = AiExplainDialog(payload, root=root)
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        sent = stub_agent.received                       # type: ignore[attr-defined]
        assert "file_content" not in sent
        assert "dataset_structure" in sent

    def test_no_dataset_means_no_evidence_fields(self, qtbot, stub_agent):
        """A finding with nothing behind it — no path, no root — must
        send neither field, not an empty section with a header and
        nothing under it."""
        _set(ai_base_url=stub_agent.url)
        payload = build_payload(
            severity="warn", message="Something is odd.",
            rule_id="evidence.test.none",
        )

        dialog = AiExplainDialog(payload)
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        sent = stub_agent.received                       # type: ignore[attr-defined]
        assert "dataset_structure" not in sent
        assert "file_content" not in sent

    def test_ask_is_a_no_op_when_the_feature_is_off(self):
        _set(ai_enabled=False)
        assert ask(build_payload(severity="err", message="x")) is None


# =====================================================================
# The dialog
# =====================================================================


class TestTheDialog:
    def test_the_agent_answer_lands_in_the_window(
        self, qtbot, stub_agent,
    ):
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(
            build_payload(
                severity="err",
                message="RepetitionTime is required.",
                rule_id="entity.required",
                extra={"path": "sub-01/func"},
            ),
        )
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()

        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()
        # What the answer was made with reads under the message itself
        # now — the per-message footer — not a summary row for the whole
        # conversation.
        assert "intent: explain" in dialog._answer.toPlainText()
        assert dialog._meta.text() == ""
        assert dialog._error.isHidden()
        # What came back is only half the contract; the finding itself
        # has to have crossed in the shape the agent documents.
        sent = stub_agent.received
        assert sent["user_input"] == "Explain this error to me."
        assert sent["context"]["rule_id"] == "entity.required"
        assert sent["context"]["path"] == "sub-01/func"

    def test_the_first_question_speaks_of_an_error_or_a_warning(self, qtbot):
        """The finding's own words, so the question reads as if the user
        wrote it — and the intent guard routes it without spending a
        model call on deciding what was asked."""
        assert ai_explainer.default_prompt({"severity": "err"}) == (
            "Explain this error to me."
        )
        assert ai_explainer.default_prompt({"severity": "warn"}) == (
            "Explain this warning to me."
        )
        assert ai_explainer.default_prompt({"severity": "info"}) == (
            ai_explainer.DEFAULT_PROMPT
        )
        assert ai_explainer.default_prompt({}) == ai_explainer.DEFAULT_PROMPT

        dialog = AiExplainDialog(
            build_payload(severity="warn", message="x", rule_id="a.b"),
        )
        qtbot.addWidget(dialog)
        assert dialog._default_prompt == "Explain this warning to me."

    def test_the_waiting_line_counts_the_seconds(self, qtbot):
        """The footer says how long the wait *is*, not just that it is
        happening — but only once the answer is being written: reading a
        checkpoint into memory is not the model answering."""
        dialog = AiExplainDialog(
            build_payload(severity="err", message="x", rule_id="tick.test"),
        )
        qtbot.addWidget(dialog)
        dialog._set_busy(True)
        try:
            assert dialog._status.text() == "Preparing the answer…"
            # The action line says what; the sentence under it says why,
            # which is the part a first-time reader actually asks.
            assert "loads the model" in dialog._status.detail()
            dialog._on_phase("loading", 0.0)
            assert dialog._status.text() == "Loading the model…"
            assert "checkpoint" in dialog._status.detail()
            assert not dialog._ticker.isActive()

            dialog._on_phase("generating", 0.0)
            dialog._busy_since = time.time() - 50
            dialog._tick_status()
            assert dialog._status.text() == "Writing the answer… (50s)"
        finally:
            dialog._set_busy(False)
        assert dialog._status.text() == ""
        assert not dialog._ticker.isActive()

    def test_the_status_line_shows_a_live_glyph_and_a_reason(self, qtbot):
        """A request in flight is a glyph that moves and two lines of
        words, not one flat "Working". Idle means the line is gone."""
        dialog = AiExplainDialog(
            build_payload(severity="err", message="x", rule_id="status.ux"),
        )
        qtbot.addWidget(dialog)
        assert dialog._status.isHidden()

        dialog._set_busy(True)
        try:
            assert not dialog._status.isHidden()
            assert dialog._status._spinner.is_busy()
            assert dialog._status.text()
            assert dialog._status.detail()
        finally:
            dialog._set_busy(False)
        assert dialog._status.isHidden()

    def test_the_clock_waits_for_the_model_to_load(self, qtbot):
        """The first request of a session can spend a minute reading the
        checkpoint; that minute is not answer time and is not shown as
        such. Generation reports how long it has been running, so the
        clock starts where the answer did, not where the poll landed."""
        dialog = AiExplainDialog(
            build_payload(severity="err", message="x", rule_id="clock.load"),
        )
        qtbot.addWidget(dialog)
        dialog._set_busy(True)
        try:
            dialog._on_phase("loading", 0.0)
            assert dialog._ticker.isActive() is False
            dialog._on_phase("generating", 12.0)
            assert dialog._ticker.isActive()
            assert 11.0 <= time.time() - dialog._busy_since <= 13.0
        finally:
            dialog._set_busy(False)

    def test_the_finding_stays_on_screen_alongside_it(
        self, qtbot, stub_agent,
    ):
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(
            build_payload(
                severity="err",
                message="Missing suffix.",
                rule_id="suffix.unknown",
                extra={"path": "sub-01/anat"},
            ),
        )
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        facts = {
            w.text()
            for w in dialog.findChildren(QLabel)
            if w.objectName() == "ai-fact"
        }
        assert "suffix.unknown" in facts
        assert "sub-01/anat" in facts

    def test_a_service_that_is_not_running_says_where_to_start_it(
        self, qtbot, dead_url,
    ):
        _set(ai_base_url=dead_url, ai_timeout=5)
        dialog = AiExplainDialog(
            build_payload(severity="err", message="x", rule_id="a.b"),
        )
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()

        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        assert not dialog._error.isHidden()
        text = dialog._error.text()
        assert "Could not reach" in text
        # The instruction is now a place in the app, not a terminal
        # command: BIDS-Manager is the one that starts the agent.
        assert "python main.py" not in text
        assert "Settings -> AI Agent" in text
        # The window is not left in a dead state: they can try again.
        assert dialog._retry_btn.isEnabled()

    def test_the_follow_up_field_sends_your_own_question(
        self, qtbot, stub_agent,
    ):
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(
            build_payload(severity="err", message="x", rule_id="a.b"),
        )
        qtbot.addWidget(dialog)
        dialog.show()

        dialog._followup.setText("What is TR?")
        send = next(
            b for b in dialog.findChildren(QPushButton)
            if b.objectName() == "val-ai"
        )
        qtbot.mouseClick(send, Qt.MouseButton.LeftButton)

        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()
        # Sent as the user's words, not the canned default prompt.
        assert stub_agent.received["user_input"] == "What is TR?"
        assert stub_agent.received["context"]["rule_id"] == "a.b"

    def test_closing_mid_answer_does_not_explode(self, qtbot, stub_agent):
        """The one thing everybody does: bail out while it is thinking.

        The worker is held by the module, not the dialog, precisely so
        that destroying the window under a running thread is survivable.
        """
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(
            build_payload(severity="err", message="x", rule_id="a.b"),
        )
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        dialog.close()
        dialog.deleteLater()
        qtbot.wait(100)


# =====================================================================
# The per-message footer
# =====================================================================

# Each footer and download test asks about its own complaint, so a
# conversation one of them wrote is never the one another one restores.
_FOOTER_FINDINGS = 0


def _unique_finding(**extra) -> dict:
    global _FOOTER_FINDINGS
    _FOOTER_FINDINGS += 1
    payload = dict(
        severity="err",
        message=f"Footer finding {_FOOTER_FINDINGS}.",
        rule_id=f"footer.unique_{_FOOTER_FINDINGS}",
        **extra,
    )
    return build_payload(**payload)


class TestTheAnswerFooter:
    def _answered_dialog(self, qtbot, stub_agent) -> AiExplainDialog:
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(_unique_finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        return dialog

    def test_the_footer_reports_what_the_answer_was_made_with(
        self, qtbot, stub_agent,
    ):
        """The checkpoint, its time, the tokens it produced, the intent
        and the button to its record read under the message itself —
        not in a summary row for the whole conversation. How long the
        model took to load is not here: it belongs to the record."""
        dialog = self._answered_dialog(qtbot, stub_agent)
        plain = dialog._answer.toPlainText()
        assert "intent: explain" in plain
        assert "42 tokens" in plain
        assert "model load" not in plain
        # The checkpoint leads the line, then the time, then the numbers.
        footer = dialog._meta_html(dialog._history[1])
        assert footer.startswith("stub/stub-7b-instruct")
        # The record opens from a real button under the answer, not from
        # a word buried in the footer text.
        assert [b.text() for b in dialog._answer.findChildren(QPushButton)
                if b.objectName() == "ai-log"] == ["Show full data"]
        # The summary row only ever says what is true of the
        # conversation as a whole: with one answer, nothing.
        assert dialog._meta.text() == ""

    def test_a_second_answer_counts_the_answers(
        self, qtbot, stub_agent,
    ):
        dialog = self._answered_dialog(qtbot, stub_agent)
        dialog._append("user", "And TRA?")
        dialog._on_done({
            "response": "TRA is the time to acquire.", "intent": "explain",
            "model": "stub/stub-7b-instruct", "generated_tokens": 9,
            "response_seconds": 0.5,
            "log": {"system_prompt": "S", "user_prompt": "And TRA?",
                    "output": "TRA is the time to acquire."},
        })
        assert dialog._meta.text() == "2 answers in this conversation"
        # Each footer carries its own button to its own record.
        assert len([b for b in dialog._answer.findChildren(QPushButton)
                    if b.objectName() == "ai-log"]) == 2

    def test_the_button_under_an_answer_opens_its_record(
        self, qtbot, stub_agent, monkeypatch,
    ):
        """A button, not a link: the whole record used to hide behind a
        word in the footer, and a word is easy not to see."""
        dialog = self._answered_dialog(qtbot, stub_agent)
        opened: list[dict] = []
        monkeypatch.setattr(
            ai_explainer.AiLogDialog, "exec",
            lambda self: opened.append(self._snapshot),
        )
        buttons = [b for b in dialog._answer.findChildren(QPushButton)
                   if b.objectName() == "ai-log"]
        assert len(buttons) == 1
        buttons[0].click()
        assert opened and opened[0]["system_prompt"] == "SYSTEM PROMPT"
        assert opened[0]["output"] == "RepetitionTime is the TR of your scan."
        assert opened[0]["generated_tokens"] == 42
        # The record repeats the load time beside the response time.
        assert opened[0]["model_load_seconds"] == 4.5
        # Closing the record leaves the conversation where it was: a
        # window that comes back empty is a window that has to be asked
        # the same thing twice.
        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()

    def test_the_full_record_holds_the_load_time(self, qtbot):
        """The footer leaves the load out; the record keeps it, beside
        the response time, where it explains a cold first answer."""
        dialog = ai_explainer.AiLogDialog({
            "generation_seconds": 2.0,
            "model_load_seconds": 4.5,
            "model": "stub/stub-7b-instruct",
            "system_prompt": "S",
            "user_prompt": "Q",
            "output": "A",
        })
        qtbot.addWidget(dialog)
        captions = [
            w.text() for w in dialog.findChildren(QLabel)
            if w.objectName() == "val-caption"
        ]
        values = [
            w.text() for w in dialog.findChildren(QLabel)
            if w.objectName() != "val-caption"
        ]
        assert "Response time" in captions
        assert "Model load" in captions
        assert "2.0 s" in values
        assert "4.5 s" in values

    def test_a_turn_without_a_record_opens_nothing(
        self, qtbot, stub_agent, monkeypatch,
    ):
        dialog = self._answered_dialog(qtbot, stub_agent)
        monkeypatch.setattr(
            ai_explainer.AiLogDialog, "exec", lambda self: pytest.fail(
                "something with no record must not open a window",
            ),
        )
        dialog._open_log(99)     # nobody
        dialog._open_log(0)      # the question, not an answer


# =====================================================================
# The copy that outlives the window
# =====================================================================


class TestTheChatLogOnDisk:
    """The window's own store restores a conversation while the app runs;
    these cover the file written beside it, which is what a user can open
    after the app is closed."""

    def _answered(self, qtbot, stub_agent, payload) -> AiExplainDialog:
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(payload)
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        return dialog

    def _lines(self, tmp_path):
        files = sorted((tmp_path / "chat-logs").glob("*.jsonl"))
        assert len(files) == 1
        return [
            json.loads(line)
            for line in files[0].read_text(encoding="utf-8").splitlines()
        ]

    def test_a_conversation_is_written_one_file_per_finding(
        self, qtbot, stub_agent, tmp_path,
    ):
        """The question carries the finding; the answer carries the
        record — the part that explains what the answer was made of."""
        self._answered(qtbot, stub_agent, _unique_finding())

        lines = self._lines(tmp_path)

        assert [line["role"] for line in lines] == ["user", "assistant"]
        assert lines[0]["text"] == "Explain this error to me."
        assert lines[0]["finding"]["severity"] == "err"
        assert lines[1]["text"] == "RepetitionTime is the TR of your scan."
        assert lines[1]["meta"]["model"] == "stub/stub-7b-instruct"
        assert lines[1]["meta"]["log"]["system_prompt"] == "SYSTEM PROMPT"

    def test_reopening_a_finding_does_not_write_it_again(
        self, qtbot, stub_agent, tmp_path,
    ):
        """Restoring is a read. A second copy of the same transcript
        would make the file claim the question was asked twice."""
        payload = _unique_finding()
        self._answered(qtbot, stub_agent, payload)

        restored = AiExplainDialog(payload)
        qtbot.addWidget(restored)

        assert restored.restored()
        assert len(self._lines(tmp_path)) == 2


# =====================================================================
# The download that runs before the question
# =====================================================================


class TestTheDownloadBeforeTheQuestion:
    def test_a_model_that_is_not_local_is_fetched_first(
        self, qtbot, stub_agent,
    ):
        """The configured model is not downloaded: it is fetched
        visibly, with a progress bar, and no question is sent while
        gigabytes are arriving — so no request timer can fire."""
        server = stub_agent.server
        server.models_payload = {"models": [{
            "name": "stub-7b-instruct", "description": "A stub",
            "size": "7B", "vram_gb": "", "downloaded": False,
        }]}
        server.download_status = {
            "name": "stub-7b-instruct", "status": "downloading",
            "downloaded_bytes": 400_000_000, "total_bytes": 3_400_000_000,
            "error": "",
        }

        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(_unique_finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()

        # The bar appears while the download runs, before any question.
        qtbot.waitUntil(lambda: dialog._download_bar.isVisible(),
                        timeout=5_000)
        assert stub_agent.requests == 0
        # The first poll lands a moment after the bar does: the note
        # goes from "looking up the size" to bytes out of bytes.
        qtbot.waitUntil(
            lambda: "of" in dialog._download_note.text()
            and "%" in dialog._download_note.text(),
            timeout=5_000,
        )

        # The fetch finishes: the question goes out, the bar goes away.
        server.download_status = {
            **server.download_status, "status": "done",
        }
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        assert stub_agent.requests == 1
        assert not dialog._download_bar.isVisible()
        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()

    def test_a_failed_download_is_reported_not_swallowed(
        self, qtbot, stub_agent,
    ):
        server = stub_agent.server
        server.models_payload = {"models": [{
            "name": "stub-7b-instruct", "description": "A stub",
            "size": "7B", "vram_gb": "", "downloaded": False,
        }]}
        server.download_status = {
            "name": "stub-7b-instruct", "status": "error",
            "downloaded_bytes": 0, "total_bytes": None,
            "error": "network gone",
        }

        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(_unique_finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()

        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        assert stub_agent.requests == 0
        assert not dialog._error.isHidden()
        assert "network gone" in dialog._error.text()
        # The window is not left in a dead state: they can try again.
        assert dialog._retry_btn.isEnabled()

    def test_a_restarted_agent_ends_the_wait_instead_of_spinning(
        self, qtbot, stub_agent,
    ):
        """The download job lives in the agent's process. If it restarts,
        the status goes back to idle; the dialog must say so instead of
        polling a bar that can never move again."""
        server = stub_agent.server
        server.models_payload = {"models": [{
            "name": "stub-7b-instruct", "description": "A stub",
            "size": "7B", "vram_gb": "", "downloaded": False,
        }]}
        # The stub's default status is exactly what a new process reports:
        # no job, no name.
        server.download_status = {
            "name": "", "status": "idle", "downloaded_bytes": 0,
            "total_bytes": None, "error": "",
        }

        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(_unique_finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()

        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        assert stub_agent.requests == 0
        assert not dialog._error.isHidden()
        assert "restarted" in dialog._error.text()
        assert not dialog._download_bar.isVisible()
        assert dialog._retry_btn.isEnabled()

    def test_a_model_outside_the_list_is_fetched_by_a_modern_agent(
        self, qtbot, stub_agent,
    ):
        """A repo the agent's shortlist does not mention (a custom model
        id) is still fetched first: a modern agent flags everything it
        has, so its silence about this one means it is not local."""
        server = stub_agent.server
        server.models_payload = {"models": [{
            "name": "other-model", "description": "Also a stub",
            "size": "1B", "vram_gb": "", "downloaded": True,
        }]}
        server.download_status = {
            "name": "stub-7b-instruct", "status": "done",
            "downloaded_bytes": 1, "total_bytes": 1, "error": "",
        }

        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(_unique_finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()

        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        assert server.download_started["name"] == "stub-7b-instruct"
        assert stub_agent.requests == 1
        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()

    def test_a_local_model_skips_the_download(
        self, qtbot, stub_agent,
    ):
        """The agent's list says the model is already there: the
        question goes out as an ordinary one, no bar, no fetch."""
        server = stub_agent.server
        server.models_payload = {"models": [{
            "name": "stub-7b-instruct", "description": "A stub",
            "size": "7B", "vram_gb": "", "downloaded": True,
        }]}

        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(_unique_finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()

        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        assert stub_agent.requests == 1
        assert not dialog._download_bar.isVisible()
        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()


# =====================================================================
# The conversation
# =====================================================================


# Each conversation test asks about its own complaint, so that the
# transcript one of them writes is never the one another one reads.
_CONVERSATION_FINDINGS = 0


class TestTheConversation:
    def _finding(self) -> dict:
        """A complaint no other test has ever asked about.

        The transcript is deliberately remembered for the life of the
        process, keyed by the finding — so a shared key would hand this
        test somebody else's turns, and "restored, so I did not
        regenerate" would then be true for the wrong reason. Each call
        mints a new identity; a test that needs the *same* finding twice
        calls this once and keeps the result.
        """
        global _CONVERSATION_FINDINGS
        _CONVERSATION_FINDINGS += 1
        return build_payload(
            severity="err",
            message="x",
            rule_id=f"conversation.test.{_CONVERSATION_FINDINGS}",
            extra={"path": "sub-01/func"},
        )

    def test_every_question_and_answer_stays_on_screen(
        self, qtbot, stub_agent,
    ):
        """The whole complaint: a chat that only ever shows the latest
        message is not a chat, it is a slot being overwritten."""
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(self._finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        dialog._followup.setText("What is TR?")
        send = next(
            b for b in dialog.findChildren(QPushButton)
            if b.objectName() == "val-ai"
        )
        qtbot.mouseClick(send, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        shown = dialog._answer.toPlainText()
        assert "Explain this error to me." in shown
        assert "What is TR?" in shown
        # Both answers, not just the one that arrived last.
        assert shown.count("RepetitionTime is the TR") == 2
        # In order, and attributed: that is what makes it read as an
        # exchange rather than a wall of text.
        first_you = shown.index("You")
        first_agent = shown.index("AI Agent")
        second_you = shown.index("You", first_you + 1)
        second_agent = shown.index("AI Agent", first_agent + 1)
        assert first_you < first_agent < second_you < second_agent

    def test_the_second_question_is_sent_with_the_first(
        self, qtbot, stub_agent,
    ):
        """Otherwise "And why does it matter?" is asked about nothing —
        the model sees a bare question with no idea what came before."""
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(self._finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        dialog._followup.setText("What is TR?")
        send = next(
            b for b in dialog.findChildren(QPushButton)
            if b.objectName() == "val-ai"
        )
        qtbot.mouseClick(send, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        history = stub_agent.received.get("history")
        assert isinstance(history, list) and history
        assert history[0] == {
            "role": "user",
            "content": "Explain this error to me.",
        }
        assert history[1]["role"] == "assistant"
        # The new question stays its own field rather than joining them.
        assert stub_agent.received["user_input"] == "What is TR?"

    def test_ask_again_re_asks_without_asking_the_question_twice(
        self, qtbot, stub_agent,
    ):
        """The button's whole job is to try again, not to ask again.

        It used to re-send the question on top of a history that already
        held that same question and its answer, so the model was handed
        the identical words twice with an answer already attached — and
        answered as if starting over: "give me the error", with the
        error in the context the whole time.
        """
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(self._finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        dialog._retry_btn.click()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        sent = stub_agent.received
        assert sent["user_input"] == "Explain this error to me."
        # Nothing came before this question, so nothing travels: the
        # request is indistinguishable from the first one.
        assert "history" not in sent
        # ...and the screen still holds one question and one answer.
        assert [t["role"] for t in dialog._history] == ["user", "assistant"]

    def test_ask_again_replaces_the_answer_it_is_re_asking_about(
        self, qtbot, stub_agent,
    ):
        """A re-ask regenerates an answer; it does not stack a second.

        Both the question and the answer being replaced are held back
        from the request: what travels is the conversation that came
        before them, so the model answers the question again instead of
        reading an answer it has already given as the reply it owes.
        """
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(self._finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        dialog._followup.setText("What is TR?")
        send = next(
            b for b in dialog.findChildren(QPushButton)
            if b.objectName() == "val-ai"
        )
        qtbot.mouseClick(send, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        dialog._retry_btn.click()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        sent = stub_agent.received
        assert sent["user_input"] == "What is TR?"
        assert sent["history"] == [
            {"role": "user", "content": "Explain this error to me."},
            {"role": "assistant",
             "content": "RepetitionTime is the TR of your scan."},
        ]
        # One copy of each turn on screen: the new answer took the old
        # one's place rather than landing under it.
        assert [t["role"] for t in dialog._history] == [
            "user", "assistant", "user", "assistant",
        ]

    def test_a_failed_ask_again_keeps_the_answer_it_was_replacing(
        self, qtbot, stub_agent, dead_url,
    ):
        """Trying again must not cost the answer already on screen.

        Nothing about the transcript changes until the new answer
        arrives, so a retry into a dead agent leaves the conversation
        exactly where it was — one click from trying once more.
        """
        _set(ai_base_url=stub_agent.url)
        dialog = AiExplainDialog(self._finding())
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.start()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()

        _set(ai_base_url=dead_url, ai_timeout=5)
        dialog._retry_btn.click()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)

        assert not dialog._error.isHidden()
        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()
        assert [t["role"] for t in dialog._history] == ["user", "assistant"]

    def test_reopening_a_finding_restores_it_and_generates_nothing(
        self, qtbot, stub_agent,
    ):
        """The second half of the complaint: coming back to an error
        already read must not spend a minute of model time handing over
        a *different* answer to the same question."""
        _set(ai_base_url=stub_agent.url)
        payload = self._finding()

        first = ask(payload)
        assert first is not None
        qtbot.waitUntil(lambda: not first.is_busy(), timeout=10_000)
        first.close()
        qtbot.wait(50)
        assert stub_agent.requests == 1

        second = ask(payload)
        assert second is not None
        qtbot.addWidget(second)

        assert second.restored()
        assert not second.is_busy()
        assert "Explain this error to me." in second._answer.toPlainText()
        assert "RepetitionTime is the TR" in second._answer.toPlainText()
        assert "restored" in second._status.text().lower()
        # The one request above is still the only one.
        assert stub_agent.requests == 1

    def test_a_fresh_finding_still_generates(self, qtbot, stub_agent):
        """Restore must not become "never asks again": a finding that
        has not been asked about before still has to be answered."""
        _set(ai_base_url=stub_agent.url)

        dialog = ask(self._finding())
        assert dialog is not None
        qtbot.addWidget(dialog)

        assert not dialog.restored()
        qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=10_000)
        assert stub_agent.requests == 1
        assert "RepetitionTime is the TR" in dialog._answer.toPlainText()

    def test_a_finding_that_never_got_an_answer_is_not_remembered(
        self, qtbot, dead_url,
    ):
        """A failed ask leaves no half-conversation behind: the question
        goes back into the box it came from, one Enter from a retry."""
        _set(ai_base_url=dead_url, ai_timeout=5)
        payload = self._finding()

        first = AiExplainDialog(payload)
        qtbot.addWidget(first)
        first.show()
        first.start()
        qtbot.waitUntil(lambda: not first.is_busy(), timeout=10_000)
        assert not first._error.isHidden()

        second = AiExplainDialog(payload)
        qtbot.addWidget(second)

        assert not second.restored()
        assert second._answer.toPlainText().strip() == ""
        # ...and the user's typing is not what got lost: it is back in
        # the box, one Enter from a retry.
        assert first._followup.text() == "Explain this error to me."


# =====================================================================
# Settings
# =====================================================================


class TestSettings:
    def test_the_three_knobs_survive_a_round_trip(self):
        s = AppSettings()
        s.ai_enabled = False
        s.ai_base_url = "http://other-box:9100"
        s.ai_timeout = 45
        s.save()

        after = AppSettings.load()
        assert after.ai_enabled is False
        assert after.ai_base_url == "http://other-box:9100"
        assert after.ai_timeout == 45

    def test_a_stored_zero_timeout_heals_instead_of_raising(self):
        """>>> urllib.request.urlopen(url, timeout=0) raises on the spot.

        Which reads as a crash rather than a slow agent, so load repairs
        anything below the floor.
        """
        AppSettings().save()
        settings = AppSettings._settings()
        settings.setValue(KEYS["ai_timeout"], 0)
        settings.sync()
        assert AppSettings.load().ai_timeout == 120

    def test_a_stored_choice_of_the_base_checkpoint_is_corrected(self):
        """google/gemma-3-270m is the pre-*training* checkpoint.

        It ships no chat template, so asking it a question raised and the
        user got an HTML error page. The -it sibling is the same size and
        the one that can converse; a choice made from the old model list
        is repaired rather than carried forward broken.
        """
        _set(ai_model="google/gemma-3-270m")

        assert AppSettings.load().ai_model == "google/gemma-3-270m-it"

    def test_any_other_model_is_left_alone(self):
        """A repair that rewrites things it was not asked about is worse
        than the bug it fixes."""
        _set(ai_model="Qwen/Qwen3-4B")

        assert AppSettings.load().ai_model == "Qwen/Qwen3-4B"

    def test_the_dialog_reads_them_back_into_its_widgets(self, qtbot):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        s = AppSettings()
        s.ai_enabled = False
        s.ai_python = "/opt/python-agent/bin/python"
        s.ai_timeout = 45
        s.ai_max_new_tokens = 512
        s.ai_device_map = "cpu"
        s.ai_quantization = "4bit"
        s.ai_enable_thinking = True
        s.ai_temperature = 0.35
        s.ai_system_prompt = "Custom instructions."
        s.ai_include_dataset_structure = False
        s.ai_include_file_content = False
        dialog = SettingsDialog(s)
        qtbot.addWidget(dialog)

        assert not dialog._ai_enabled.isChecked()
        assert dialog._ai_python.text() == "/opt/python-agent/bin/python"
        assert dialog._ai_timeout.value() == 45
        assert dialog._ai_tokens.value() == 512
        assert dialog._ai_device.currentData() == "cpu"
        assert dialog._ai_quant.currentData() == "4bit"
        assert dialog._ai_think.isChecked()
        assert dialog._ai_temperature.value() == pytest.approx(0.35)
        assert dialog._ai_system_prompt.toPlainText() == "Custom instructions."
        assert not dialog._ai_dataset_structure.isChecked()
        assert not dialog._ai_file_content.isChecked()
        # There is deliberately no URL field: the process that starts the
        # service already knows where it is.
        assert not hasattr(dialog, "_ai_url")

    def test_the_wheel_scrolls_the_page_instead_of_changing_a_dropdown(
        self, qtbot,
    ):
        """Hovering an AI dropdown and scrolling must not pick a value."""
        from PyQt6.QtCore import QPoint, QPointF
        from PyQt6.QtGui import QWheelEvent
        from PyQt6.QtWidgets import QApplication, QScrollArea

        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._ai_model.clear()
        dialog._ai_model.addItems(["first", "second"])
        dialog._ai_model.setCurrentIndex(0)

        for combo in (
            dialog._ai_model, dialog._ai_quant, dialog._ai_device,
        ):
            combo.setCurrentIndex(0)
            before = combo.currentIndex()
            event = QWheelEvent(
                QPointF(combo.rect().center()),
                QPointF(combo.mapToGlobal(combo.rect().center())),
                QPoint(0, 0),
                QPoint(0, -120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(combo, event)
            assert combo.currentIndex() == before

        # The event is not merely swallowed: the page underneath moves.
        combo = dialog._ai_quant
        parent = combo.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        assert parent is not None
        bar = parent.verticalScrollBar()
        bar.setValue(0)
        event = QWheelEvent(
            QPointF(combo.rect().center()),
            QPointF(combo.mapToGlobal(combo.rect().center())),
            QPoint(0, 0),
            QPoint(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(combo, event)
        assert bar.value() > 0

    def test_the_latest_builtin_prompt_is_shown(self, qtbot, monkeypatch):
        """The box answers "what does the agent see?" without a click."""
        from bidsmgr.gui import settings_dialog

        monkeypatch.setattr(
            settings_dialog, "_installed_system_prompt",
            lambda: "BUILT-IN PROMPT",
        )
        dialog = settings_dialog.SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        assert dialog._ai_system_prompt_builtin == "BUILT-IN PROMPT"
        assert dialog._ai_system_prompt.toPlainText() == "BUILT-IN PROMPT"

        dialog._ai_system_prompt.setPlainText("Edited prompt.")
        dialog._reset_ai_system_prompt()
        assert dialog._ai_system_prompt.toPlainText() == "BUILT-IN PROMPT"

    def test_an_untouched_builtin_prompt_is_not_stored_as_an_override(
        self, qtbot, monkeypatch,
    ):
        """Showing the built-in text must not pin a copy of it on Save."""
        from bidsmgr.gui import settings_dialog

        monkeypatch.setattr(
            settings_dialog, "_installed_system_prompt",
            lambda: "BUILT-IN PROMPT",
        )
        dialog = settings_dialog.SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._on_save()

        assert AppSettings.load().ai_system_prompt == ""

    def test_the_builtin_prompt_is_read_from_the_installed_agent(
        self, tmp_path, monkeypatch,
    ):
        from bidsmgr.gui import settings_dialog

        (tmp_path / "system_prompt.py").write_text(
            'BIDS_SYSTEM_PROMPT = """\\nRead from disk.\\n"""\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            settings_dialog.agent_installer, "install_dir", lambda: tmp_path,
        )

        assert settings_dialog._installed_system_prompt() == "Read from disk."

    def test_saving_writes_the_model_knobs_back(self, qtbot):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._ai_python.setText("C:/python-agent/python.exe")
        dialog._ai_tokens.setValue(768)
        dialog._ai_device.setCurrentIndex(
            dialog._ai_device.findData("cuda")
        )
        dialog._ai_quant.setCurrentIndex(
            dialog._ai_quant.findData("8bit")
        )
        dialog._ai_think.setChecked(True)
        dialog._ai_temperature.setValue(1.25)
        dialog._ai_system_prompt.setPlainText("Custom instructions.")
        dialog._ai_dataset_structure.setChecked(False)
        dialog._ai_file_content.setChecked(False)

        dialog._on_save()

        after = AppSettings.load()
        assert after.ai_python == "C:/python-agent/python.exe"
        assert after.ai_max_new_tokens == 768
        assert after.ai_device_map == "cuda"
        assert after.ai_quantization == "8bit"
        assert after.ai_enable_thinking is True
        assert after.ai_temperature == pytest.approx(1.25)
        assert after.ai_system_prompt == "Custom instructions."
        assert after.ai_include_dataset_structure is False
        assert after.ai_include_file_content is False

    def test_a_model_the_user_picks_is_saved(self, qtbot):
        """The read-back selection is not a choice; an activated row is."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._ai_model.addItem("picked-model", "picked-model")
        dialog._ai_model.setCurrentIndex(
            dialog._ai_model.findData("picked-model")
        )
        dialog._on_ai_model_picked(0)

        dialog._on_save()

        assert AppSettings.load().ai_model == "picked-model"

    def test_both_folders_are_shown_and_saved(self, qtbot, tmp_path):
        """Where models land and where conversations are written — the
        two paths a download and a chat log actually use — are on the
        page, editable, and saved."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        models = tmp_path / "models-here"
        logs = tmp_path / "logs-here"
        s = AppSettings()
        s.ai_enabled = False
        s.ai_models_dir = str(models)
        s.ai_chat_log_dir = str(logs)
        dialog = SettingsDialog(s)
        qtbot.addWidget(dialog)

        assert dialog._ai_models_dir.text() == str(models)
        assert dialog._ai_chat_log_dir.text() == str(logs)

        changed = tmp_path / "logs-elsewhere"
        dialog._ai_chat_log_dir.setText(str(changed))
        dialog._on_save()

        after = AppSettings.load()
        assert after.ai_models_dir == str(models)
        assert after.ai_chat_log_dir == str(changed)

    def test_the_folder_boxes_show_the_path_in_force(self, qtbot):
        """An unset folder means the per-user default. The boxes name
        that folder rather than hiding it in a placeholder, and saving
        an untouched box stores "" so it keeps following the machine."""
        from bidsmgr import agent_service
        from bidsmgr.ai_chat_log import logs_dir
        from bidsmgr.gui.settings_dialog import SettingsDialog

        s = AppSettings()
        s.ai_enabled = False
        dialog = SettingsDialog(s)
        qtbot.addWidget(dialog)

        assert dialog._ai_models_dir.text() == str(
            agent_service.models_dir("")
        )
        assert dialog._ai_chat_log_dir.text() == str(logs_dir(""))

        dialog._on_save()

        after = AppSettings.load()
        assert after.ai_models_dir == ""
        assert after.ai_chat_log_dir == ""

    def test_the_model_dropdown_offers_only_what_the_loader_accepts(
        self, qtbot,
    ):
        """A value the loader would reject can never be selected."""
        from bidsmgr.gui.app_settings import AI_DEVICE_MAPS, AI_QUANTIZATIONS
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        assert tuple(dialog._ai_device.itemData(i) for i in range(
            dialog._ai_device.count()
        )) == AI_DEVICE_MAPS
        assert tuple(dialog._ai_quant.itemData(i) for i in range(
            dialog._ai_quant.count()
        )) == AI_QUANTIZATIONS

    def test_the_tab_is_called_AI_Agent(self, qtbot):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        # The pages hang off the nav list (one row per page, a stack to
        # the right), not a QTabWidget.
        assert "AI Agent" in dialog._page_titles

    def test_connection_reports_the_running_model(self, qtbot, stub_agent):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        _set(ai_base_url=stub_agent.url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._test_agent_connection()

        text = dialog._ai_test_result.text()
        assert "Connected to" in text
        assert "stub-7b-instruct" in text
        # The dropdown is refilled from the same call, so the two can
        # never disagree about what the agent offers.
        assert dialog._ai_model.currentData() == "stub-7b-instruct"

    def test_connection_says_how_to_start_an_agent_that_is_down(
        self, qtbot, dead_url,
    ):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        _set(ai_base_url=dead_url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._test_agent_connection()

        assert "Could not reach" in dialog._ai_test_result.text()

    def test_a_download_that_lost_its_job_stops_asking(
        self, qtbot, stub_agent, fake_agent_service,
    ):
        """The status lives in the agent's process: after a restart the
        slot is idle, and polling it would show a bar at zero forever."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        fake_agent_service.pretend_running(stub_agent.url)
        _set(ai_base_url=stub_agent.url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._refresh_agent_models()
        dialog._ai_model.setCurrentIndex(
            dialog._ai_model.findData("stub-7b-instruct")
        )
        stub_agent.server.download_status = {
            "name": "", "status": "idle", "downloaded_bytes": 0,
            "total_bytes": None, "error": "",
        }

        dialog._download_selected_model()
        assert not dialog._ai_download_btn.isEnabled()

        dialog._poll_model_download()                 # one poll, by hand

        assert "restarted" in dialog._ai_download_info.text()
        assert dialog._ai_download_btn.isEnabled()
        assert not dialog._ai_download_bar.isVisible()
        assert not dialog._ai_download_timer.isActive()

    def test_an_install_from_the_dialog_fills_the_list_once_the_agent_is_up(
        self, qtbot, stub_agent, dead_url, fake_agent_service,
    ):
        """The install starts the agent, which then needs seconds to
        answer. The first list request fails; the status poll has to try
        again when the service says it is serving, or the page keeps an
        empty dropdown."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        _set(ai_base_url=dead_url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._on_agent_installed("C:/agent/.venv/python.exe")

        assert fake_agent_service.started_with[-1] == (
            "C:/agent/.venv/python.exe"
        )
        assert dialog._ai_waiting_for_agent
        assert not dialog._ai_models_fetched

        # Now the agent is serving, and the poll reaches it.
        _set(ai_base_url=stub_agent.url)
        fake_agent_service.pretend_running(stub_agent.url)
        dialog._update_agent_status()

        assert dialog._ai_models_fetched
        assert dialog._ai_model.currentData() == "stub-7b-instruct"
        assert not dialog._ai_waiting_for_agent

    # -- what the answers will be made on ---------------------------

    def test_the_hardware_row_reports_what_the_agent_can_use(
        self, qtbot, stub_agent,
    ):
        """"Automatic" is not an answer on its own; this is."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        _set(ai_base_url=stub_agent.url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._refresh_agent_models()
        # The device the report actually describes, so the row is read
        # in the case a user would hit it: one that matches.
        dialog._ai_device.setCurrentIndex(
            dialog._ai_device.findData("cuda")
        )

        text = dialog._ai_hardware.text()
        assert "Stub GPU" in text
        assert "CUDA 12.4" in text
        assert "8.0 GB" in text
        assert "2.5.1" in dialog._ai_hardware.toolTip()
        assert "color: #d29922" not in dialog._ai_hardware.styleSheet()

    def test_a_device_this_machine_cannot_serve_is_flagged_before_saving(
        self, qtbot, stub_agent,
    ):
        """The round trip this removes: picking Apple GPU, saving, and
        finding out on the next answer that it does not exist here."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        _set(ai_base_url=stub_agent.url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._refresh_agent_models()
        dialog._ai_device.setCurrentIndex(
            dialog._ai_device.findData("mps")
        )

        text = dialog._ai_hardware.text()
        assert "does not have" in text
        assert "Apple GPU (MPS)" in text
        assert "fail to load" in text
        assert "color: #d29922" in dialog._ai_hardware.styleSheet()

    def test_the_hardware_row_admits_when_it_does_not_know(
        self, qtbot, dead_url,
    ):
        """Silence must not be read as "CPU only": that would be a
        confident wrong answer about the machine."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        _set(ai_base_url=dead_url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._refresh_agent_models()

        assert "Not reported yet" in dialog._ai_hardware.text()


# =====================================================================
# Starting the agent
# =====================================================================


class TestStartingTheAgent:
    def test_saving_brings_the_service_up(self, qtbot, fake_agent_service):
        """The first half of the ask: the app starts it, not the user."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._ai_python.setText("/opt/python311/bin/python")

        dialog._on_save()

        assert fake_agent_service.started_with == [
            "/opt/python311/bin/python"
        ]

    def test_a_second_save_while_starting_does_not_double_start(
        self, qtbot, fake_agent_service,
    ):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._on_save()
        dialog._on_save()

        assert len(fake_agent_service.started_with) == 1

    def test_turning_the_button_off_stops_the_service_we_started(
        self, qtbot, fake_agent_service,
    ):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        fake_agent_service.pretend_running("http://127.0.0.1:8000")
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._ai_enabled.setChecked(False)

        dialog._on_save()

        assert fake_agent_service.stop_calls == 1

    def test_a_service_somebody_else_started_is_left_alone(
        self, qtbot, fake_agent_service,
    ):
        """Not ours to kill — the user may have other tools pointed at it."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        fake_agent_service.pretend_running(
            "http://127.0.0.1:8000", owned=False,
        )
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._ai_enabled.setChecked(False)

        dialog._on_save()

        assert fake_agent_service.stop_calls == 0

    def test_saving_pushes_the_knobs_to_a_running_agent(
        self, qtbot, stub_agent, fake_agent_service,
    ):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        fake_agent_service.pretend_running(stub_agent.url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._ai_tokens.setValue(640)
        dialog._ai_quant.setCurrentIndex(
            dialog._ai_quant.findData("8bit")
        )
        dialog._ai_temperature.setValue(0.2)

        dialog._on_save()

        sent = stub_agent.config
        assert sent["max_new_tokens"] == 640
        assert sent["quantization"] == "8bit"
        assert sent["temperature"] == pytest.approx(0.2)
        # A model the user never picked is left out rather than sent
        # blank — a blank would travel all the way to from_pretrained().
        assert "model_name" not in sent

    def test_the_status_row_describes_the_service(
        self, qtbot, fake_agent_service,
    ):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        fake_agent_service.pretend_running(owned=True)
        dialog._update_agent_status()
        assert "started by BIDS-Manager" in dialog._ai_status.text()
        assert dialog._ai_start_btn.text() == "Stop"
        assert dialog._ai_start_btn.isEnabled()

        fake_agent_service.pretend_failed("Could not find a Python for it")
        dialog._update_agent_status()
        assert "Could not find a Python" in dialog._ai_status.text()
        assert dialog._ai_start_btn.text() == "Start"

    def test_a_foreign_agent_is_reported_but_not_offered_a_stop(
        self, qtbot, fake_agent_service,
    ):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        fake_agent_service.pretend_running(owned=False)

        dialog._update_agent_status()

        assert "outside BIDS-Manager" in dialog._ai_status.text()
        assert not dialog._ai_start_btn.isEnabled()

    def test_pressing_start_asks_the_service_to_start(
        self, qtbot, fake_agent_service,
    ):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._ai_python.setText("C:/py/python.exe")

        dialog._on_agent_start_stop()

        assert fake_agent_service.started_with == ["C:/py/python.exe"]

    def test_the_model_list_comes_from_the_agent(
        self, qtbot, stub_agent, fake_agent_service,
    ):
        from bidsmgr.gui.settings_dialog import SettingsDialog

        fake_agent_service.pretend_running(stub_agent.url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._refresh_agent_models()

        names = [
            dialog._ai_model.itemData(i)
            for i in range(dialog._ai_model.count())
        ]
        assert "stub-7b-instruct" in names
        assert "other-model" in names
        # The agent's own current model is what ends up selected, not
        # the placeholder: the user is being shown what is running.
        assert dialog._ai_model.currentData() == "stub-7b-instruct"

    def test_the_dropdown_flags_what_is_downloaded_and_has_no_placeholder(
        self, qtbot, stub_agent, fake_agent_service, tmp_path, monkeypatch,
    ):
        """The rows are the names plus a ✓ flag; there is no "as
        configured on the agent" row that can be picked by mistake."""
        from bidsmgr.gui import ai_models
        from bidsmgr.gui.settings_dialog import SettingsDialog

        stub_agent.server.models_payload = {          # type: ignore[attr-defined]
            "models": [
                {
                    "name": "ready-model",
                    "description": "Already here",
                    "downloaded": True,
                },
                {
                    "name": "remote-model",
                    "description": "Not here",
                    "downloaded": False,
                },
            ],
        }
        fake_agent_service.pretend_running(stub_agent.url)
        _set(ai_base_url=stub_agent.url)
        empty = tmp_path / "models"
        empty.mkdir()
        monkeypatch.setattr(
            ai_models, "models_dir", lambda override="": empty,
        )
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._refresh_agent_models()

        texts = [
            dialog._ai_model.itemText(i)
            for i in range(dialog._ai_model.count())
        ]
        assert any(
            text.startswith("ready-model") and text.endswith("✔")
            for text in texts
        ), texts
        assert any(text.startswith("remote-model") for text in texts), texts
        assert not any("as configured" in text for text in texts)

    def test_a_stopped_agent_still_lists_what_is_on_disk(
        self, qtbot, dead_url, tmp_path, monkeypatch,
    ):
        """The ✔ marks do not need a running agent: what is on disk is
        what the next start will find, and the page exists to show it."""
        from bidsmgr.gui import ai_models
        from bidsmgr.gui.settings_dialog import SettingsDialog

        root = tmp_path / "models"
        repo = root / "models--Qwen--Qwen3-1.7B"
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("abc123", encoding="utf-8")
        snapshot = repo / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(b"x")
        monkeypatch.setattr(
            ai_models, "models_dir", lambda override="": root,
        )
        _set(ai_base_url=dead_url)

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)

        dialog._refresh_agent_models()

        texts = [
            dialog._ai_model.itemText(i)
            for i in range(dialog._ai_model.count())
        ]
        assert any(
            text.startswith("Qwen/Qwen3-1.7B") and text.endswith("✔")
            for text in texts
        ), texts

    def test_the_knobs_are_corrected_to_what_the_agent_runs(
        self, qtbot, stub_agent, fake_agent_service,
    ):
        """Somebody can edit the agent's own config while the app is
        open. Reading it back is what stops a later Save from quietly
        overwriting their edit with a stale guess."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        fake_agent_service.pretend_running(stub_agent.url)
        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._ai_tokens.setValue(768)
        dialog._ai_temperature.setValue(1.5)
        dialog._ai_device.setCurrentIndex(
            dialog._ai_device.findData("cuda")
        )
        dialog._ai_quant.setCurrentIndex(
            dialog._ai_quant.findData("none")
        )
        dialog._ai_think.setChecked(False)

        dialog._refresh_agent_models()

        assert dialog._ai_tokens.value() == 123
        assert dialog._ai_temperature.value() == pytest.approx(0.5)
        assert dialog._ai_device.currentData() == "cpu"
        assert dialog._ai_quant.currentData() == "4bit"
        assert dialog._ai_think.isChecked()

    def test_one_bad_field_leaves_the_rest_alone(self, qtbot):
        """A config the user half-edited by hand must not blank the
        widgets, and the pre-quantization boolean must still mean 4bit."""
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppSettings())
        qtbot.addWidget(dialog)
        dialog._ai_tokens.setValue(444)
        dialog._ai_temperature.setValue(1.5)

        dialog._sync_agent_knobs({
            "load_in_4bit": True,
            "max_new_tokens": "not a number",
            "temperature": 99,          # out of range
            "device_map": "nonsense",   # not a vocabulary value
            "enable_thinking": False,
        })

        assert dialog._ai_quant.currentData() == "4bit"
        assert not dialog._ai_think.isChecked()
        assert dialog._ai_tokens.value() == 444
        assert dialog._ai_temperature.value() == pytest.approx(1.5)
        assert dialog._ai_device.currentData() == "auto"
