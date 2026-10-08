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


class _StubAgent(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:               # noqa: D102
        pass

    def do_GET(self) -> None:                           # noqa: N802
        if self.path == "/models":
            body = json.dumps({"models": [
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
            ]}).encode()
        elif self.path == "/config":
            body = json.dumps({"llm": {
                "model_name": "stub-7b-instruct",
                "max_new_tokens": 123,
                "temperature": 0.5,
                "device_map": "cpu",
                "quantization": "4bit",
                "enable_thinking": True,
            }}).encode()
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
        self.server.received = received                 # type: ignore[attr-defined]
        body = json.dumps({
            "response": "RepetitionTime is the TR of your scan.",
            "intent": "explain",
            "tool_used": None,
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
    def config(self) -> dict:
        """The body of the last ``PUT /config``, as crossed the wire."""
        return self.server.config                     # type: ignore[attr-defined]


@pytest.fixture
def stub_agent() -> Iterator[_Stub]:
    server = HTTPServer(("127.0.0.1", 0), _StubAgent)
    server.received = {}                                # type: ignore[attr-defined]
    server.config = {}                                  # type: ignore[attr-defined]
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
    _set(ai_enabled=True, ai_base_url="http://127.0.0.1:8000", ai_timeout=30)
    yield
    _set(ai_enabled=True, ai_base_url="http://127.0.0.1:8000", ai_timeout=120)


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

        def fake_ask(payload, parent=None):
            seen["payload"] = payload
            seen["parent"] = parent
            return None

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
        assert "explain" in dialog._meta.text()
        assert dialog._error.isHidden()
        # What came back is only half the contract; the finding itself
        # has to have crossed in the shape the agent documents.
        sent = stub_agent.received
        assert sent["user_input"] == ai_explainer.DEFAULT_PROMPT
        assert sent["context"]["rule_id"] == "entity.required"
        assert sent["context"]["path"] == "sub-01/func"

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
        # There is deliberately no URL field: the process that starts the
        # service already knows where it is.
        assert not hasattr(dialog, "_ai_url")

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

        dialog._on_save()

        after = AppSettings.load()
        assert after.ai_python == "C:/python-agent/python.exe"
        assert after.ai_max_new_tokens == 768
        assert after.ai_device_map == "cuda"
        assert after.ai_quantization == "8bit"
        assert after.ai_enable_thinking is True
        assert after.ai_temperature == pytest.approx(1.25)

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
        titles = [
            dialog._tabs.tabText(i) for i in range(dialog._tabs.count())
        ]
        assert "AI Agent" in titles

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
