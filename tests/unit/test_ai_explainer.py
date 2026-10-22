"""The AI-explainer transport: payload shape, HTML stripping, HTTP.

The agent in ``bidsmgr/ai_agent`` is a separate service, so everything
that can be checked without a GUI or a language model is checked here:
what we send it, what we do with the answer, and — the case that will
actually happen during somebody's first run — what happens when the
service is not up.

Stands up a real ``HTTPServer`` on a loopback port rather than
monkeypatching ``urlopen``: the point of this module is that it speaks
HTTP correctly, and a mocked transport proves only that the mocks were
wired the way the test expected.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest

from bidsmgr.gui.ai_explainer import (
    HISTORY_SENT,
    AgentClient,
    AgentError,
    build_payload,
    conversation_key,
    hardware_text,
    plain_text,
)


# =====================================================================
# Stub agent
# =====================================================================


class _StubAgent(BaseHTTPRequestHandler):
    """Minimal stand-in for ``ai_agent/main.py``.

    Records every ``/predict`` body on the server instance so a test can
    assert on what actually crossed the wire, which is the only place a
    contract mistake can hide.
    """

    def log_message(self, *args) -> None:  # noqa: D102 - silence access log
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                       # noqa: N802
        if self.path == "/models":
            self._send({"models": ["stub-7b-instruct"]})
        elif self.path == "/config":
            self._send({"llm": {"model_name": "stub-7b-instruct"}})
        elif self.path == "/raw":
            # A proxy that answers in HTML: what a wrong URL on a
            # shared machine tends to get you.
            body = b"<html>not the agent</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send({"error": "unknown endpoint"}, 404)

    def do_POST(self) -> None:                      # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw or b"{}")
        self.server.received = body                 # type: ignore[attr-defined]
        self._send({
            "response": f"answer to: {body.get('user_input')}",
            "intent": "explain",
            "tool_used": None,
        })

    def do_PUT(self) -> None:                       # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        self.server.config = json.loads(raw or b"{}")   # type: ignore[attr-defined]
        self._send({})


@pytest.fixture
def stub_agent() -> Iterator[tuple[AgentClient, HTTPServer]]:
    """A running stub agent plus a client pointed at it."""
    server = HTTPServer(("127.0.0.1", 0), _StubAgent)
    server.received = {}                            # type: ignore[attr-defined]
    server.config = {}                              # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield (
            AgentClient(base_url=f"http://{host}:{port}", timeout=5.0),
            server,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def dead_url() -> str:
    """A URL on loopback that nothing is listening on.

    Bind-and-close gets a port the OS just handed out, so it is free by
    the time we use it — the reliable way to force ECONNREFUSED without
    depending on any particular port being unclaimed.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


# =====================================================================
# Payload
# =====================================================================


class TestBuildPayload:
    def test_sends_the_documented_issue_fields(self):
        payload = build_payload(
            severity="err",
            message="RepetitionTime is required.",
            rule_id="entity.required",
            field="RepetitionTime",
            fix_label="Add the missing key",
            extra={"path": "sub-01/func", "datatype": "func"},
        )
        assert payload["severity"] == "err"
        assert payload["rule_id"] == "entity.required"
        assert payload["field"] == "RepetitionTime"
        assert payload["message"] == "RepetitionTime is required."
        assert payload["fix_label"] == "Add the missing key"
        assert payload["mirrored"] is False
        assert payload["path"] == "sub-01/func"
        assert payload["datatype"] == "func"

    def test_omits_absent_optionals_rather_than_sending_null(self):
        """``utils.compact_json`` renders Python's ``None`` as the word
        "None" — a worse thing to hand a language model than no key."""
        payload = build_payload(severity="warn", message="soft problem")
        for key in ("field", "line", "lines", "fix_label", "fix_action"):
            assert key not in payload

    def test_drops_empty_extras(self):
        payload = build_payload(
            severity="err", message="x", extra={"path": "", "row": None},
        )
        assert "path" not in payload
        assert "row" not in payload

    def test_extra_keys_override_nothing_but_always_apply(self):
        payload = build_payload(
            severity="err",
            message="x",
            rule_id="a.b",
            extra={"rule_id": "c.d", "suffix": "bold"},
        )
        assert payload["rule_id"] == "a.b"    # core fields win
        assert payload["suffix"] == "bold"    # extras still land

    def test_lines_are_ints(self):
        payload = build_payload(
            severity="err", message="x", lines=["3", "7", "not-a-number"],
        )
        assert payload["lines"] == [3, 7]


class TestPlainText:
    def test_passes_plain_text_through(self):
        assert plain_text("RepetitionTime is required.") == (
            "RepetitionTime is required."
        )

    def test_unwraps_code_tags(self):
        assert plain_text('Entity <code>RepetitionTime</code> missing.') == (
            "Entity RepetitionTime missing."
        )

    def test_does_not_eat_a_literal_angle_bracket(self):
        """A message containing ``<3`` would be half-swallowed as a tag."""
        assert plain_text("TR must be <3 seconds.") == "TR must be <3 seconds."

    def test_turns_breaks_into_newlines(self):
        assert plain_text("first<br/>second") == "first\nsecond"

    def test_unescapes_entities(self):
        assert plain_text("Tom &amp; Jerry") == "Tom & Jerry"

    def test_empty_is_empty(self):
        assert plain_text(None) == ""
        assert plain_text("") == ""


# =====================================================================
# Transport
# =====================================================================


class TestTransport:
    def test_ping_talks_to_the_cheapest_endpoint(self, stub_agent):
        client, _ = stub_agent
        assert client.ping() == {"models": ["stub-7b-instruct"]}

    def test_config_reports_which_model_is_running(self, stub_agent):
        """Settings shows this next to "Connected", so a user can tell a
        stale agent from a fresh one without reading its console."""
        client, _ = stub_agent
        assert client.config()["llm"]["model_name"] == "stub-7b-instruct"

    def test_predict_posts_both_required_keys(self, stub_agent):
        client, server = stub_agent
        result = client.predict("Explain it", {"rule_id": "a.b"})
        assert result["response"] == "answer to: Explain it"
        assert result["intent"] == "explain"
        # ``tool_used`` came back JSON null; ``_on_done`` must not render
        # that as the string "None".
        assert result["tool_used"] is None

        sent = server.received                         # type: ignore[attr-defined]
        assert sent["user_input"] == "Explain it"
        assert sent["context"] == {"rule_id": "a.b"}

    def test_predict_refuses_an_empty_answer(self, stub_agent):
        client, _ = stub_agent
        client._request = lambda *a, **k: {"response": "   "}  # type: ignore[method-assign]
        with pytest.raises(AgentError, match="empty"):
            client.predict("hi", {})

    def test_a_first_question_carries_no_history(self, stub_agent):
        """Nothing to remember yet, so nothing is sent: an absent key
        and an empty list are different things to a server that treats
        one as "no history" and the other as "unread"."""
        client, server = stub_agent
        client.predict("Explain it", {"rule_id": "a.b"})
        assert "history" not in server.received            # type: ignore[attr-defined]

    def test_a_follow_up_sends_the_earlier_turns(self, stub_agent):
        """This is what makes a follow-up a follow-up rather than a
        second, unrelated question about the same file."""
        client, server = stub_agent
        history = [
            {"role": "user", "content": "What is TR?"},
            {"role": "assistant", "content": "RepetitionTime."},
        ]

        client.predict("And why does it matter?", {"rule_id": "a.b"},
                       history=history)

        assert server.received["history"] == history        # type: ignore[attr-defined]
        # ...and the new question is still its own field, not folded in.
        assert server.received["user_input"] == "And why does it matter?"

    def test_a_long_transcript_is_trimmed_from_the_front(self, stub_agent):
        """A session's worth of turns would otherwise grow the prompt
        without bound — and the oldest turns are the least relevant."""
        client, server = stub_agent
        history = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn {i}"}
            for i in range(HISTORY_SENT + 5)
        ]

        client.predict("now", {"rule_id": "a.b"}, history=history)

        sent = server.received["history"]                  # type: ignore[attr-defined]
        assert len(sent) == HISTORY_SENT
        assert sent[-1]["content"] == f"turn {HISTORY_SENT + 4}"
        # Kept from the front of the *end*, so nothing invented appears.
        assert all(t["content"].startswith("turn ") for t in sent)

    def test_junk_in_the_transcript_never_reaches_the_wire(self, stub_agent):
        """Non-dict entries are dropped rather than serialized as-is."""
        client, server = stub_agent
        client.predict("now", {"rule_id": "a.b"},
                       history=["a string", {"role": "user"}])
        assert "history" not in server.received            # type: ignore[attr-defined]

    def test_unreachable_service_says_where_to_start_it(self, dead_url):
        """Not "start it in a terminal": BIDS-Manager does that now."""
        client = AgentClient(base_url=dead_url, timeout=3.0)
        with pytest.raises(AgentError) as excinfo:
            client.ping()
        message = str(excinfo.value)
        assert "Could not reach" in message
        assert "Settings -> AI Agent" in message

    def test_a_wrong_service_is_diagnosed_as_one(self, stub_agent):
        """``/models`` missing means something else owns that port."""
        client, _ = stub_agent
        client.base_url += "/nope"
        with pytest.raises(AgentError) as excinfo:
            client.ping()
        assert "404" in str(excinfo.value)
        assert "not the BIDS AI agent" in str(excinfo.value)

    def test_a_non_json_body_is_reported_not_swallowed(self, stub_agent):
        client, _ = stub_agent
        # Point the client straight at the stub's HTML endpoint: an
        # empty body is ``{}``, but a body that is not JSON at all must
        # be quoted back rather than swallowed as a parse failure.
        client.base_url = f"{client.base_url}/raw"
        with pytest.raises(AgentError, match="not JSON"):
            client._request("GET", "")

    def test_timeout_is_clamped_never_below_five_seconds(self, monkeypatch):
        import bidsmgr.gui.ai_explainer as mod

        # A stored 0 makes urlopen() raise on construction instead of
        # timing out, which reads as a crash rather than a slow agent.
        monkeypatch.setattr(mod, "_read", lambda key, default: 0)
        assert mod.agent_timeout() == 5.0

    def test_base_url_gets_a_scheme_if_the_user_left_it_off(self, monkeypatch):
        import bidsmgr.gui.ai_explainer as mod

        monkeypatch.setattr(mod, "_read", lambda key, default: "host:9000")
        assert mod.agent_base_url() == "http://host:9000"

    def test_where_the_service_is_beats_where_a_url_says_it_is(
        self, monkeypatch,
    ):
        """If 8000 was already taken, the agent moved to another port —
        and only the service that moved it knows where it went. The
        stored URL is the fallback for an agent nobody here started."""
        from types import SimpleNamespace

        from bidsmgr import agent_service
        import bidsmgr.gui.ai_explainer as mod

        monkeypatch.setattr(
            agent_service, "_SERVICE", SimpleNamespace(url="http://127.0.0.1:8123"),
        )
        monkeypatch.setattr(
            mod, "_read", lambda key, default: "http://stored:8000",
        )

        assert mod.agent_base_url() == "http://127.0.0.1:8123"

    def test_a_stored_url_is_used_when_the_service_is_not_running(
        self, monkeypatch,
    ):
        from types import SimpleNamespace

        from bidsmgr import agent_service
        import bidsmgr.gui.ai_explainer as mod

        monkeypatch.setattr(agent_service, "_SERVICE", SimpleNamespace(url=""))
        monkeypatch.setattr(
            mod, "_read", lambda key, default: "http://stored:8000",
        )

        assert mod.agent_base_url() == "http://stored:8000"


# =====================================================================
# Applying the LLM preferences
# =====================================================================


class TestPushingConfig:
    def test_every_knob_the_settings_tab_offers_is_sent(self):
        from bidsmgr.gui.ai_explainer import llm_config_payload
        from bidsmgr.gui.app_settings import AppSettings

        s = AppSettings()
        s.ai_model = "google/gemma-3-4b-it"
        s.ai_max_new_tokens = 512
        s.ai_device_map = "cuda"
        s.ai_quantization = "8bit"
        s.ai_enable_thinking = True
        s.ai_temperature = 0.4

        assert llm_config_payload(s) == {
            "model_name": "google/gemma-3-4b-it",
            "max_new_tokens": 512,
            "device_map": "cuda",
            "quantization": "8bit",
            "enable_thinking": True,
            "temperature": 0.4,
        }

    def test_an_unpicked_model_is_left_out_rather_than_sent_blank(self):
        """A blank model_name would travel to from_pretrained() as the
        name of a model that does not exist; omitting it keeps the
        agent's own config.json decision."""
        from bidsmgr.gui.ai_explainer import llm_config_payload
        from bidsmgr.gui.app_settings import AppSettings

        payload = llm_config_payload(AppSettings())
        assert "model_name" not in payload

    def test_it_crosses_the_wire_and_reports_failure_quietly(
        self, monkeypatch, stub_agent,
    ):
        import bidsmgr.gui.ai_explainer as mod
        from bidsmgr.gui.app_settings import AppSettings

        client, server = stub_agent
        monkeypatch.setattr(mod, "agent_base_url", lambda: client.base_url)

        s = AppSettings()
        s.ai_quantization = "4bit"
        assert mod.push_llm_config(s) is None
        assert server.config["quantization"] == "4bit"
        assert server.config["max_new_tokens"] == 250

    def test_a_dead_agent_is_quiet_not_an_error_dialog(
        self, monkeypatch, dead_url,
    ):
        """Best effort by design: it may be down because the user
        stopped it, and nobody wants a modal for that."""
        import bidsmgr.gui.ai_explainer as mod
        from bidsmgr.gui.app_settings import AppSettings

        monkeypatch.setattr(mod, "agent_base_url", lambda: dead_url)

        error = mod.push_llm_config(AppSettings())
        assert error                      # returned, not raised
        assert "Could not reach" in error


# =====================================================================
# Which conversation a question belongs to
# =====================================================================


class TestConversationIdentity:
    def test_the_same_finding_is_the_same_conversation(self):
        """Reopening an error has to find what was said about it, and
        the message being reworded between runs must not lose it."""
        first = build_payload(
            severity="err",
            message="<code>RepetitionTime</code> is required.",
            rule_id="entity.required",
            extra={"path": "sub-01/func"},
        )
        second = build_payload(
            severity="err",
            message="RepetitionTime is required for this file.",
            rule_id="entity.required",
            extra={"path": "sub-01/func"},
        )
        assert conversation_key(first) == conversation_key(second)

    def test_another_file_is_another_conversation(self):
        one = build_payload(severity="err", message="x",
                            rule_id="a.b", extra={"path": "sub-01/func"})
        two = build_payload(severity="err", message="x",
                            rule_id="a.b", extra={"path": "sub-01/anat"})
        assert conversation_key(one) != conversation_key(two)

    def test_another_rule_is_another_conversation(self):
        one = build_payload(severity="err", message="x", rule_id="a.b")
        two = build_payload(severity="err", message="x", rule_id="a.c")
        assert conversation_key(one) != conversation_key(two)

    def test_a_finding_with_nothing_to_identify_it_by_still_gets_a_key(self):
        """A coherence warning carries no rule id and sometimes no path.
        The digest is what is left — and two different ones must not
        land in the same transcript."""
        vague = build_payload(severity="warn", message="Files disagree.")
        other = build_payload(severity="warn", message="Something else.")
        assert conversation_key(vague) == conversation_key(
            build_payload(severity="warn", message="Files disagree.")
        )
        assert conversation_key(vague) != conversation_key(other)
        assert conversation_key(vague)


# =====================================================================
# What the hardware row says
# =====================================================================


class TestSayingWhatTheAnswersAreMadeOn:
    def test_no_report_yet_is_not_read_as_a_machine_with_no_gpu(self):
        """Silence is not an answer: guessing "CPU" from an agent that
        has not spoken would be a confident wrong one."""
        for silent in ({}, None, {"backend": ""}):
            text, tip = hardware_text(silent)
            assert "Not reported yet" in text
            assert "CPU only" not in text
            assert "Test connection" in tip

    def test_an_nvidia_card_is_named_with_its_memory(self):
        text, tip = hardware_text({
            "backend": "cuda",
            "device_name": "NVIDIA GeForce RTX 3060",
            "cuda_version": "12.1",
            "memory_gb": 12.0,
            "device_count": 1,
            "torch": "2.5.1",
            "python": "3.10.9",
        })
        assert "NVIDIA GeForce RTX 3060" in text
        assert "CUDA 12.1" in text
        assert "12.0 GB" in text
        # One card is not worth announcing as "1 cards".
        assert "cards" not in text
        assert "2.5.1" in tip

    def test_more_than_one_card_is_counted(self):
        text, _ = hardware_text({"backend": "cuda",
                                 "device_name": "GPU", "device_count": 2})
        assert "2 cards" in text

    def test_apple_silicon_is_recognised(self):
        text, _ = hardware_text({"backend": "mps",
                                 "device_name": "Apple GPU"})
        assert "MPS" in text
        assert "CUDA" not in text

    def test_cpu_only_says_so_and_that_it_is_the_slow_one(self):
        text, tip = hardware_text({"backend": "cpu", "cpu_cores": 8})
        assert "CPU only" in text
        assert "8 cores" in text
        assert "slower" in tip.lower()

    def test_an_unrecognised_backend_is_reported_not_translated(self):
        text, _ = hardware_text({"backend": "tpu", "device_name": "Thing"})
        assert "Thing" in text


# =====================================================================
# What a failure looks like from the window
# =====================================================================


_URL = "http://127.0.0.1:8000/predict"


def _failing(code: int, body: str) -> urllib.error.HTTPError:
    """An ``HTTPError`` carrying *body*, as ``urlopen`` hands it over."""
    return urllib.error.HTTPError(
        _URL, code, "error", {}, io.BytesIO(body.encode("utf-8")),
    )


class TestWhatTheUserSeesWhenItFails:
    def test_the_agents_own_reason_is_what_the_dialog_shows(self):
        """The whole point of the JSON error body: one line naming the
        actual exception beats a page of HTML describing that a page of
        HTML describes an error."""
        from bidsmgr.gui.ai_explainer import _http_error

        body = json.dumps({
            "error": "ValueError: tokenizer.chat_template is not set",
        })

        text = _http_error(_URL, _failing(500, body))

        assert "ValueError" in text
        assert "could not answer" in text

    def test_a_flask_error_page_is_never_dumped_into_the_dialog(self):
        """Its text tells a reader nothing they can act on — which is
        exactly what got past us once, verbatim, as the whole answer."""
        from bidsmgr.gui.ai_explainer import _http_error

        html = (
            "<!doctype html>\n<title>500 Internal Server Error</title>\n"
            "<p>The server encountered an internal error and was unable "
            "to complete the request.</p>"
        )

        text = _http_error(_URL, _failing(500, html))

        assert "500" in text
        assert "<html" not in text
        assert "internal error" not in text.lower()
        # ...and it still says what to do about it.
        assert "Settings -> AI Agent" in text

    def test_a_404_still_says_where_to_start_the_agent(self):
        from bidsmgr.gui.ai_explainer import _http_error

        text = _http_error(_URL, _failing(404, "<html>404 Not Found</html>"))

        assert "404" in text
        assert "Settings -> AI Agent" in text
        assert "<html" not in text

    def test_a_plain_text_body_survives(self):
        """A proxy that answers in words is worth reading; only markup
        and structured errors get rewritten."""
        from bidsmgr.gui.ai_explainer import _http_error

        text = _http_error(_URL, _failing(502, "upstream said no"))

        assert "HTTP 502" in text
        assert "upstream said no" in text

    def test_no_body_at_all_is_still_a_sentence(self):
        from bidsmgr.gui.ai_explainer import _http_error

        text = _http_error(_URL, _failing(500, ""))

        assert "HTTP 500" in text
        assert text.strip()
