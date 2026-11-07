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
    DIGEST_MAX_LINES,
    TREE_MAX_LINES,
    AgentClient,
    AgentError,
    build_payload,
    conversation_key,
    dataset_tree,
    file_digest,
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
# Evidence: the dataset and the file behind a finding
# =====================================================================


def _dataset(root) -> dict:
    """A small dataset shaped like a real one, plus its offending file.

    Returns the paths the tests ask about. Deliberately built with the
    same layout the desktop app would find — root, subjects, sessions,
    datatypes — because the tree's whole job is to describe *that*
    shape rather than any shape a test could invent.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "README").write_text("A dataset.\n", encoding="utf-8")
    (root / "dataset_description.json").write_text(
        '{"Name": "Probe", "BIDSVersion": "1.9.0"}', encoding="utf-8")
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\n"
        + "".join(f"sub-{i:02d}\t{20 + i}\tF\n" for i in range(1, 9)),
        encoding="utf-8",
    )
    anat = root / "sub-01/ses-01/anat"
    func = root / "sub-01/ses-01/func"
    anat.mkdir(parents=True, exist_ok=True)
    func.mkdir(parents=True, exist_ok=True)
    (anat / "sub-01_ses-01_T1w.json").write_text(
        '{"RepetitionTime": 2.0, "EchoTime": 0.003}', encoding="utf-8")
    # Busy enough that a folder listing has to choose: twelve runs in
    # one folder is a quiet Tuesday for a real study.
    for i in range(12):
        (func / f"sub-01_ses-01_task-rest_run-{i:02d}_bold.json").write_text(
            "{}", encoding="utf-8")
    (func / "sub-01_ses-01_task-rest_bold.json").write_text(
        '{"RepetitionTime": 2.0, "TaskName": "rest"}', encoding="utf-8")
    sub02 = root / "sub-02/anat"
    sub02.mkdir(parents=True, exist_ok=True)
    binary = sub02 / "sub-02_T1w.nii.gz"
    binary.write_bytes(b"\x1f\x8b" + bytes(2048))
    return {
        "root": root,
        "sidecar": anat / "sub-01_ses-01_T1w.json",
        "bold": func / "sub-01_ses-01_task-rest_bold.json",
        "binary": binary,
    }


class TestDatasetTree:
    def test_the_main_parts_of_a_dataset_are_drawn(self, tmp_path):
        paths = _dataset(tmp_path / "my_mri_dataset")
        tree = dataset_tree(paths["root"])

        assert tree.startswith("my_mri_dataset/")
        for expected in (
            "README", "dataset_description.json", "participants.tsv",
            "sub-01/", "sub-02/", "ses-01/", "anat/", "func/",
            "├── ", "└── ",
        ):
            assert expected in tree
        # …and it explains itself rather than listing bare names.
        assert "# dataset metadata" in tree
        assert "# subject" in tree

    def test_the_finding_folder_is_listed_properly(self, tmp_path):
        """The folder the model is asked about is the one folder worth
        listing: four files would hide eleven of the twelve runs."""
        paths = _dataset(tmp_path / "my_mri_dataset")
        focused = dataset_tree(paths["root"], focus="sub-01/ses-01/func")

        assert "run-11_bold.json" in focused
        assert "more files" not in focused.split("func/")[1]

    def test_busy_folders_collapse_behind_a_marker(self, tmp_path):
        """Elsewhere four files stand for the folder: a listing of
        287,000 files is not a prompt, it is a hostage."""
        paths = _dataset(tmp_path / "my_mri_dataset")
        plain = dataset_tree(paths["root"])

        assert "run-04_bold.json" not in plain
        assert "more files" in plain

    def test_no_absolute_path_leaks_into_the_tree(self, tmp_path):
        """``context["path"]`` is made relative so the reviewer's home
        directory never reaches a prompt; the tree must not undo that."""
        paths = _dataset(tmp_path / "my_mri_dataset")
        tree = dataset_tree(paths["root"], focus="sub-01/ses-01/func")

        assert str(tmp_path).replace("\\", "/") not in tree
        assert str(tmp_path) not in tree

    def test_the_line_budget_holds(self, tmp_path):
        """Depth, not width: 300 folders side by side collapse behind
        ``… more folders`` long before the budget matters, so this one
        nests deep enough to draw every line it can."""
        root = tmp_path / "huge"
        root.mkdir()
        for i in range(8):
            subject = root / f"sub-{i:02d}"
            subject.mkdir()
            for j in range(8):
                session = subject / f"ses-{j:02d}"
                session.mkdir()
                for k in range(8):
                    (session / f"anat-{k}").mkdir()
        tree = dataset_tree(root)

        # The budget covers the drawn entries; the root line and the
        # truncation marker sit outside it, and say so.
        assert len(tree.splitlines()) <= TREE_MAX_LINES + 1
        assert tree.rstrip().endswith("(dataset tree truncated)")

    def test_nothing_readable_means_no_tree(self, tmp_path):
        assert dataset_tree(None) == ""
        assert dataset_tree(tmp_path / "does-not-exist") == ""
        assert dataset_tree(_dataset(tmp_path / "d")["binary"]) == ""


class TestFileDigest:
    def test_a_sidecar_becomes_its_keys(self, tmp_path):
        paths = _dataset(tmp_path / "my_mri_dataset")
        digest = file_digest(paths["sidecar"])

        assert "Compiled from sub-01_ses-01_T1w.json" in digest
        assert "RepetitionTime: 2.0" in digest
        assert "EchoTime: 0.003" in digest

    def test_a_table_becomes_columns_and_rows(self, tmp_path):
        paths = _dataset(tmp_path / "my_mri_dataset")
        digest = file_digest(paths["root"] / "participants.tsv")

        assert "TSV table: 3 columns, 8 data rows" in digest
        assert "columns: participant_id, age, sex" in digest
        assert "| sub-01 | 21 | F" in digest

    def test_a_binary_file_is_named_but_never_read(self, tmp_path):
        """A ``.nii.gz`` is megabytes of floats. The model is told the
        file exists and that it was *not* inspected — which is what
        lets it say so instead of describing an image it never saw."""
        paths = _dataset(tmp_path / "my_mri_dataset")
        digest = file_digest(paths["binary"])

        assert digest.startswith("Binary file: sub-02_T1w.nii.gz")
        assert "not inspected" in digest
        assert "\x1f\x8b" not in digest

    def test_a_text_file_is_summarised_not_shipped(self, tmp_path):
        paths = _dataset(tmp_path / "my_mri_dataset")
        long_readme = paths["root"] / "CHANGES"
        long_readme.write_text(
            "\n".join(f"change {i}" for i in range(500)), encoding="utf-8")

        digest = file_digest(long_readme)

        assert len(digest.splitlines()) <= DIGEST_MAX_LINES + 2
        assert "change 0" in digest

    def test_a_table_past_the_cap_says_so(self, tmp_path):
        """Row counts come from the part that was read — a 400-subject
        table is already past the cap — so the digest must not let them
        be quoted as a count of the whole file."""
        paths = _dataset(tmp_path / "my_mri_dataset")
        big = paths["root"] / "sessions.tsv"
        big.write_text(
            "session_id\tacq\n"
            + "".join(f"sub-01_ses-{i:02d}\tacq-{i}\n" for i in range(900)),
            encoding="utf-8",
        )

        digest = file_digest(big)

        assert "data rows" in digest
        assert "truncated" in digest
        assert "counts cover only that part" in digest

    def test_an_extensionless_file_is_still_text(self, tmp_path):
        """``README``, ``CHANGES`` and ``.bidsignore`` have no suffix to
        judge them by — and they are plain text, so calling them binary
        would withhold exactly the file a finding is often about."""
        paths = _dataset(tmp_path / "my_mri_dataset")
        digest = file_digest(paths["root"] / "README")

        assert digest.startswith("Text file: README")
        assert "A dataset." in digest

    def test_a_missing_file_sends_nothing(self, tmp_path):
        assert file_digest(tmp_path / "nope.json") == ""
        assert file_digest(None) == ""


class TestEvidenceFor:
    def test_an_explicit_root_reads_a_relative_path(self, tmp_path):
        from bidsmgr.gui import ai_explainer as mod

        paths = _dataset(tmp_path / "my_mri_dataset")
        structure, digest = mod._evidence_for(
            {"path": "sub-01/ses-01/anat/sub-01_ses-01_T1w.json"},
            paths["root"],
        )

        assert structure.startswith("my_mri_dataset/")
        assert "# dataset metadata" in structure
        assert "RepetitionTime: 2.0" in digest

    def test_an_absolute_path_establishes_its_own_root(self, tmp_path):
        """The coherence dialog and the issues list hand over a path and
        nothing else; finding the dataset above it is the only way they
        get a tree at all."""
        from bidsmgr.gui import ai_explainer as mod

        paths = _dataset(tmp_path / "my_mri_dataset")
        structure, digest = mod._evidence_for(
            {"path": str(paths["bold"])}, None,
        )

        assert structure.startswith("my_mri_dataset/")
        assert "participants.tsv" in structure
        assert "TaskName: rest" in digest
        assert str(tmp_path) not in structure

    def test_a_relative_path_with_no_root_sends_nothing(self, tmp_path):
        """Guessing at one would read the wrong file — or somebody
        else's — so an unresolvable path sends neither field."""
        from bidsmgr.gui import ai_explainer as mod

        assert mod._evidence_for({"path": "sub-01/func/x.json"}, None) == ("", "")
        assert mod._evidence_for({}, None) == ("", "")


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

    def test_a_long_transcript_is_sent_whole(self, stub_agent):
        """Nothing falls off the front.

        It used to be trimmed to the newest 12 turns so a session could
        not grow the prompt without limit — but a conversation cut off
        part way is a model that has forgotten what it was told, which
        is the worse failure of the two. The whole transcript is what
        "continue where we left off" has to mean.
        """
        client, server = stub_agent
        history = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn {i}"}
            for i in range(40)
        ]

        client.predict("now", {"rule_id": "a.b"}, history=history)

        sent = server.received["history"]                  # type: ignore[attr-defined]
        assert sent == history
        assert sent[0]["content"] == "turn 0"

    def test_junk_in_the_transcript_never_reaches_the_wire(self, stub_agent):
        """Non-dict entries are dropped rather than serialized as-is."""
        client, server = stub_agent
        client.predict("now", {"rule_id": "a.b"},
                       history=["a string", {"role": "user"}])
        assert "history" not in server.received            # type: ignore[attr-defined]

    def test_dataset_evidence_travels_as_its_own_fields(self, stub_agent):
        """The tree and the file digest are fields of their own, not
        more keys inside ``context``: the agent renders each as its own
        section, and a section it cannot find is a section it cannot
        print."""
        client, server = stub_agent
        client.predict(
            "Explain it", {"rule_id": "a.b"},
            dataset_structure="my_dataset/\n└── participants.tsv",
            file_content="TSV table: 3 columns, 12 data rows",
        )

        assert server.received["dataset_structure"].startswith("my_dataset/")  # type: ignore[attr-defined]
        assert "12 data rows" in server.received["file_content"]              # type: ignore[attr-defined]
        assert "dataset_structure" not in server.received["context"]          # type: ignore[attr-defined]
        assert "file_content" not in server.received["context"]               # type: ignore[attr-defined]

    def test_absent_evidence_sends_no_field_at_all(self, stub_agent):
        """A finding with no dataset behind it costs no prompt lines —
        and an absent key is the only honest way to say "none", since
        an empty section is a header with nothing under it."""
        client, server = stub_agent
        client.predict("Explain it", {"rule_id": "a.b"},
                       dataset_structure="   ", file_content=None)
        assert "dataset_structure" not in server.received            # type: ignore[attr-defined]
        assert "file_content" not in server.received                 # type: ignore[attr-defined]

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
