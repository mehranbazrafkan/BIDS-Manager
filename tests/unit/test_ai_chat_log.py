"""The AI conversations as files: where they go and what a line holds.

The window keeps its own copy in memory for restoring; these tests are
about the second copy — the one a user can open after the app is closed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from bidsmgr import ai_chat_log


class TestWhereTheLogsLive:
    def test_the_default_sits_beside_the_models(
        self, monkeypatch, tmp_path,
    ):
        """One per-user application folder holds both, so a user looking
        for either finds the other."""
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        got = ai_chat_log.default_dir()

        assert got.name == "chat-logs"
        if os.name == "nt":
            assert got == tmp_path / "bidsmgr" / "bidsmgr" / "chat-logs"

    def test_a_stored_folder_wins_and_blank_means_the_default(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        assert ai_chat_log.logs_dir("") == ai_chat_log.default_dir()
        assert ai_chat_log.logs_dir("   ") == ai_chat_log.default_dir()
        assert ai_chat_log.logs_dir(f"  {tmp_path / 'mine'}  ") == (
            tmp_path / "mine"
        )

    def test_a_tilde_means_the_home_folder(self):
        """A hand-typed ``~/logs`` is the home folder, not a folder
        literally called ``~``."""
        assert ai_chat_log.logs_dir("~/logs") == Path.home() / "logs"


class TestTheFile:
    def test_one_finding_keeps_one_file(self, tmp_path):
        first = ai_chat_log.conversation_file("rule|/data/a.nii.gz|TR", tmp_path)
        second = ai_chat_log.conversation_file("rule|/data/b.nii.gz|TR", tmp_path)

        assert first.suffix == ".jsonl"
        assert first.parent == tmp_path
        assert first != second
        # Asking the same finding again appends to the same file, across
        # sessions: the conversation has one history, not one per run.
        assert ai_chat_log.conversation_file(
            "rule|/data/a.nii.gz|TR", tmp_path,
        ) == first

    def test_a_path_in_the_key_cannot_escape_the_folder(self, tmp_path):
        """The key is a rule id and a file path; the file name cannot be
        allowed to contain either a separator or a leading dot-dot."""
        name = ai_chat_log.conversation_file(
            "../../etc/passwd", tmp_path,
        ).name

        assert "/" not in name
        assert "\\" not in name
        assert not name.startswith(".")


class TestWritingATurn:
    def _turns(self, path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def test_a_question_and_an_answer_are_two_lines(self, tmp_path):
        folder = tmp_path / "chat-logs"
        key = "rule|/data/sub-01/func/bold.nii.gz|RepetitionTime"
        ai_chat_log.write_turn(
            folder, key, "user", "Explain this error to me.",
            finding={"severity": "err", "message": "TR is missing"},
        )
        path = ai_chat_log.write_turn(
            folder, key, "assistant", "RepetitionTime is the TR.",
            meta={"model": "stub/model", "seconds": 1.5, "tokens": 7},
        )

        lines = self._turns(path)

        assert [line["role"] for line in lines] == ["user", "assistant"]
        assert lines[0]["text"] == "Explain this error to me."
        assert lines[0]["finding"]["severity"] == "err"
        # The metadata belongs to the answer; putting it on the question
        # would claim a question had a model and a token count.
        assert "meta" not in lines[0]
        assert lines[1]["meta"]["model"] == "stub/model"
        assert lines[1]["meta"]["tokens"] == 7
        assert lines[0]["time"] and lines[1]["time"]

    def test_a_log_that_cannot_be_written_is_not_an_error(self, tmp_path):
        """A full disk must not turn an answer that arrived into an
        exception in the chat window."""
        blocked = tmp_path / "not-a-folder"
        blocked.write_text("a file where the folder should be", encoding="utf-8")

        assert ai_chat_log.write_turn(blocked, "key", "user", "hi") is None
