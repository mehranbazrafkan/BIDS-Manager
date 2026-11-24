"""The on-disk record of the AI window's conversations.

The app keeps conversations in memory while it runs (see
``_CONVERSATIONS`` in :mod:`bidsmgr.gui.ai_explainer`), and that is still
what a reopened window restores. This module is the second copy: every
question and answer is appended to a file, so a conversation can be read
after the app is closed, searched, or sent to somebody else.

One file per conversation, named after the finding (rule, file, field)
with a short digest so two findings can never share a file. The format is
JSON Lines — one JSON object per line — because it is readable as text,
appends without rewriting the file, and survives the app being killed
between two turns. A question carries the finding it is about; an answer
carries the same metadata the window shows under it (model, seconds,
tokens, intent) and the full prompt record, which is the part worth
keeping: what the model was fed is what explains its answer.

The folder lives next to the models in the per-user application data
folder unless the user chose another one in Settings → AI Agent. The
write is best-effort: a log that cannot be written is a lost log, never a
lost answer, so nothing here raises into the chat window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import agent_service

log = logging.getLogger(__name__)

#: The folder's name inside the per-user application data folder.
DEFAULT_FOLDER_NAME = "chat-logs"

#: Characters a file name may keep. Everything else becomes a dash: the
#: conversation key is a rule id plus a path, and a path is full of
#: separators that a file name cannot hold.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_dir() -> Path:
    """Where conversations are written when the user chose nothing."""
    return agent_service.default_data_dir() / DEFAULT_FOLDER_NAME


def logs_dir(override: str | Path = "") -> Path:
    """The chat-log folder: a stored choice when there is one, else the
    per-user default. ``~`` is expanded, like the models folder."""
    text = str(override or "").strip()
    return Path(text).expanduser() if text else default_dir()


def file_stem(key: str) -> str:
    """A safe, stable file name stem for one finding.

    Stable because the same finding should keep appending to the same
    file across sessions; digest-suffixed because two different findings
    can sanitize to the same slug (paths differing only in a character
    that became a dash), and silently mixing them would be worse than an
    uglier name.

    Public because the dataset export writes its own files beside the
    conversations and names them the same way: one finding, one stem.
    """
    slug = _UNSAFE.sub("-", str(key or "")).strip("-._")
    digest = hashlib.sha1(str(key or "").encode("utf-8")).hexdigest()[:8]
    return f"{slug[:60] or 'conversation'}-{digest}"


def conversation_file(key: str, folder: str | Path = "") -> Path:
    """The JSONL file one conversation appends to."""
    return logs_dir(folder) / f"{file_stem(key)}.jsonl"


def write_turn(
    folder: str | Path,
    key: str,
    role: str,
    text: str,
    *,
    meta: Optional[dict[str, Any]] = None,
    finding: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    """Append one turn to the conversation's file, or return ``None``.

    ``finding`` is written on questions (so a line says what rule and
    file it is about without the reader having to look at the file name)
    and ``meta`` on answers. A failure to write is logged and swallowed:
    the chat window is not the place to report a full disk, and an answer
    that arrived is an answer.
    """
    path = conversation_file(key, folder)
    record: dict[str, Any] = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "key": key,
        "role": role,
        "text": str(text),
    }
    if role == "user" and finding:
        record["finding"] = dict(finding)
    if role == "assistant" and meta:
        record["meta"] = dict(meta)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str))
            handle.write("\n")
    except OSError as exc:                                # pragma: no cover
        log.warning("could not write the AI chat log %s: %s", path, exc)
        return None
    return path


__all__ = [
    "DEFAULT_FOLDER_NAME",
    "conversation_file",
    "default_dir",
    "file_stem",
    "logs_dir",
    "write_turn",
]
