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
     -> {"response": str, "intent": str, "model": str, "log": {...},
         "prompt_tokens" / "generated_tokens" / "generation_seconds",
         "model_load_seconds"}
    GET  /models    cheapest endpoint there is; never loads the model
    GET  /status    what the process is doing: loading, or generating
    GET  /config    current knobs — which model is running
    PUT  /config    apply this app's stored LLM preferences
    POST /models/download         fetch a model in the background
    GET  /models/download/status  that download's progress, in bytes

``/predict`` also returns the record of the exchange (``log``: the
system prompt, the user's question, the output) and how it was produced
(checkpoint, token counts, seconds) — the old ``prompt_log.txt`` file,
now travelling per message instead of on disk. The conversation itself
is also appended to a JSONL file per finding (:mod:`bidsmgr.ai_chat_log`)
in the folder chosen in Settings → AI Agent. When the configured model is
not downloaded yet, the worker fetches it first and reports progress,
rather than leaving a request frozen for the minutes a multi-gigabyte
download takes.

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

import csv
import hashlib
import html
import json
import logging
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..agent_service import service
from .app_settings import KEYS, AppSettings
from .dialog_chrome import (
    WrapLabel,
    build_footer,
    build_footer_stacked,
    build_header,
    card,
    scrollable_body,
)
from .theme_manager import CUR
from .typefaces import mono_families
from .widgets.agent_status import AgentStatus

log = logging.getLogger(__name__)

# The agent's own ``python main.py`` default (ai_agent/main.py line 245).
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# Generous: the first /predict loads a local LLM onto the device.
DEFAULT_TIMEOUT = 120
# Sent when the user has not typed anything. Deliberately the word
# "Explain" — ``Planner._decided_intent`` routes on the prompt, and the
# button is there to explain, not to reach for a fix tool.
DEFAULT_PROMPT = "Explain this BIDS validation finding to me."


def default_prompt(payload: dict) -> str:
    """The first question asked about a finding, in the finding's own words.

    BIDS Manager calls an error an error and a warning a warning; the
    sentence that opens the conversation should too. It is also the
    sentence the intent guard routes most reliably — "Explain this
    error" is an unmistakable explanation request, so the agent spends
    its first model call on the answer rather than on deciding what was
    asked (see ``ai_agent/intent.py``).

    Anything that is neither (a coherence note, an informational hint)
    keeps the neutral wording.
    """
    severity = str((payload or {}).get("severity") or "").strip().lower()
    if severity.startswith("err"):
        return "Explain this error to me."
    if severity.startswith("warn"):
        return "Explain this warning to me."
    return DEFAULT_PROMPT


# Workers the user closed the dialog on. Held HERE, never as a child of
# the dialog: a QThread destroyed while ``run()`` is in flight takes the
# process with it, and "close while the model is still answering" is the
# first thing anybody does. See :func:`_retire`.
_ORPHANS: set[QThread] = set()


# =====================================================================
# Conversation
# =====================================================================

# What has been asked and answered about a finding, kept for as long as
# the app runs. It lives here rather than in the dialog for two reasons:
# the dialog is destroyed on close by design, and reopening a finding is
# precisely the moment the user expects to find it again.
#
# The whole transcript is kept and the whole of it travels with the next
# question. It used to be capped (40 turns stored, 12 sent) so an
# afternoon of follow-ups could not grow the prompt without limit — but
# a conversation cut off part way is a model that has forgotten what it
# was told, which is a worse failure than a long prompt. The agent's own
# local model is the only thing that pays for the length, and it is the
# thing that needs the context.
_CONVERSATIONS: dict[str, list[dict]] = {}


def conversation_key(payload: dict) -> str:
    """The identity of the finding, not of the moment it was asked about.

    ``rule + file + field`` is what a person means by "that error
    again": the same problem in the same place is the same conversation
    however the message happened to be worded. The digest is a fallback
    for findings that carry none of those — a coherence warning with no
    path, say — where the whole payload is all there is to go on.
    """
    parts = [
        str(payload.get(k) or "").strip()
        for k in ("rule_id", "path", "field", "datatype", "suffix")
    ]
    if "|".join(parts).strip("|"):
        return "|".join(parts)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


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
# Evidence: the dataset around the finding
# =====================================================================
#
# Two more things the agent needs beside the finding itself, each sent
# as its own field of ``POST /predict``: the shape of the dataset, and
# what is in the file the finding is about. Both are *compiled* here
# rather than forwarded raw — a real dataset holds tens of thousands of
# files and a ``.nii.gz`` is megabytes of floats, and neither belongs
# in a prompt. What the model needs is the shape of the thing and the
# few lines that are actually about this issue.

#: Lines the dataset tree may occupy. Past this the tree stops; the
#: finding's own folder is drawn first, so what is cut is the part
#: nobody was asking about.
TREE_MAX_LINES = 160
#: How far the tree descends: root → subject → session → datatype.
TREE_MAX_DEPTH = 4
#: Files drawn in an ordinary directory, and in the directory the
#: finding actually sits in (which is the one worth listing).
TREE_FILE_CAP = 4
TREE_FOCUS_FILE_CAP = 14
#: Directories drawn in any one directory before the rest collapse.
TREE_DIR_CAP = 12
#: Characters of a file digest. A sidecar is small; a raw dump of a
#: participants table with 400 rows is not an explanation.
DIGEST_MAX_CHARS = 6000
DIGEST_MAX_ROWS = 6
DIGEST_MAX_LINES = 60
#: Sizes past which a file is described rather than read.
DIGEST_MAX_BYTES = 4 * 1024 * 1024

#: What each well-known BIDS entry *is*, so the tree explains itself.
#: Annotated by name; anything unlisted is shown bare, which is the
#: right answer for a file nobody has heard of.
_TREE_NOTES = {
    "README": "dataset README",
    "README.md": "dataset README",
    "dataset_description.json": "dataset metadata",
    "participants.tsv": "subject demographics table",
    "participants.json": "column metadata for participants.tsv",
    "CHANGES": "changelog",
    "code/": "analysis scripts",
    "derivatives/": "pipeline outputs",
    ".bidsignore": "patterns the validator skips",
}
_DATATYPE_NOTES = {
    "anat": "structural imaging",
    "func": "functional imaging",
    "dwi": "diffusion imaging",
    "fmap": "fieldmaps",
    "perf": "ASL perfusion",
    "pet": "PET",
    "eeg": "EEG",
    "meg": "MEG",
    "ieeg": "intracranial EEG",
    "behaviour": "behavioural data",
    "behavior": "behavioural data",
}
#: Directory names that carry no information for a model and can be
#: thousands of entries deep. Hidden directories go the same way — but
#: hidden *files* do not: ``.bidsignore`` is exactly what a validation
#: finding is about.
_TREE_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".datalad"}
_TREE_JUNK = {".DS_Store", "Thumbs.db", "desktop.ini"}
#: Extensions we can decode as text. Anything else — and anything at
#: all with no extension past the size guard — is described rather
#: than read.
_TEXTUAL_SUFFIXES = {
    ".json", ".tsv", ".csv", ".txt", ".md", ".bval", ".bvec",
    ".jsonl", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".log",
    ".py", ".m", ".r", ".sh", ".html", ".htm", ".xml", ".rst",
}
_BINARY_SUFFIXES = (
    ".nii", ".nii.gz", ".gz", ".bz2", ".xz", ".zip", ".tar", ".dcm",
    ".img", ".hdr", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif",
    ".edf", ".bdf", ".set", ".fif", ".vhdr", ".vmrk", ".eeg", ".nwb",
    ".snirf", ".parquet", ".feather", ".sqlite", ".db",
)


def _tree_note(name: str, is_dir: bool) -> str:
    """The ``# comment`` for *name*, or ``""`` when it is unlisted."""
    if is_dir:
        key = name if name.endswith("/") else f"{name}/"
        if key in _TREE_NOTES:
            return _TREE_NOTES[key]
        if name.startswith("sub-"):
            return "subject"
        if name.startswith("ses-"):
            return "session"
        return _DATATYPE_NOTES.get(name, "")
    return _TREE_NOTES.get(name, "")


def _natural(value: str) -> tuple:
    """Sort ``sub-2`` before ``sub-10``, names before numbers."""
    return (0, int(value), "") if value.isdigit() else (1, value.lower())


def dataset_tree(root: Any, *, focus: Optional[str] = None) -> str:
    """A condensed tree of the dataset at *root* — the main parts.

    Built for the prompt, not for a file listing: every top-level entry
    (that is the dataset's shape), the directory skeleton below it, and
    a handful of files per folder — with the folder the finding is in
    listed properly, because that is the folder the model is being
    asked about. Busy subtrees collapse behind ``… N more``, hidden and
    generated directories are skipped, and the whole thing stops at
    :data:`TREE_MAX_LINES`.

    *focus* is the offending file's path **relative to the root**. The
    chain to it is always drawn, so a finding in
    ``sub-01/ses-01/func/`` never loses its folder to the line budget.

    Returns ``""`` when *root* is not a readable directory — a payload
    with no dataset behind it simply sends no structure.
    """
    try:
        base = Path(root)
        if not base.is_dir():
            return ""
    except (OSError, TypeError, ValueError):
        return ""

    focus = str(focus or "").strip().strip("/").replace("\\", "/")
    lines: list[str] = [f"{base.name or base}/"]
    budget = [TREE_MAX_LINES - 1]

    def emit(text: str) -> None:
        lines.append(text)

    def draw(directory: Path, rel: str, prefix: str, depth: int) -> None:
        if budget[0] <= 0:
            return
        try:
            # Files first — a BIDS root reads README → tables → folders,
            # and so does every datatype folder.
            entries = sorted(
                directory.iterdir(),
                key=lambda p: (not p.is_file(), _natural(p.name)),
            )
        except OSError:
            return
        entries = [
            p for p in entries
            if not (p.is_dir() and (
                p.name.startswith(".") or p.name in _TREE_SKIP_DIRS
            ))
            and p.name not in _TREE_JUNK
        ]
        files = [p for p in entries if p.is_file()]
        dirs = [p for p in entries if p.is_dir()]

        # The root lists everything (that *is* the dataset's shape), as
        # does the folder the finding sits in. Everywhere else a few
        # files stand for the folder.
        cap = (TREE_FOCUS_FILE_CAP
               if depth == 0 or _is_focus(rel, focus)
               else TREE_FILE_CAP)
        shown = files[:cap]
        for i, path in enumerate(shown):
            if budget[0] <= 0:
                return
            # ``└──`` only when nothing follows it: a folder whose files
            # were cut mid-list must not close the list twice.
            last = (i == len(shown) - 1
                    and not dirs and len(files) == len(shown))
            emit(f"{prefix}{'└── ' if last else '├── '}{path.name}"
                 + _comment(_tree_note(path.name, False)))
            budget[0] -= 1
        if len(files) > len(shown):
            if budget[0] <= 0:
                return
            # ``└──`` only when nothing follows: folders may still be
            # listed under this one.
            emit(f"{prefix}{'└── ' if not dirs else '├── '}… "
                 f"{len(files) - len(shown)} more files")
            budget[0] -= 1

        extra = len(dirs) - min(len(dirs), TREE_DIR_CAP)
        for i, path in enumerate(dirs[:TREE_DIR_CAP]):
            if budget[0] <= 0:
                return
            last = i == len(dirs) - 1
            emit(f"{prefix}{'└── ' if last else '├── '}{path.name}/"
                 + _comment(_tree_note(path.name, True)))
            budget[0] -= 1
            if depth < TREE_MAX_DEPTH and budget[0] > 0:
                child_rel = f"{rel}/{path.name}".strip("/")
                # Always descend towards the finding; elsewhere descend
                # only as deep as the shape needs.
                if _is_focus(child_rel, focus) or depth + 1 < TREE_MAX_DEPTH:
                    draw(path, child_rel, prefix + ("    " if last else "│   "),
                         depth + 1)
        if extra > 0 and budget[0] > 0:
            emit(f"{prefix}└── … {extra} more folders")
            budget[0] -= 1

    draw(base, "", "", 0)
    if budget[0] <= 0:
        lines.append("… (dataset tree truncated)")
    return "\n".join(lines)


def _is_focus(rel: str, focus: str) -> bool:
    """Is *rel* the focus file, or a folder on the way to it?"""
    if not focus:
        return False
    rel = rel.strip("/")
    if not rel:
        return True
    return focus == rel or focus.startswith(f"{rel}/")


def _comment(note: str) -> str:
    return f"    # {note}" if note else ""


def _short(value: Any, limit: int = 120) -> str:
    """One value on one line, however long it really is."""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return text if len(text) <= limit else f"{text[:limit]}…"


def _describe_json(value: Any, depth: int = 0) -> list[str]:
    """A JSON document as ``key: type = value`` lines, not as bytes.

    A sidecar's *keys* are what an explanation leans on — ``RepetitionTime``,
    ``TaskName`` — while a megabyte of nested schema is not. Depth and
    breadth are both capped; anything past the cap is counted instead.
    """
    if isinstance(value, dict):
        if depth > 3:
            return [f"… ({len(value)} keys)"]
        out: list[str] = []
        items = list(value.items())
        for key, item in items[:20]:
            out.append(f"{key}: {_scalar(item, depth)}")
        if len(items) > 20:
            out.append(f"… {len(items) - 20} more keys")
        return out
    if isinstance(value, list):
        if not value:
            return ["[] (empty list)"]
        out = [f"list of {len(value)} items, first:"]
        for item in value[:3]:
            out.append(f"  - {_scalar(item, depth)}")
        return out
    return [_short(value)]


def _scalar(value: Any, depth: int) -> str:
    """One JSON value, folded onto a line — nested structures recurse."""
    if isinstance(value, dict):
        if depth > 2:
            return f"{{… {len(value)} keys}}"
        items = [f"{k}: {_scalar(v, depth + 1)}"
                 for k, v in list(value.items())[:4]]
        if len(value) > 4:
            items.append(f"… {len(value) - 4} more keys")
        return "{" + "; ".join(items) + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [_scalar(v, depth + 1) for v in value[:3]]
        if len(value) > 3:
            items.append(f"… {len(value) - 3} more")
        return "[" + ", ".join(items) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return repr(round(value, 6))
    return _short(value)


def _digest_json(text: str) -> Optional[str]:
    try:
        doc = json.loads(text)
    except (ValueError, TypeError):
        return None
    body = "\n".join(f"  {line}" for line in _describe_json(doc))
    return f"JSON document, {len(text):,} characters:\n{body}"


def _digest_table(text: str, delimiter: str) -> Optional[str]:
    """A table as columns + row count + the first few rows.

    The header is the part an explanation cites ("``RepetitionTime`` is
    missing from participants.json" needs the column names, not row
    387), and a handful of rows shows what the values look like without
    shipping the whole table.
    """
    try:
        rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    except csv.Error:
        return None
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return None
    header, data = rows[0], rows[1:]
    label = "TSV" if delimiter == "\t" else "CSV"
    out = [f"{label} table: {len(header)} columns, {len(data):,} data rows",
           f"  columns: {', '.join(_short(c, 40) for c in header)}"]
    for row in data[:DIGEST_MAX_ROWS]:
        cells = [_short(c, 40) for c in row]
        if len(cells) > 12:
            cells = cells[:12] + [f"… {len(row) - 12} more"]
        out.append("  | " + " | ".join(cells))
    if len(data) > DIGEST_MAX_ROWS:
        out.append(f"  … {len(data) - DIGEST_MAX_ROWS:,} more rows")
    return "\n".join(out)


def _digest_numbers(text: str, kind: str) -> Optional[str]:
    values = text.split()
    nums: list[float] = []
    for token in values:
        try:
            nums.append(float(token))
        except ValueError:
            return None
    if not nums:
        return None
    return (
        f"{kind}: {len(nums)} values, "
        f"range {min(nums):g} … {max(nums):g}, "
        f"first: {', '.join(f'{v:g}' for v in nums[:8])}"
        + (" …" if len(nums) > 8 else "")
    )


def file_digest(path: Any) -> str:
    """A compiled summary of the file the finding is about.

    Compiled, not raw: a JSON sidecar becomes its keys and values, a
    table its columns and first rows, a bval/bvec its shape. Binary
    payloads — a ``.nii.gz`` is the common case — are named and sized
    but never read, so the model can say "the volume itself was not
    inspected" instead of inventing what is in it.

    Returns ``""`` when the file cannot be read, which sends nothing.
    """
    try:
        target = Path(path)
        size = target.stat().st_size
    except (OSError, TypeError, ValueError):
        return ""

    name = target.name.lower()
    if name.endswith(_BINARY_SUFFIXES) or not _looks_textual(name, size):
        return (f"Binary file: {target.name} "
                f"({_human(size)}). Contents not sent — treat the file "
                f"as present but not inspected.")

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(DIGEST_MAX_CHARS + 1)
    except OSError:
        return ""
    truncated = len(text) > DIGEST_MAX_CHARS
    text = text[:DIGEST_MAX_CHARS]

    body: Optional[str] = None
    if name.endswith(".json"):
        body = _digest_json(text)
    elif name.endswith(".tsv"):
        body = _digest_table(text, "\t")
    elif name.endswith(".csv"):
        body = _digest_table(text, ",")
    elif name.endswith(".bval"):
        body = _digest_numbers(text, "b-values")
    elif name.endswith(".bvec"):
        body = _digest_numbers(text, "b-vectors")
    if body is None:
        rows = text.splitlines()[:DIGEST_MAX_LINES]
        body = "\n".join(rows)
        note = (f"Text file: {target.name} ({_human(size)}).")
    else:
        note = f"Compiled from {target.name} ({_human(size)}):"
        body = "\n".join(f"  {line}" for line in body.splitlines())
    if truncated:
        # Counts above (rows, keys) come from the part that was read —
        # a 400-subject participants.tsv is already past the cap — so
        # say so rather than let them be quoted as the whole file.
        body += ("\n… (truncated: the file is larger than the part "
                 "read, so counts cover only that part)")
    return f"{note}\n{body}"


def _looks_textual(name: str, size: int = 0) -> bool:
    """Is this a file we can expect to decode as text at all?

    A known extension says yes; an extension we do not recognise says
    no, because guessing wrong means shipping megabytes of replacement
    characters into a prompt. A file with *no* extension is the
    interesting case: BIDS' ``README``, ``CHANGES``, ``LICENSE`` and
    ``.bidsignore`` are all plain text and are exactly the files a
    finding is often about, so they are read — up to a size that keeps
    a nameless blob out.
    """
    suffix = Path(name).suffix.lower()
    if not suffix:
        return size <= DIGEST_MAX_BYTES
    return suffix in _TEXTUAL_SUFFIXES


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size} B"


def _evidence_for(payload: dict, root: Any = None) -> tuple[str, str]:
    """The dataset tree and the file digest for *payload*.

    The root comes from whoever opened the window when they know one —
    the coherence dialog and the validation pane both do — or from an
    absolute ``context["path"]`` when they do not, via
    :func:`bidsmgr.viz.bids.dataset_root`. A *relative* path with no
    root beside it cannot be resolved to a file on disk, and guessing
    at one would read the wrong file, so that case sends neither.

    Nothing absolute leaves this function: the tree is drawn from the
    root down, so the prompt learns the dataset's *shape* and never the
    reviewer's home directory — the reason ``context["path"]`` was made
    relative in the first place.
    """
    raw = str(payload.get("path") or "").strip().replace("\\", "/")
    base: Optional[Path] = Path(root) if root else None
    target: Optional[Path] = None

    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            target = candidate
            if base is None:
                # Deferred: this is the only caller, and a dialog with
                # an explicit root never pays for the import.
                from ..viz.bids import dataset_root

                try:
                    base = dataset_root(candidate)
                except (OSError, ValueError):
                    base = None
        elif base is not None:
            target = base / candidate

    if base is None:
        return "", ""

    focus = ""
    if target is not None:
        try:
            focus = target.resolve().relative_to(base.resolve()).as_posix()
        except (OSError, ValueError):
            # Not under the root we were handed: draw the dataset we were
            # given without pointing at a folder it does not contain.
            focus = ""
    return dataset_tree(base, focus=focus), file_digest(target)


# =====================================================================
# Transport
# =====================================================================

class AgentError(RuntimeError):
    """Transport or protocol failure talking to the agent service.

    Carries a message written for a *user*, not a log: every one of
    these surfaces in the dialog, and the common case (the service is
    not running) has to say how to start it. ``status`` is the HTTP
    code when the failure is one, so a caller can tell a missing
    endpoint (an older agent) from a refused connection.
    """

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


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
            raise AgentError(_http_error(url, exc), status=exc.code) from exc
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

    def activity(self) -> dict:
        """``GET /status`` — what the agent is doing, without waiting.

        Polled from beside a blocking ``/predict``: "reading the
        checkpoint" and "writing the answer" are otherwise the same
        silence. Short timeout on purpose — a poll that cannot be
        answered says nothing, and the request it is watching is still
        on its way.
        """
        return self._request("GET", "/status", timeout=2.0)

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

    def download_model(self, name: str) -> dict:
        """``POST /models/download`` — fetch *name* in the background.

        The agent answers immediately with the job's status; the fetch
        itself runs in the agent's process (the only one with the model
        libraries), and progress is polled with :meth:`download_status`.
        """
        try:
            return self._request("POST", "/models/download", {"name": str(name)})
        except AgentError as exc:
            if getattr(exc, "status", None) == 404:
                raise AgentError(
                    "The AI agent does not support model downloads — it "
                    "is an older version. Update the AI Agent module in "
                    "Settings and try again.",
                    status=404,
                ) from exc
            raise

    def download_status(self) -> dict:
        """``GET /models/download/status`` — how far the fetch has got.

        ``total_bytes`` is ``None`` when the size could not be
        established; the caller then shows an indeterminate bar with the
        bytes counted so far.
        """
        try:
            return self._request(
                "GET", "/models/download/status", timeout=8.0,
            )
        except AgentError as exc:
            if getattr(exc, "status", None) == 404:
                raise AgentError(
                    "The AI agent does not support model downloads — it "
                    "is an older version. Update the AI Agent module in "
                    "Settings and try again.",
                    status=404,
                ) from exc
            raise

    def predict(
        self,
        user_input: str,
        context: dict,
        history: Optional[list] = None,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> dict:
        """Run the agent. Returns ``{response, intent, model, log, ...}``.

        ``history`` is the whole transcript of this conversation so far,
        so a follow-up question — or a re-ask — has everything that came
        before it to stand on. Omitted when there is nothing yet: the
        first ask of a finding.

        ``dataset_structure`` and ``file_content`` are the two pieces
        of *evidence* about the dataset: a condensed tree of it, and a
        compiled digest of the file the finding is on. Both are
        pre-compiled by :func:`dataset_tree` / :func:`file_digest` and
        both are omitted when there is nothing to say — a finding with
        no path behind it simply carries no file.
        """
        body: dict[str, Any] = {
            "user_input": str(user_input),
            "context": dict(context),
        }
        # Rebuilt rather than passed as-is: a turn may carry local
        # metadata (the per-message footer), which is this window's
        # business and not the model's.
        turns = [
            {
                "role": str(t.get("role") or ""),
                "content": str(t.get("content") or ""),
            }
            for t in (history or [])
            if isinstance(t, dict)
            and str(t.get("role") or "").strip().lower() in ("user", "assistant")
            and str(t.get("content") or "").strip()
        ]
        if turns:
            body["history"] = turns
        for key, value in (
            ("dataset_structure", dataset_structure),
            ("file_content", file_content),
        ):
            text = str(value or "").strip()
            if text:
                body[key] = text
        out = self._request("POST", "/predict", body)
        out.setdefault("response", "")
        out.setdefault("intent", "")
        if not str(out["response"]).strip():
            raise AgentError(
                "The AI agent answered, but its response was empty."
            )
        return out

    def build_prompt(
        self,
        user_input: str,
        context: dict,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> dict:
        """``POST /prompt`` — the request ``/predict`` would send, built.

        Retrieval and prompt rendering run in the agent; no checkpoint is
        loaded and no answer is generated. Used by the Settings dataset
        export, which writes one record per finding.

        The optional evidence is the same two pieces :meth:`predict`
        sends: a condensed dataset tree and a compiled file digest. Both
        are omitted when there is nothing to say.
        """
        body: dict[str, Any] = {
            "user_input": str(user_input),
            "context": dict(context),
        }
        for key, value in (
            ("dataset_structure", dataset_structure),
            ("file_content", file_content),
        ):
            text = str(value or "").strip()
            if text:
                body[key] = text
        try:
            out = self._request("POST", "/prompt", body)
        except AgentError as exc:
            if getattr(exc, "status", None) == 404:
                raise AgentError(
                    "The AI agent does not support full-data export — it "
                    "is an older version. Update the AI Agent module in "
                    "Settings and try again.",
                    status=404,
                ) from exc
            raise
        out.setdefault("system_prompt", "")
        out.setdefault("user_prompt", str(user_input))
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
    # The likeliest cause on a first run is a module that was never
    # installed, which no amount of pressing Start will fix.
    from .. import agent_installer

    if not agent_installer.installed():
        return (
            f"Could not reach the BIDS AI agent at {url}.\n\n"
            "The AI Agent module is not installed on this machine. Open "
            "Settings -> AI Agent and choose Install — the app clones "
            "the agent's repository and sets up its dependencies for "
            f"you.\n\n({reason})"
        )
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

class _PhaseWatcher:
    """Ask the agent what it is doing while ``/predict`` is in flight.

    The request occupies its worker thread until the answer exists, so
    the one thing that thread cannot do is ask. This side thread polls
    ``GET /status`` beside it: first to see the checkpoint being loaded,
    then to catch the moment generation begins — which is what the
    dialog's clock is waiting for.

    ``answer_delay`` is how long the request waited before the answer
    started being written: the seconds the dialog must not count, and
    the ones the per-message time leaves out.
    """

    #: How often the phase is asked for. Short enough that an already
    #: loaded model feels immediate, long enough to cost nothing.
    INTERVAL = 0.25

    def __init__(
        self, client: AgentClient, emit: Callable[[str, float], None],
    ) -> None:
        self._client = client
        self._emit = emit
        self._stop = threading.Event()
        self._started = 0.0
        self.answer_delay: Optional[float] = None
        self._thread = threading.Thread(
            target=self._run, name="ai-agent-phase", daemon=True,
        )

    def start(self, started: float) -> None:
        """Begin polling; *started* is the request's ``perf_counter``."""
        self._started = started
        self._thread.start()

    def stop(self) -> None:
        """Stop at the next tick, and wait for the thread to be gone."""
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        last = ""
        while not self._stop.is_set():
            try:
                state = self._client.activity()
            except AgentError as exc:
                if getattr(exc, "status", None) == 404:
                    # An agent from before ``/status``: its silence is
                    # not a reason to hide the clock, so the request
                    # itself becomes the anchor.
                    self._emit("unknown", 0.0)
                    return
                # Anything else is a poll that could not be answered —
                # the request is still on its way; ask again.
            except Exception:                          # noqa: BLE001
                # A poll is a courtesy; a failed one changes nothing.
                log.debug("agent status poll failed", exc_info=True)
            else:
                phase = str(state.get("phase") or "")
                if phase == "generating":
                    elapsed = state.get("generating_for")
                    elapsed = (
                        float(elapsed)
                        if isinstance(elapsed, (int, float)) else 0.0
                    )
                    # The agent reports a duration, the request start is
                    # ours: anchoring here keeps both clocks in this
                    # process and leaves the load out of the count.
                    self.answer_delay = max(
                        0.0, time.perf_counter() - elapsed - self._started,
                    )
                    self._emit("generating", elapsed)
                    return
                if phase == "loading" and last != "loading":
                    last = "loading"
                    self._emit("loading", 0.0)
            if self._stop.wait(self.INTERVAL):
                return


class _PredictWorker(QThread):
    """One ``/predict`` round-trip off the GUI thread.

    Signals rather than a return value, because the whole point is that
    ``run()`` outlives the click. ``done`` / ``failed`` land on the
    dialog via a queued connection (it lives in the main thread) and
    simply vanish if the dialog has been closed — which is why the
    worker is never its child: see :func:`_retire`.

    ``model_missing`` / ``download_progress`` carry the download that
    runs first when the configured model is not local: the dialog
    switches to a progress bar and no request is in flight, so no
    timeout can fire while gigabytes arrive.
    """

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    model_missing = pyqtSignal(str, dict)
    download_progress = pyqtSignal(dict)
    #: The agent's own account of the request: ``loading`` while the
    #: checkpoint is read, ``generating`` once the answer is being written.
    phase = pyqtSignal(str, float)

    def __init__(
        self,
        user_input: str,
        context: dict,
        history: Optional[list] = None,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._user_input = user_input
        self._context = dict(context)
        self._history = list(history or [])
        self._dataset_structure = str(dataset_structure or "")
        self._file_content = str(file_content or "")

    def run(self) -> None:                               # noqa: D102
        try:
            # The first click of a session can land while torch is still
            # importing on the agent's side. Without this, a refused
            # connection during that window would be reported to the user
            # as "the agent is not running" when it is very much on its way.
            service().wait_running(timeout=60.0)
            # The model is fetched *before* the question is sent when it
            # is not local: the download runs once, visibly, and the
            # request that follows is an ordinary one.
            self._ensure_model()
            result = self._predict()
        except AgentError as exc:
            log.info("AI agent request failed: %s", exc)
            self.failed.emit(str(exc))
        except Exception:                                # noqa: BLE001
            log.exception("AI agent request crashed")
            self.failed.emit(traceback.format_exc())
        else:
            self.done.emit(result)

    def _predict(self) -> dict:
        """The question itself, with the wall-clock time of the answer."""
        client = AgentClient()
        started = time.perf_counter()
        # Asked beside the request, which cannot ask for itself: the
        # dialog's clock must not count the checkpoint being loaded.
        watcher = _PhaseWatcher(client, self.phase.emit)
        watcher.start(started)
        try:
            result = client.predict(
                self._user_input, self._context, history=self._history,
                dataset_structure=self._dataset_structure,
                file_content=self._file_content,
            )
        finally:
            watcher.stop()
        # The response time starts when the answer starts being written,
        # not when the request left: that is what the dialog's clock
        # shows, and a cold first answer should not read as a slow model.
        result["response_seconds"] = max(
            0.0,
            time.perf_counter() - started - (watcher.answer_delay or 0.0),
        )
        return result

    def _ensure_model(self) -> None:
        """Fetch the configured model first when it is not local.

        The agent's list says what is downloaded; when the configured
        model is not among them, it is downloaded here — visibly, with
        progress — rather than silently inside the first request. When
        the model is already local, or the agent's answer cannot say,
        nothing happens and /predict behaves as before.
        """
        client = AgentClient()
        model, downloaded = self._configured_model(client)
        if model is None or downloaded:
            return
        self.model_missing.emit(model, {})
        client.download_model(model)
        while True:
            status = client.download_status()
            self.download_progress.emit(status)
            state = str(status.get("status") or "")
            if state == "done":
                return
            if state == "error":
                raise AgentError(
                    f"Downloading {model} failed: "
                    f"{status.get('error') or 'unknown error'}"
                )
            if state == "idle" or (
                status.get("name") and str(status["name"]) != model
            ):
                # The job slot is empty or holds somebody else's fetch:
                # the agent restarted (the job lives in its process), so
                # waiting on it would spin forever with a bar at zero.
                raise AgentError(
                    f"Downloading {model} stopped — the AI agent "
                    "restarted or another download took over. Ask "
                    "again: the bytes already fetched are kept."
                )
            # No timeout on purpose: a download the user can see is not
            # a hang, and the loop ends when the agent's job does.
            time.sleep(0.8)

    def _configured_model(self, client) -> tuple[Optional[str], bool]:
        """The model the agent runs and whether it is local, or
        ``(None, True)`` when its answer cannot say — /predict then
        behaves as before rather than refusing to run on a guess."""
        model = ""
        try:
            cfg = client.config()
            llm = cfg.get("llm") if isinstance(cfg, dict) else None
            if isinstance(llm, dict):
                model = str(llm.get("model_name") or "").strip()
        except AgentError:
            return None, True
        if not model:
            return None, True
        try:
            info = client.ping()
        except AgentError:
            return None, True
        models = info.get("models") if isinstance(info, dict) else None
        entries = [e for e in (models or []) if isinstance(e, dict)]
        for entry in entries:
            if str(entry.get("name") or "") == model:
                flag = entry.get("downloaded")
                if isinstance(flag, bool):
                    return model, flag
                # An older agent: no flags, so the state is unknown.
                return None, True
        # Not in the agent's list. A modern agent flags what it has, so
        # its silence about this model means "not here" and the fetch
        # may be started; an older one knows nothing either way.
        if any(isinstance(e.get("downloaded"), bool) for e in entries):
            return model, False
        return None, True


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
    history: Optional[list] = None,
    dataset_structure: Optional[str] = None,
    file_content: Optional[str] = None,
    on_done,
    on_failed,
    on_model_missing=None,
    on_download_progress=None,
    on_phase=None,
) -> _PredictWorker:
    """Start a request, hold the thread until it is genuinely done.

    Both result slots are connected *before* ``start()``. Getting that
    backwards is not a theoretical race: a refused connection fails
    inside a few milliseconds, and a slot connected a moment too late
    never fires — which leaves the footer promising an answer forever.
    The download slots are wired the same way: a missing model reports
    itself before the request, and a slot connected a moment too late
    would leave the status line saying the fetch is on with no bar and
    no sizes. The phase slot matters for the same reason: it is how the
    clock learns that generation has begun.
    """
    worker = _PredictWorker(
        user_input, context, history=history,
        dataset_structure=dataset_structure, file_content=file_content,
    )
    worker.done.connect(on_done)
    worker.failed.connect(on_failed)
    if on_model_missing is not None:
        worker.model_missing.connect(on_model_missing)
    if on_download_progress is not None:
        worker.download_progress.connect(on_download_progress)
    if on_phase is not None:
        worker.phase.connect(on_phase)
    # Parentless by construction (``parent=None`` above): never a child
    # of the dialog, whose lifetime the user controls.
    _ORPHANS.add(worker)
    worker.finished.connect(lambda w=worker: _retire(w))
    worker.start()
    return worker


# =====================================================================
# Transcript bubbles
# =====================================================================
class BubbleColors(NamedTuple):
    """The four colours a single turn of the chat is drawn with."""

    name: str    # the sender label
    fill: str    # the bubble behind the words
    border: str  # its outline
    text: str    # the words themselves


def _mix(fg: str, bg: str, alpha: float) -> str:
    """Blend *fg* over *bg* at *alpha*, returned as an opaque hex.

    Qt's rich-text engine is unreliable about ``rgba()`` in a ``style``
    attribute, so a translucent bubble has to be flattened here — over
    the surface the widget actually paints it on, which is what makes a
    light window and a dark one each look intentional.
    """
    a, b = QColor(fg), QColor(bg)
    return "#{:02x}{:02x}{:02x}".format(
        round(a.red() * alpha + b.red() * (1 - alpha)),
        round(a.green() * alpha + b.green() * (1 - alpha)),
        round(a.blue() * alpha + b.blue() * (1 - alpha)),
    )


class TranscriptView(QScrollArea):
    """The conversation, one widget per turn.

    This used to be a ``QTextBrowser`` of HTML bubbles, and a browser can
    draw the same shapes — but rich text has no buttons, only links, and
    the record behind an answer has to be a button. So each turn is a
    widget: a bubble on its own side, the footer under an answer, and,
    when there is a record to open, a real ``QPushButton`` under it.

    ``toPlainText`` stays, because "Copy conversation" is a text
    operation and the turns are the text — no window chrome in it.
    """

    #: A bubble never takes more than this share of the width, so a long
    #: answer still reads as a message with a margin beside it rather
    #: than as the page itself.
    WIDTH_SHARE = 0.78

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        body = QWidget()
        body.setObjectName("ai-transcript-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._lay = QVBoxLayout(body)
        self._lay.setContentsMargins(8, 8, 8, 8)
        self._lay.setSpacing(7)
        self._lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(body)
        self._bubbles: list[tuple[QWidget, int]] = []
        self._plain: list[str] = []

    # -- reading -----------------------------------------------------

    def toPlainText(self) -> str:
        """Every turn as words, in order — what "Copy" puts on the clipboard."""
        return "\n\n".join(self._plain)

    # -- drawing -----------------------------------------------------

    def clear(self) -> None:
        """Drop every turn; the caller redraws from the history it holds."""
        while self._lay.count():
            item = self._lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Hide before deleting: a widget with its parent cleared
                # becomes a top-level window for one event loop turn, and
                # a redraw must not flash little windows. ``deleteLater``
                # tears it down once the loop is free.
                widget.hide()
                widget.deleteLater()
        self._bubbles.clear()
        self._plain.clear()

    def show_placeholder(self, text: str) -> None:
        """The empty state, centred where the conversation will appear."""
        self.clear()
        label = QLabel(text)
        label.setObjectName("ai-answer-empty")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lay.addWidget(label)

    def add_turn(
        self,
        side: str,
        who: str,
        text: str,
        colors: BubbleColors,
        *,
        meta: str = "",
        log: Optional[Callable[[], None]] = None,
        log_tip: str = "",
    ) -> None:
        """Append one bubble.

        *side* is ``"left"`` for the agent or ``"right"`` for the user;
        *meta* is the footer under the words (how the answer was made);
        *log*, when given, is what the "Show full data" button runs.
        """
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(0)

        bubble = QFrame()
        bubble.setObjectName("ai-bubble")
        bubble.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bubble.setStyleSheet(
            "QFrame#ai-bubble {"
            f" background-color: {colors.fill};"
            f" border: 1px solid {colors.border};"
            " border-radius: 4px; }"
        )
        # Maximum, not Preferred: a short "Thanks" must stay a short
        # bubble. The width cap applied in ``_apply_widths`` is what
        # stops a long answer from taking the window.
        bubble.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        col = QVBoxLayout(bubble)
        col.setContentsMargins(9, 6, 9, 6)
        col.setSpacing(4)

        body = QLabel()
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        body.setText(
            f'<span style="color:{colors.name}; font-weight:bold;">'
            f"{html.escape(who)}</span><br>"
            f'<span style="color:{colors.text};">{_words_html(text)}</span>'
        )
        col.addWidget(body)
        # The width the words would take without wrapping. Qt's answer to
        # "how wide is a wrapped label?" is a squarish guess that turns a
        # three-sentence answer into ten short lines; this is what stops
        # the bubble collapsing to it (see ``_apply_widths``).
        wanted = max(
            (body.fontMetrics().horizontalAdvance(line)
             for line in str(text or "").splitlines()),
            default=0,
        ) + 26          # the bubble's padding and border

        footer = ""
        if str(meta or "").strip():
            footer = str(meta)
            label = QLabel(footer)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            label.setStyleSheet(
                f"color: {colors.name}; font-size: 10px;"
            )
            col.addWidget(label)

        if log is not None:
            button = QPushButton("Show full data")
            button.setObjectName("ai-log")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(
                log_tip or (
                    "The system prompt, your question and the agent's "
                    "output for this answer."
                )
            )
            button.clicked.connect(log)
            col.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)

        if side == "right":
            row_lay.addStretch(1)
            row_lay.addWidget(bubble, 0)
        else:
            row_lay.addWidget(bubble, 0)
            row_lay.addStretch(1)
        self._lay.addWidget(row)
        self._bubbles.append((bubble, wanted))
        self._apply_widths()

        plain = f"{who}\n{text}"
        if footer:
            plain += f"\n{html.unescape(footer)}"
        self._plain.append(plain)

    def scroll_to_bottom(self) -> None:
        """Show the newest turn; the reader is at the end of a conversation."""
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _apply_widths(self) -> None:
        """Size the bubbles to their words, never past their share.

        A bubble shrinks to what it says, which is what makes two short
        messages read as a chat; and it stops at ``WIDTH_SHARE`` so a
        long answer keeps a margin beside it instead of becoming the
        page.
        """
        cap = max(240, int(self.viewport().width() * self.WIDTH_SHARE))
        for bubble, wanted in self._bubbles:
            bubble.setMaximumWidth(cap)
            bubble.setMinimumWidth(min(wanted, cap))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_widths()


def _words_html(text: str) -> str:
    """Plain turns rendered as rich text: escaped, newlines kept."""
    body = html.escape(str(text or "")).replace("\r\n", "\n")
    return body.replace("\n", "<br>")


def _has_log(turn: dict) -> bool:
    """True when this turn carries a prompt record worth opening."""
    meta = turn.get("meta")
    return (
        isinstance(meta, dict)
        and isinstance(meta.get("log"), dict)
        and bool(meta.get("log"))
    )



# =====================================================================
# Dialog
# =====================================================================

class AiExplainDialog(QDialog):
    """One finding, and the whole conversation about it.

    Modeless — findings come in groups and closing the window to read
    the next one would be a chore. The reported side never changes; the
    chat below accumulates every question and every answer, so a
    follow-up reads like a conversation instead of replacing what was
    said before.

    Reopening a finding restores that conversation rather than
    regenerating it: :func:`ask` only fires the model when there is
    nothing to show, which is what makes "I read this error earlier"
    cost nothing the second time.
    """

    def __init__(
        self, payload: dict, parent=None, root: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS-Manager - AI Explanation")
        self.resize(660, 640)
        self.setMinimumWidth(440)

        self._payload: dict[str, Any] = dict(payload)
        # Where the dataset lives, for the two evidence fields sent with
        # every question. Used only here — to walk the tree and read the
        # file — and never itself sent, so the tree can describe the
        # dataset without naming the folder it sits in.
        self._root = Path(root) if root else None
        # Built on the first ask and remembered: a directory walk and a
        # file read are not free, and a restored conversation never
        # asks again at all.
        self._evidence: Optional[tuple[str, str]] = None
        # A flag, not the thread. Holding the QThread here is what makes
        # "user closes the dialog while the model is answering" unsafe.
        self._busy = False
        self._started = 0.0
        # Where the question being re-asked already sits in the
        # transcript, or ``None`` when this request is a fresh one. Set
        # by :meth:`start`, consumed by :meth:`_on_done`: a retry
        # replaces an answer, it does not stack a second copy of the
        # question under the first.
        self._retry_from: Optional[int] = None

        # The transcript, shared with the module-level store: the dialog
        # mutates it and a later dialog for the same finding reads it
        # back. Aliasing rather than copying is the point — there is one
        # conversation, not one per window.
        self._key = conversation_key(self._payload)
        self._history: list[dict] = _CONVERSATIONS.setdefault(self._key, [])
        self._restored = bool(self._history)
        # The sentence this finding is asked about. An error is asked
        # about as an error and a warning as a warning: the dialog names
        # the severity in its subtitle, and the first question should
        # read as if the user wrote it.
        self._default_prompt = default_prompt(self._payload)
        self._last_prompt = (
            self._previous_prompt() if self._restored
            else self._default_prompt
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(build_header("AI Explanation", self._subtitle()))

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
            "Copy conversation", QDialogButtonBox.ButtonRole.ActionRole,
        )
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        # What the request is doing, in two lines: the phase, and why it
        # takes that long. Stacked above the buttons so the sentence has
        # the width of the dialog to wrap in, not the gap beside three
        # buttons.
        self._status = AgentStatus()
        outer.addWidget(build_footer_stacked(self._status, buttons))

        buttons.rejected.connect(self.reject)
        # A retry, not a second question: ``start`` holds the question
        # and the answer it replaces back from the request.
        self._retry_btn.clicked.connect(
            lambda: self.start(self._last_prompt, retry=True),
        )
        self._copy_btn.clicked.connect(self._copy)

        # The clock counts the seconds out loud. The glyph says that
        # something is happening; the number says for how long, which is
        # the question somebody staring at a first answer on a CPU
        # actually asks. One tick a second is all a clock needs.
        self._busy_since = 0.0
        self._ticking = False
        self._ticker = QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._tick_status)

        self._set_busy(False)
        self._render()
        if self._restored:
            # Said after ``_set_busy(False)``, which clears the line: this
            # window opens on a conversation from before, and "nothing
            # was re-generated" is the whole point of it.
            self._status.show_state("restored")

    def _previous_prompt(self) -> str:
        """The question to put back in front of "Ask again" on restore."""
        for turn in reversed(self._history):
            if turn.get("role") == "user":
                return str(turn.get("content") or "") or self._default_prompt
        return self._default_prompt

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
        frame, lay = card("Conversation with the AI Agent")

        # The progress of a model that had to be fetched first: hidden
        # until the worker reports a missing model, then bytes out of
        # bytes, polled from the agent while no request is in flight.
        self._download_bar = QProgressBar()
        self._download_bar.setVisible(False)
        self._download_bar.setRange(0, 100)
        self._download_note = WrapLabel("")
        self._download_note.setStyleSheet("color: #8b949e;")
        self._download_note.setVisible(False)
        lay.addWidget(self._download_bar)
        lay.addWidget(self._download_note)

        self._error = WrapLabel("")
        self._error.setObjectName("ai-error")
        self._error.setTextFormat(Qt.TextFormat.PlainText)
        self._error.setVisible(False)
        lay.addWidget(self._error)

        # A transcript rather than a single answer box: every turn is
        # rendered together from ``_history``, so the window can be
        # rebuilt from scratch on restore without re-asking anything.
        # Widgets rather than HTML in a browser, because the record
        # behind an answer opens from a button, and rich text only knows
        # how to draw a link.
        self._answer = TranscriptView()
        self._answer.setObjectName("ai-answer")
        self._answer.setMinimumHeight(150)
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
            "Ask another question about this finding  (Enter to send)"
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

    def start(self, prompt: Optional[str] = None, *, retry: bool = False) -> None:
        """Ask *prompt* about the finding shown in this dialog.

        ``retry`` re-asks the question that is already in the transcript
        — what the "Ask again" button does. It must not ask it twice:
        the question and the answer it replaces are held back from the
        request, so the model sees the conversation that came *before*
        them and simply answers again. Nothing on screen changes until
        that answer arrives, so a retry that fails leaves the previous
        answer where it was.

        With no *prompt*, the question is the one this finding is asked
        about: "Explain this error to me." or "... warning ...", in the
        words of its severity.
        """
        if self._busy:
            return
        prompt = str(prompt or "").strip() or self._default_prompt
        self._last_prompt = prompt

        at = self._repeat_of(prompt) if retry else None
        if at is None:
            # Captured before the new question joins it: sending the same
            # words twice would read to the model as if they were asked
            # twice.
            known = [
                dict(t) for t in self._history
                if t.get("role") in ("user", "assistant")
            ]
            self._retry_from = None
            # On screen the moment it is sent rather than when it is
            # answered, which is what makes a long first answer look like
            # a conversation instead of a freeze.
            self._append("user", prompt)
        else:
            # Everything that came before the question being re-asked —
            # neither it nor the answer standing in for it travels, or
            # the model is handed the same question twice with an answer
            # already attached, and answers as if starting over.
            known = [
                dict(t) for t in self._history[:at]
                if t.get("role") in ("user", "assistant")
            ]
            self._retry_from = at
        self._set_busy(True)
        self._started = time.time()
        structure, digest = self._evidence_fields()
        # The thread is owned by ``_ORPHANS``, not by this dialog; only
        # its signals are wired here, so closing the window mid-answer
        # just leaves the request to finish into the void.
        _spawn(
            prompt,
            self._payload,
            history=known,
            dataset_structure=structure,
            file_content=digest,
            on_done=self._on_done,
            on_failed=self._on_failed,
            on_model_missing=self._on_model_missing,
            on_download_progress=self._on_download_progress,
            on_phase=self._on_phase,
        )

    def _evidence_fields(self) -> tuple[str, str]:
        """The dataset tree and the file digest that travel with a question.

        Both are evidence about the *dataset*, not about this question,
        so they are computed once and reused by every follow-up: the
        folder does not move between one question and the next.
        """
        if self._evidence is None:
            self._evidence = _evidence_for(self._payload, self._root)
        return self._evidence

    def _repeat_of(self, prompt: str) -> Optional[int]:
        """Where *prompt* already sits in the transcript, or ``None``.

        The **last** matching question, because that is the one "Ask
        again" is re-asking: a question asked three times has three
        copies, and only the newest one is the answer being replaced.
        """
        for i in range(len(self._history) - 1, -1, -1):
            turn = self._history[i]
            if (turn.get("role") == "user"
                    and str(turn.get("content") or "") == prompt):
                return i
        return None

    def restored(self) -> bool:
        """True when this window opened on a conversation from before.

        Public for the same reason :meth:`is_busy` is: it is what
        :func:`ask` needs to decide whether to generate anything, and a
        caller reaching for ``_restored`` would pin the dialog to an
        implementation detail.
        """
        return self._restored

    def _append(self, role: str, text: str, meta: Optional[dict] = None) -> None:
        """Add one turn to the shared transcript, write it down, redraw.

        Nothing is trimmed: the transcript is the conversation, and the
        conversation is what travels with the next question. ``meta``
        rides along on assistant turns — how the answer was produced —
        and is rendered here; it is never sent to the agent. Each turn
        also goes to the chat-log folder on disk, which is what survives
        closing the app (see :mod:`bidsmgr.ai_chat_log`).
        """
        entry = {"role": role, "content": str(text)}
        if meta:
            entry["meta"] = meta
        self._history.append(entry)
        self._write_log_turn(entry)
        self._render()

    def _write_log_turn(self, turn: dict) -> None:
        """Append one turn to the conversation's JSONL file.

        The folder is read on every turn rather than at window open, so
        a path just saved in Settings is used by the next question
        without restarting anything. Writing cannot fail the answer:
        :func:`bidsmgr.ai_chat_log.write_turn` reports and swallows its
        own I/O errors.
        """
        from ..ai_chat_log import write_turn

        write_turn(
            str(_read("ai_chat_log_dir", "") or ""),
            self._key,
            str(turn.get("role") or ""),
            str(turn.get("content") or ""),
            meta=turn.get("meta"),
            finding=self._payload,
        )

    def _chat_colors(self) -> dict[str, BubbleColors]:
        """Bubble colours for both sides, read out of the active theme.

        Green is the side the user is on in every messenger there is,
        and violet is the assistant's. Neither is picked here: they
        come from ``CUR()``, so a light window and a dark one each get
        a transcript that belongs to it instead of a scheme bolted on
        top.
        """
        pal = CUR()
        surface = pal.get("surface2", "#ffffff")
        text = pal.get("text", "#000000")
        out: dict[str, BubbleColors] = {}
        for side, token in (("user", "success"), ("agent", "purple")):
            ink = pal.get(token, "#888888")
            out[side] = BubbleColors(
                name=ink,
                fill=_mix(ink, surface, 0.16),
                border=_mix(ink, surface, 0.42),
                text=text,
            )
        return out

    def _render(self) -> None:
        """Redraw the whole transcript from ``_history``.

        Rebuilt rather than appended to, because the window can be
        opened on a conversation that already exists — rendering is
        what turns that stored list back into a readable exchange.
        """
        if not self._history:
            self._answer.show_placeholder(
                "Nothing asked yet. Your questions and the agent's "
                "answers collect here."
            )
            return
        colors = self._chat_colors()
        self._answer.clear()
        for i, turn in enumerate(self._history):
            mine = turn.get("role") == "user"
            self._answer.add_turn(
                "right" if mine else "left",
                "You" if mine else "AI Agent",
                str(turn.get("content") or ""),
                colors["user" if mine else "agent"],
                meta="" if mine else self._meta_html(turn),
                log=(None if mine or not _has_log(turn)
                     else self._log_opener(i)),
            )
        self._answer.scroll_to_bottom()

    def _meta_html(self, turn: dict) -> str:
        """The footer under an assistant bubble: how the answer was made.

        The checkpoint that wrote it leads (the reader recognises the
        model sooner than the numbers), then the response time, then the
        tokens it produced and the intent actually used. The record —
        including how long the checkpoint took to load before that time
        began — is not a word in this line: it is the button under it.
        A turn without metadata (an older conversation, a restored one
        from before this existed) gets no footer at all.
        """
        meta = turn.get("meta")
        if not isinstance(meta, dict):
            return ""
        bits: list[str] = []
        model = str(meta.get("model") or "").strip()
        if model:
            bits.append(html.escape(model))
        seconds = meta.get("seconds")
        if isinstance(seconds, (int, float)) and seconds > 0:
            bits.append(f"{seconds:.1f} s")
        tokens = meta.get("tokens")
        if isinstance(tokens, int) and tokens > 0:
            bits.append(f"{tokens} tokens")
        intent = str(meta.get("intent") or "").strip()
        if intent and intent != "—":
            bits.append(f"intent: {html.escape(intent)}")
        return "  ·  ".join(bits)

    def _log_opener(self, index: int) -> Callable[[], None]:
        """The click handler for the button under one answer."""
        return lambda: self._open_log(index)

    def _open_log(self, index: int) -> None:
        """Open the whole record behind one answer, then redraw.

        The record is a modal window over this one, so the transcript
        spends the time disabled behind a nested event loop. Redrawing
        from ``_history`` afterwards — the single source of truth — is
        what guarantees coming back to the conversation shows the
        conversation, whatever the window system did to it meanwhile.
        """
        if not 0 <= index < len(self._history):
            return
        if not _has_log(self._history[index]):
            return
        meta = self._history[index].get("meta") or {}
        AiLogDialog(dict(meta.get("log") or {}), parent=self).exec()
        self._render()

    def _send_followup(self) -> None:
        text = self._followup.text().strip()
        if not text or self._busy:
            return
        self._followup.clear()
        self.start(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._retry_btn.setEnabled(not busy)
        self._followup.setEnabled(not busy)
        if busy:
            self._error.setVisible(False)
            # The clock does not start yet. ``/predict`` first loads the
            # checkpoint on the agent's side — a minute of work on a CPU
            # that is not the model answering — so the seconds begin when
            # the agent reports generation (``_on_phase``), not here.
            self._busy_since = 0.0
            self._ticking = False
            self._ticker.stop()
            self._status.show_state("waiting")
        else:
            self._ticker.stop()
            self._ticking = False
            # Nothing by default. Every caller that has something worth
            # reporting writes the line itself as soon as this returns;
            # filling it with the agent's address here would say the
            # same thing after every answer and bury the one sentence
            # that explains a restored window.
            self._status.clear()
            # A download that ran before the question is over: its bar
            # has no business staying visible under a finished answer.
            self._download_bar.setVisible(False)
            self._download_note.setVisible(False)

    def _tick_status(self) -> None:
        """Put the seconds into the action line, once a second.

        Only the number is rewritten: the sentence above it says what
        the request is doing, and a clock overwriting it is what the old
        one-line label did. A model download owns the line with its own
        sentence and progress bar, so the clock stays off there.
        """
        if not self._ticking or not self._busy:
            return
        seconds = max(0, int(time.time() - self._busy_since))
        self._status.set_seconds(seconds)

    def _on_phase(self, phase: str, elapsed: float = 0.0) -> None:
        """The agent's own account of what the request is doing.

        ``loading`` means the checkpoint is still being read: the glyph
        already says the request is alive, and the clock has nothing to
        count yet. ``generating`` is the moment the answer starts being
        written, and the moment the clock starts — anchored a little in
        the past when the poll landed after generation began, so the
        first second on screen is honest.

        ``unknown`` comes from an agent too old to have ``/status``: the
        best anchor left is the request itself, which is what this
        window used before the phase existed.
        """
        if not self._busy:
            return
        if phase == "loading":
            if not self._ticking:
                self._status.show_state("loading")
            return
        if phase not in ("generating", "unknown") or self._ticking:
            return
        self._ticking = True
        if phase == "generating":
            anchor = max(0.0, float(elapsed or 0.0))
        else:
            anchor = time.time() - self._started if self._started else 0.0
        self._busy_since = time.time() - anchor
        self._status.show_state("generating")
        self._tick_status()
        self._ticker.start()

    def _on_model_missing(self, model: str, info: dict) -> None:
        """The configured model is not downloaded: fetch it visibly.

        The status line says the fetch is running, the bar and note say
        how far it has got — and no request is in flight, so the request
        timeout cannot fire while gigabytes arrive.
        """
        self._download_note.setText(
            f"Downloading {model} — the size is being looked up ..."
        )
        self._download_note.setVisible(True)
        self._download_bar.setVisible(True)
        self._download_bar.setRange(0, 0)
        # The footer belongs to the download now: the clock would write
        # over the one sentence that says why nothing is being answered.
        self._ticking = False
        self._ticker.stop()
        self._status.show_state("downloading")

    def _on_download_progress(self, status: dict) -> None:
        """One poll of the download, into bar + note, in bytes."""
        if str(status.get("status") or "") == "done":
            # 100% is not the end of the wait: the model still has to
            # load and the answer be written, and a bar sitting at 100%
            # reads as stuck.
            self._download_bar.setVisible(False)
            self._download_note.setText(
                "Download finished — loading the model and writing the "
                "answer. The first answer after a download takes a moment."
            )
            # The download is over; the model still has to be read into
            # memory. The clock starts when the agent says it is writing
            # (``_on_phase``), not while that load runs — and if that
            # report won the race, the clock is already right and stays.
            if not self._ticking:
                self._ticker.stop()
                self._status.show_state("loading")
            return
        downloaded = int(status.get("downloaded_bytes") or 0)
        total = status.get("total_bytes")
        if total:
            total = int(total)
            # A percentage bar, not bytes: QProgressBar is an int32, and
            # a multi-GB model would overflow it. The bytes still read in
            # the note, where a person can act on them.
            pct = min(100, int(100 * downloaded / total)) if total else 0
            self._download_bar.setRange(0, 100)
            self._download_bar.setValue(pct)
            self._download_note.setText(
                f"Downloading — {_human(downloaded)} of {_human(total)} "
                f"({pct}%). This runs once; later questions use the "
                "local copy."
            )
        else:
            self._download_bar.setRange(0, 0)
            self._download_note.setText(
                f"Downloading — {_human(downloaded)} so far ..."
            )

    def _on_done(self, result: dict) -> None:
        elapsed = result.get("response_seconds")
        if not isinstance(elapsed, (int, float)) or elapsed <= 0:
            elapsed = (time.time() - self._started) if self._started else 0.0
        if self._retry_from is not None:
            # A re-ask replaces the answer it is re-asking about: the
            # question stays, the old answer comes out, and the new one
            # takes its place instead of stacking under it.
            del self._history[self._retry_from + 1:]
            self._retry_from = None
        # The answer joins the transcript before anything else happens,
        # so the window is never showing a question with no reply. The
        # metadata rides on the turn itself, so the footer renders under
        # this message and survives a restore.
        meta = {
            "seconds": float(elapsed),
            "model": str(result.get("model") or ""),
            "tokens": result.get("generated_tokens"),
            "intent": str(result.get("intent") or ""),
            "log": result.get("log") if isinstance(result.get("log"), dict) else {},
        }
        self._append("assistant", str(result.get("response") or ""), meta=meta)
        self._set_busy(False)
        self._error.setVisible(False)
        # What the answer was built with now reads under the message
        # itself (time, model, tokens, intent) with its own button to the
        # whole record; the summary row keeps only what is true of the
        # conversation as a whole.
        answers = sum(
            1 for t in self._history if t.get("role") == "assistant"
        )
        if answers > 1:
            self._meta.setText(f"{answers} answers in this conversation")
        else:
            self._meta.setText("")
        self._followup.setFocus()

    def _on_failed(self, message: str) -> None:
        # A retry that fails changes nothing: the question and the
        # answer it was replacing are still on screen, still in the
        # transcript, and "Ask again" is one click from trying again.
        self._retry_from = None
        # A question nobody answered is not part of the conversation, so
        # it comes back out of the transcript and into the box it came
        # from: one Enter away from a retry instead of retyping, and no
        # run of unanswered turns left behind for the next opener to
        # wonder about.
        if self._history and self._history[-1].get("role") == "user":
            orphan = str(self._history[-1].get("content") or "")
            self._history.pop()
            self._render()
            if not self._followup.text().strip():
                self._followup.setText(orphan)
        self._set_busy(False)
        # A WrapLabel, not a QLabel: this is a short paragraph of prose
        # and it has to wrap to the card's width.
        self._error.setText(message)
        self._error.setVisible(True)
        self._meta.setText("")
        self._status.show_state("unreachable")

    def _copy(self) -> None:
        text = self._answer.toPlainText()
        if not text:
            return
        QApplication.clipboard().setText(text)
        # Never over a request in flight: the line is the only place
        # that says what is running, and "copied" is not an answer.
        if not self._busy:
            self._status.show_state("copied")


# =====================================================================
# The prompt log behind one answer
# =====================================================================

class AiLogDialog(QDialog):
    """The record of one exchange: what the model was fed, and how.

    Opened from the "Show full data" button under an answer. This
    replaces the ``prompt_log.txt`` file the agent used to write — the
    same sections (system prompt, user prompt, output) plus the metadata
    beside them, shown in the app instead of sitting in a file nobody
    knew to look for. Read-only and modal: it is a reference, not a
    second window to lose in the taskbar.
    """

    def __init__(self, snapshot: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BIDS-Manager - Prompt Log")
        self.resize(720, 640)
        self.setMinimumWidth(480)
        self._snapshot = dict(snapshot or {})

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Prompt log", "What the model was fed, and the answer it produced",
        ))

        scroll, body = scrollable_body()
        body.setSpacing(12)

        frame, lay = card("How the answer was produced")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(4)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        def fact(name: str, value: str) -> None:
            label = QLabel(value)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            label.setWordWrap(True)
            form.addRow(self._caption(name), label)

        seconds = self._snapshot.get("generation_seconds")
        if isinstance(seconds, (int, float)) and seconds > 0:
            fact("Response time", f"{seconds:.1f} s")
        load = self._snapshot.get("model_load_seconds")
        if isinstance(load, (int, float)) and load >= 0:
            fact("Model load", f"{load:.1f} s")
        model = str(self._snapshot.get("model") or "").strip()
        if model:
            fact("Model", model)
        prompt_tokens = self._snapshot.get("prompt_tokens")
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            fact("Prompt tokens", str(prompt_tokens))
        generated = self._snapshot.get("generated_tokens")
        if isinstance(generated, int) and generated > 0:
            fact("Generated tokens", str(generated))
        if form.rowCount():
            lay.addLayout(form)
        else:
            lay.addWidget(WrapLabel(
                "The agent that wrote this answer did not report how it "
                "was produced — it is an older version."
            ))
        body.addWidget(frame)

        frame2, lay2 = card("The prompt, as the model saw it")
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setMinimumHeight(200)
        font = QFont()
        families = mono_families()
        if families:
            font.setFamilies(families)
        else:
            font.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(font)
        self._text.setPlainText(self._rendered_log())
        lay2.addWidget(self._text)
        body.addWidget(frame2)

        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        copy_btn = buttons.addButton(
            "Copy", QDialogButtonBox.ButtonRole.ActionRole,
        )
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        copy_btn.clicked.connect(self._copy)
        buttons.rejected.connect(self.reject)
        outer.addWidget(build_footer(buttons))

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("val-caption")
        return label

    def _rendered_log(self) -> str:
        """The three sections, exactly as the old prompt_log.txt laid
        them out, so a log read in one place reads the same in the other."""
        s = self._snapshot
        sections = (
            ("[System prompt]", str(s.get("system_prompt") or "")),
            ("[User prompt]", str(s.get("user_prompt") or "")),
            ("[Agent output]", str(s.get("output") or "")),
        )
        out = []
        for title, content in sections:
            rule = "-" * len(title)
            out.append(f"{title}\n{rule}\n{content or '(none)'}")
        return "\n\n".join(out)

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._text.toPlainText())


# =====================================================================
# Entry point
# =====================================================================

def ask(
    payload: dict, parent=None, root: Any = None,
) -> Optional[AiExplainDialog]:
    """Open an explanation window for *payload*, generating only if needed.

    Returns the dialog (so a caller could, say, test it) or ``None``
    when the feature is switched off in Settings. Modeless and
    ``WA_DeleteOnClose``: findings are clicked one after another and a
    window that outlives its purpose is a leak.

    A finding that has been asked about before opens on that
    conversation with **no** request: the user is coming back to
    something they already read, and regenerating would spend a minute
    of model time to hand them a different answer to the same question.

    *root* is the dataset the finding belongs to, passed by callers
    that know it so the window can show the agent the dataset's shape
    and read the offending file. Without it the dialog falls back to
    whatever an absolute ``payload["path"]`` can establish on its own,
    and sends neither when there is nothing to establish.
    """
    if not ai_enabled():
        return None
    dlg = AiExplainDialog(payload, parent=parent, root=root)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    if not dlg.restored():
        dlg.start()
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


def hardware_text(hardware: Any) -> tuple[str, str]:
    """``(text, tooltip)`` for the Hardware row in Settings.

    The agent reports what PyTorch in *its* process can actually open,
    which is the only thing that answers "what will be used to generate
    my answers" — a desktop app can see that a card exists, not whether
    the runtime it will hand it to was built for it.

    Deliberately wording only: the detection has to stay in the agent,
    and this has to be testable without one.
    """
    if not isinstance(hardware, dict) or not hardware.get("backend"):
        return (
            "Not reported yet.",
            "The agent reports this as soon as it is running — the "
            "detection loads no model, so it costs nothing. Press "
            "<b>Test connection</b> to fetch it.",
        )

    backend = str(hardware.get("backend") or "").strip().lower()
    name = str(hardware.get("device_name") or "").strip()
    cores = hardware.get("cpu_cores")

    if backend == "cuda":
        bits = [name or "NVIDIA GPU", "CUDA"]
        if hardware.get("cuda_version"):
            bits[1] = f"CUDA {hardware['cuda_version']}"
        if hardware.get("memory_gb"):
            bits.append(f"{hardware['memory_gb']} GB")
        count = int(hardware.get("device_count") or 0)
        if count > 1:
            bits.append(f"{count} cards")
        text = "  ·  ".join(b for b in bits if b)
    elif backend == "mps":
        text = f"{name or 'Apple GPU'}  ·  Apple GPU (MPS)"
    elif backend == "cpu":
        text = "CPU only"
        if cores:
            text += f"  ·  {cores} cores"
    else:
        text = name or "Unknown"

    tip_bits = [
        b for b in (
            f"PyTorch {hardware.get('torch')}" if hardware.get("torch") else "",
            f"Python {hardware.get('python')}" if hardware.get("python") else "",
        ) if b
    ]
    tip = "Reported by the agent.\n" + "\n".join(tip_bits)
    if backend == "cuda":
        tip += (
            "\n\nThis is what <b>Automatic</b> under Device resolves to, "
            "and where 4-bit and 8-bit memory savings come from."
        )
    elif backend == "mps":
        tip += (
            "\n\nThis is what <b>Automatic</b> under Device resolves to: "
            "answers are generated on the Apple GPU."
        )
    elif backend == "cpu":
        tip += (
            "\n\nNo CUDA or Apple GPU is available to the agent, so "
            "answers are generated on the CPU. That works — it is "
            "slower than a GPU, and <b>Answer length</b> is the setting "
            "that controls how long a single answer takes."
        )
    return text, tip


__all__ = [
    "AgentClient",
    "AgentError",
    "AiExplainDialog",
    "AiLogDialog",
    "DEFAULT_BASE_URL",
    "DEFAULT_PROMPT",
    "ai_enabled",
    "agent_base_url",
    "agent_timeout",
    "ask",
    "build_payload",
    "conversation_key",
    "dataset_tree",
    "default_prompt",
    "file_digest",
    "hardware_text",
    "plain_text",
]
