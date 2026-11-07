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

import csv
import hashlib
import html
import json
import logging
import re
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
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
from .theme_manager import CUR
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

    def predict(
        self,
        user_input: str,
        context: dict,
        history: Optional[list] = None,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> dict:
        """Run the agent. Returns ``{response, intent, tool_used}``.

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
        turns = [
            t for t in (history or [])
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
            result = AgentClient().predict(
                self._user_input, self._context, history=self._history,
                dataset_structure=self._dataset_structure,
                file_content=self._file_content,
            )
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
    history: Optional[list] = None,
    dataset_structure: Optional[str] = None,
    file_content: Optional[str] = None,
    on_done,
    on_failed,
) -> _PredictWorker:
    """Start a request, hold the thread until it is genuinely done.

    Both result slots are connected *before* ``start()``. Getting that
    backwards is not a theoretical race: a refused connection fails
    inside a few milliseconds, and a slot connected a moment too late
    never fires — which leaves the dialog stuck on "Working…" forever.
    """
    worker = _PredictWorker(
        user_input, context, history=history,
        dataset_structure=dataset_structure, file_content=file_content,
    )
    worker.done.connect(on_done)
    worker.failed.connect(on_failed)
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


def chat_bubble(side: str, who: str, text: str, colors: BubbleColors) -> str:
    """One turn of the transcript as HTML, as a bubble on *side*.

    The spacer cell beside it is what keeps a long answer from
    swallowing the window: Qt shrinks a table to fit its content, so
    without something to push against, an eight-line answer would paint
    itself edge to edge and stop reading as a message.
    """
    body = html.escape(str(text or "")).replace("\r\n", "\n")
    body = body.replace("\n", "<br>")
    inner = (
        '<table cellspacing="0" cellpadding="6" '
        f'style="background-color:{colors.fill}; '
        f'border:1px solid {colors.border};">'
        f'<tr><td style="color:{colors.text};">'
        f'<span style="color:{colors.name}; font-weight:bold;">'
        f"{html.escape(who)}</span>"
        f"<br>{body}</td></tr></table>"
    )
    if side == "right":
        cells = (
            '<td width="25%"></td>'
            f'<td width="75%" align="right" valign="top">{inner}</td>'
        )
    else:
        cells = (
            f'<td width="75%" align="left" valign="top">{inner}</td>'
            '<td width="25%"></td>'
        )
    return (
        '<table width="100%" cellspacing="0" cellpadding="0" '
        'style="margin-bottom:7px;">'
        f"<tr>{cells}</tr></table>"
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
        self._last_prompt = (
            self._previous_prompt() if self._restored else DEFAULT_PROMPT
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
        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        outer.addWidget(build_footer_with(self._status, buttons))

        buttons.rejected.connect(self.reject)
        # A retry, not a second question: ``start`` holds the question
        # and the answer it replaces back from the request.
        self._retry_btn.clicked.connect(
            lambda: self.start(self._last_prompt, retry=True),
        )
        self._copy_btn.clicked.connect(self._copy)

        self._set_busy(False)
        self._render()
        if self._restored:
            # Spoken after ``_set_busy(False)``, which clears the line:
            # this window opens on a conversation from before, and
            # "nothing was re-generated" is the whole point of it.
            self._status.setText(
                "Earlier conversation restored — nothing was re-generated."
            )

    def _previous_prompt(self) -> str:
        """The question to put back in front of "Ask again" on restore."""
        for turn in reversed(self._history):
            if turn.get("role") == "user":
                return str(turn.get("content") or "") or DEFAULT_PROMPT
        return DEFAULT_PROMPT

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

        self._spinner = BusySpinner()
        lay.addWidget(self._spinner)

        self._error = WrapLabel("")
        self._error.setObjectName("ai-error")
        self._error.setTextFormat(Qt.TextFormat.PlainText)
        self._error.setVisible(False)
        lay.addWidget(self._error)

        # A transcript rather than a single answer box: every turn is
        # rendered together from ``_history``, so the window can be
        # rebuilt from scratch on restore without re-asking anything.
        self._answer = QTextEdit()
        self._answer.setObjectName("ai-answer")
        self._answer.setReadOnly(True)
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

    def start(self, prompt: str = DEFAULT_PROMPT, *, retry: bool = False) -> None:
        """Ask *prompt* about the finding shown in this dialog.

        ``retry`` re-asks the question that is already in the transcript
        — what the "Ask again" button does. It must not ask it twice:
        the question and the answer it replaces are held back from the
        request, so the model sees the conversation that came *before*
        them and simply answers again. Nothing on screen changes until
        that answer arrives, so a retry that fails leaves the previous
        answer where it was.
        """
        if self._busy:
            return
        prompt = str(prompt or "").strip() or DEFAULT_PROMPT
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

    def _append(self, role: str, text: str) -> None:
        """Add one turn to the shared transcript and redraw.

        Nothing is trimmed: the transcript is the conversation, and the
        conversation is what travels with the next question.
        """
        self._history.append({"role": role, "content": str(text)})
        self._render()

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
            self._answer.clear()
            self._answer.setPlaceholderText(
                "Nothing asked yet. Your questions and the agent's "
                "answers collect here."
            )
            return
        colors = self._chat_colors()
        blocks = []
        for turn in self._history:
            mine = turn.get("role") == "user"
            blocks.append(
                chat_bubble(
                    "right" if mine else "left",
                    "You" if mine else "AI Agent",
                    str(turn.get("content") or ""),
                    colors["user" if mine else "agent"],
                )
            )
        self._answer.setHtml("".join(blocks))
        bar = self._answer.verticalScrollBar()
        bar.setValue(bar.maximum())

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
            # Nothing by default. Every caller that has something worth
            # reporting writes the line itself as soon as this returns;
            # filling it with the agent's address here would say the
            # same thing after every answer and bury the one sentence
            # that explains a restored window.
            self._status.clear()

    def _on_done(self, result: dict) -> None:
        elapsed = time.time() - self._started
        if self._retry_from is not None:
            # A re-ask replaces the answer it is re-asking about: the
            # question stays, the old answer comes out, and the new one
            # takes its place instead of stacking under it.
            del self._history[self._retry_from + 1:]
            self._retry_from = None
        # The answer joins the transcript before anything else happens,
        # so the window is never showing a question with no reply.
        self._append("assistant", str(result.get("response") or ""))
        self._set_busy(False)
        self._error.setVisible(False)
        intent = str(result.get("intent") or "").strip() or "—"
        tool = str(result.get("tool_used") or "").strip()
        answers = sum(
            1 for t in self._history if t.get("role") == "assistant"
        )
        bits: list[str] = []
        if answers > 1:
            bits.append(f"{answers} answers in this conversation")
        bits.append(f"intent: {intent}")
        if tool:
            bits.append(f"tool: {tool}")
        if answers > 1:
            bits.append(f"last answer {elapsed:.1f}s")
        else:
            bits.append(f"{elapsed:.1f}s")
        self._meta.setText("  ·  ".join(bits))
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
        self._status.setText("Conversation copied to the clipboard.")


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
    "DEFAULT_BASE_URL",
    "DEFAULT_PROMPT",
    "ai_enabled",
    "agent_base_url",
    "agent_timeout",
    "ask",
    "build_payload",
    "conversation_key",
    "dataset_tree",
    "file_digest",
    "hardware_text",
    "plain_text",
]
