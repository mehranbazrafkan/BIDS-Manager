"""Export the request side of every validation finding.

Settings → AI Agent offers one button that writes the "full data" of every
error and warning in the current validation report: the system prompt the
agent would send (instructions, retrieved BIDS knowledge, issue context,
dataset structure, file digest), the user prompt, and how the request was
routed — with no answer generated and no model loaded.

The point is a dataset, not a conversation. Each record is the input half
of a fine-tuning pair (``output`` is deliberately ``null``), and the
retrieved evidence in ``system_prompt`` shows where the knowledge base
helped and where it did not, so the same files serve a knowledge-base
review.

Every export gets its own timestamped folder inside the chat-log path, so
a run that has been annotated is never overwritten by the next one. One
JSON file per finding, plus a manifest describing the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ..ai_chat_log import file_stem, logs_dir
from ..editor.types import Issue, ValidationReport
from .ai_explainer import (
    AgentClient,
    AgentError,
    build_payload,
    conversation_key,
    dataset_tree,
    default_prompt,
    file_digest,
)

log = logging.getLogger(__name__)

#: Version of the record shape. Bumped if the keys change enough that a
#: reader (or a fine-tuning script) would have to handle two formats.
FORMAT = "bidsmgr-ai-full-data/1"

#: Characters a folder name may keep; everything else becomes a dash.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class FindingRequest:
    """One finding's prompt request, ready for ``POST /prompt``."""

    key: str
    prompt: str
    payload: dict
    #: Relative path the dataset tree should keep in view. The finding's
    #: file, its folder, or "" for a dataset-level finding.
    focus: str = ""


def collect_requests(report: ValidationReport) -> list[FindingRequest]:
    """The prompt request for every error and warning in *report*.

    File findings carry their file's path, datatype and suffix — what the
    retriever keys on to pick the right BIDS rule. Folder findings name
    their folder and say so; dataset findings carry no path at all rather
    than borrowing whichever file happened to be selected in the Editor.

    ``ok``/``info`` findings are left out (they are not problems), and so
    are mirrored duplicates: the same sidecar finding surfaced a second
    time on a sibling data file. Accepted findings are kept — they are
    still warnings and errors, and a reviewed "this one is fine" is
    exactly the kind of example a knowledge base wants.
    """
    requests: list[FindingRequest] = []
    for verdict in report.files:
        rel = _relative(verdict.path)
        for issue in verdict.issues:
            if not _is_reported(issue):
                continue
            extra: dict[str, Any] = {}
            if rel:
                extra["path"] = rel
            if verdict.datatype:
                extra["datatype"] = str(verdict.datatype)
            if verdict.suffix:
                extra["suffix"] = str(verdict.suffix)
            requests.append(_request(issue, extra, focus=rel))
    for folder in sorted(report.folder_issues):
        folder_rel = _relative(folder)
        for issue in report.folder_issues[folder]:
            if not _is_reported(issue):
                continue
            extra = {"scope": "folder"}
            if folder_rel:
                extra["folder"] = folder_rel
            requests.append(_request(issue, extra, focus=folder_rel))
    for issue in report.dataset_issues:
        if not _is_reported(issue):
            continue
        requests.append(_request(issue, {"scope": "dataset"}, focus=""))
    return requests


def export_folder(folder: str | Path, root: str | Path) -> Path:
    """A new folder for one export, inside the chat-log path.

    Timestamped so re-running never overwrites records that have been
    reviewed or annotated; the dataset's own folder name makes a run
    recognisable without opening its manifest. A second export in the
    same second gets a numeric suffix rather than sharing the folder.
    """
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    name = _UNSAFE.sub("-", Path(root).name).strip("-._")[:60] or "dataset"
    base = logs_dir(folder) / "full-data" / f"{name}-{stamp}"
    path = base
    index = 2
    while path.exists():
        path = base.with_name(f"{base.name}-{index}")
        index += 1
    return path


def export_records(
    requests: list[FindingRequest],
    root: str | Path,
    folder: str | Path,
    build: Callable[..., dict],
    *,
    on_progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Build and write every request; return the run's summary.

    *build* is :meth:`AgentClient.build_prompt` in the app and a plain
    function in tests: naming, evidence and writing do not need a server.
    The dataset tree and the file digest are compiled here and cached per
    folder/file, because a report can hold thousands of findings over the
    same handful of paths.

    An agent that cannot be reached aborts the run after what was already
    written; a single request the agent rejects is recorded as a failure
    and the rest continue. The manifest is written either way.
    """
    root_path = Path(root)
    run_dir = export_folder(folder, root_path)
    result: dict[str, Any] = {
        "folder": str(run_dir),
        "saved": 0,
        "failed": [],
        "aborted": "",
        "total": len(requests),
    }
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result["aborted"] = f"Could not create {run_dir}: {exc}"
        return result

    trees: dict[str, str] = {}
    digests: dict[str, str] = {}
    for done, request in enumerate(requests, start=1):
        if should_stop is not None and should_stop():
            result["aborted"] = "Export cancelled."
            break
        try:
            focus = request.focus
            if focus not in trees:
                trees[focus] = dataset_tree(root_path, focus=focus)
            built = build(
                request.prompt,
                request.payload,
                dataset_structure=trees[focus],
                file_content=_digest_for(request, root_path, digests),
            )
            record = _record(request, built)
            target = run_dir / f"{file_stem(request.key)}.json"
            target.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str)
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            result["saved"] += 1
        except AgentError as exc:
            if getattr(exc, "status", None) is None:
                # Unreachable agent: every remaining request would fail the
                # same way. Keep what was written and say why it stopped.
                result["aborted"] = str(exc)
                break
            result["failed"].append({"key": request.key, "error": str(exc)})
        except OSError as exc:
            result["failed"].append({"key": request.key, "error": str(exc)})
        except Exception as exc:                       # pragma: no cover
            log.exception("could not export full data for %s", request.key)
            result["failed"].append({
                "key": request.key,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if on_progress is not None:
            on_progress(done, len(requests))
    _write_manifest(run_dir, root_path, requests, result)
    return result


class ExportWorker(QThread):
    """Build and write the full data off the GUI thread.

    One request per finding can take a while on a large report (the agent
    retrieves for each), so the page stays usable: progress is a signal,
    and the result travels as one object rather than a dozen attributes.
    """

    progress = pyqtSignal(int, int)
    finished_with = pyqtSignal(object)

    def __init__(
        self,
        requests: list[FindingRequest],
        root: str | Path,
        folder: str | Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._requests = requests
        self._root = root
        self._folder = folder
        self._stop = False

    def stop(self) -> None:
        """Ask the loop to stop after the finding in flight."""
        self._stop = True

    def run(self) -> None:
        client = AgentClient()
        try:
            result = export_records(
                self._requests,
                self._root,
                self._folder,
                client.build_prompt,
                on_progress=self.progress.emit,
                should_stop=lambda: self._stop,
            )
        except Exception as exc:                       # pragma: no cover
            log.exception("full-data export failed")
            result = {
                "folder": "",
                "saved": 0,
                "failed": [],
                "aborted": f"{type(exc).__name__}: {exc}",
                "total": len(self._requests),
            }
        self.finished_with.emit(result)


#: Workers outlive the Settings dialog they were started from: closing the
#: page while a large report is being written should finish the write, not
#: kill it. The set is what keeps them alive; a signal into a destroyed
#: dialog is dropped by Qt, which is the right outcome.
_RUNNING: set[ExportWorker] = set()


def start_export(worker: ExportWorker) -> ExportWorker:
    """Start one prepared export and keep it alive until done."""
    _RUNNING.add(worker)
    worker.finished_with.connect(
        lambda _result, w=worker: _RUNNING.discard(w)
    )
    worker.start()
    return worker


def _request(
    issue: Issue,
    extra: dict[str, Any],
    *,
    focus: str,
) -> FindingRequest:
    """One issue as the payload and opening question the app would send."""
    payload = build_payload(
        severity=_severity_text(issue.severity),
        message=issue.message,
        rule_id=issue.rule_id,
        field=issue.field or None,
        line=issue.line,
        lines=list(issue.lines or []),
        fix_label=issue.fix_label,
        fix_action=issue.fix_action,
        mirrored=bool(issue.mirrored),
        extra=extra,
    )
    return FindingRequest(
        key=_key_for(payload),
        prompt=default_prompt(payload),
        payload=payload,
        focus=focus,
    )


def _is_reported(issue: Issue) -> bool:
    """Whether *issue* is an error or warning worth exporting."""
    return (
        _severity_text(issue.severity) in ("err", "warn")
        and not bool(issue.mirrored)
    )


def _severity_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _relative(value: Any) -> str:
    """A dataset-relative path in the form the prompt uses (POSIX)."""
    text = str(value or "").strip()
    return Path(text).as_posix() if text else ""


def _key_for(payload: dict) -> str:
    """A stable key unique to this finding's exact payload.

    The conversation key names the finding (rule, file, field) and is what
    chat logs file under; the digest adds the message, because two
    dataset-level findings can share the rule and field while saying
    different things.
    """
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str,
    )
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    return f"{conversation_key(payload)}|{digest}"


def _digest_for(
    request: FindingRequest,
    root: Path,
    cache: dict[str, str],
) -> str:
    """The compiled digest of the file a finding is on, or ""."""
    rel = str(request.payload.get("path") or "").strip()
    if not rel:
        return ""
    target = root / rel
    if not target.is_file():
        return ""
    key = str(target)
    if key not in cache:
        try:
            cache[key] = file_digest(target)
        except OSError:
            cache[key] = ""
    return cache[key]


def _record(request: FindingRequest, built: dict) -> dict:
    """The on-disk record: exactly the request side of one answer."""
    return {
        "format": FORMAT,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "key": request.key,
        "finding": request.payload,
        "intent": str(built.get("intent") or ""),
        "classification_source": str(
            built.get("classification_source") or ""
        ),
        "prompt_available": bool(built.get("prompt_available", True)),
        "system_prompt": str(built.get("system_prompt") or ""),
        "user_prompt": str(built.get("user_prompt") or request.prompt),
        "output": None,
        "retrieval": {
            "found": built.get("retrieval_found"),
            "quality": built.get("retrieval_quality"),
        },
        "note": (
            "No answer was generated: this is the request side of the "
            "prompt, saved for review, knowledge-base work and fine-tuning."
        ),
    }


def _write_manifest(
    run_dir: Path,
    root: Path,
    requests: list[FindingRequest],
    result: dict,
) -> None:
    manifest = {
        "format": FORMAT,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(root),
        "findings": len(requests),
        "saved": result["saved"],
        "failed": result["failed"],
        "aborted": result["aborted"],
        "note": (
            "Each .json file holds the prompt a real question would send "
            "(system_prompt, user_prompt, intent, retrieval outcome) and "
            "no answer. `output` is null for that reason."
        ),
    }
    try:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:                             # pragma: no cover
        log.warning("could not write the full-data manifest: %s", exc)


__all__ = [
    "FORMAT",
    "ExportWorker",
    "FindingRequest",
    "collect_requests",
    "export_folder",
    "export_records",
    "start_export",
]
