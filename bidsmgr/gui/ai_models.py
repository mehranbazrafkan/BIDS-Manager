"""The models folder: where downloaded models live, and what is there.

Models are multi-gigabyte, per-user data: they belong in the per-user
application data folder (``%LOCALAPPDATA%\\bidsmgr\\bidsmgr\\models`` on
Windows, ``~/Library/Application Support/bidsmgr/models`` on macOS,
``~/.local/share/bidsmgr/models`` on Linux), not inside a read-only
install directory and not in the hidden Hugging Face default. The
default resolution lives in :mod:`bidsmgr.agent_service` — the same
value is what gets set in the agent's environment at spawn, so the
agent and this module can never disagree about where the folder is.

Inside the folder, models keep the standard huggingface_hub cache
layout — one ``models--<org>--<name>`` folder per model, with ``refs``,
``snapshots`` and ``blobs`` inside — so a model copied in by hand (or
fetched by any HF tool) is a model this app can use, and anything
downloaded through the GUI is usable by any HF tool in return.

The agent's ``GET /models`` is the authority on what is downloaded when
it is up (it lists its own cache); the scan here is the fallback for
the moments the agent is not running yet — the Settings tab marks the
dropdown with what is ready to use offline either way.
"""

from __future__ import annotations

from pathlib import Path

from .. import agent_service

#: Where downloaded models are kept: the stored override when the user
#: chose one, else the per-user default.
MODELS_DIR_KEY = "ai_models_dir"


def models_dir(override: str = "") -> Path:
    """The models folder: a stored choice when there is one, else the
    per-user default. Same resolution the agent is spawned with."""
    return agent_service.models_dir(override)


def downloaded_model_ids(root: Path | None = None) -> set[str]:
    """Repo ids (``org/name``) of the models already in the local cache.

    The honest-enough check, mirroring what the agent reports: the
    ``models--<org>--<name>`` folder exists, ``refs`` points at a
    revision, and that revision's snapshot holds at least one file. A
    half-fetched download (an interrupted run leaves ``*.incomplete``
    blobs behind) does not count as ready. Nothing here touches a
    model or the network — one directory listing.
    """
    base = root if root is not None else models_dir()
    if not base.is_dir():
        return set()
    out: set[str] = set()
    try:
        entries = list(base.iterdir())
    except OSError:
        return set()
    for folder in entries:
        name = folder.name
        if not name.startswith("models--") or not folder.is_dir():
            continue
        repo_id = name[len("models--"):].replace("--", "/", 1)
        if _ready(folder):
            out.add(repo_id)
    return out


def partial_model_ids(root: Path | None = None) -> set[str]:
    """Repo ids whose download was started but did not finish.

    An interrupted fetch leaves ``*.incomplete`` blobs behind (the
    snapshots folder may already hold the small files). huggingface_hub
    resumes from those blobs, so such a model is not broken — it is
    half here, and the dropdown says so instead of calling it absent:
    "press Download to resume" is a different story from "nothing
    fetched yet".
    """
    base = root if root is not None else models_dir()
    if not base.is_dir():
        return set()
    out: set[str] = set()
    try:
        entries = list(base.iterdir())
    except OSError:
        return set()
    for folder in entries:
        name = folder.name
        if not name.startswith("models--") or not folder.is_dir():
            continue
        if _ready(folder):
            continue
        blobs = folder / "blobs"
        try:
            if not any(blobs.glob("*.incomplete")):
                continue
        except OSError:
            continue
        out.add(name[len("models--"):].replace("--", "/", 1))
    return out


def _ready(repo: Path) -> bool:
    """Whether one model folder holds a usable snapshot."""
    refs = repo / "refs"
    snapshots = repo / "snapshots"
    if not refs.is_dir() or not snapshots.is_dir():
        return False
    try:
        revision = next(
            (p for p in refs.iterdir() if p.is_file() and p.read_bytes().strip()),
            None,
        )
    except OSError:
        return False
    if revision is None:
        return False
    head = snapshots / revision.read_text(encoding="utf-8", errors="replace").strip()
    if not head.is_dir():
        return False
    try:
        if not any(head.rglob("*")):
            return False
        blobs = repo / "blobs"
        if blobs.is_dir() and any(blobs.glob("*.incomplete")):
            return False
    except OSError:
        return False
    return True


__all__ = [
    "MODELS_DIR_KEY",
    "downloaded_model_ids",
    "models_dir",
    "partial_model_ids",
]
