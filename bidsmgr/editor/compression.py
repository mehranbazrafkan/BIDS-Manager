"""Compressing and decompressing the NIfTI images of a dataset.

BIDS allows an image as ``.nii.gz`` or ``.nii``: the same bytes, gzip-compressed
or not. The conversion writes either (Settings > Convert, ``bidsmgr-convert
--no-compress``); this changes the images already in a dataset, in either
direction, and everything that names them:

- the ``filename`` column of every ``*_scans.tsv``;
- every sidecar field that points at a file (``IntendedFor``, ``Sources``, ...,
  :data:`bidsmgr.editor.rename.PATH_FIELDS`), in all three spellings BIDS uses
  (:func:`bidsmgr.editor.rename.build_ref_map`);
- the viewer's saved scenes (``.bidsmgr/viz/scenes/``) and a ``.bidsignore``
  line naming one image.

The sidecar (``x.json``), ``.bval`` and ``.bvec`` keep their names: the
extension is not part of what they share with the image.

The images are re-encoded first, beside their targets and in parallel (zlib
works outside the interpreter lock), and only then is the dataset changed, as
ONE operation of the Editor's history: every new image moved in, every old one
moved aside for Undo (moved, not copied: a dataset's images are gigabytes), and
the references rewritten. Stopping or failing before that leaves the dataset as
it was; failing during it rolls it back. Gzip is written the reproducible way:
no file name and no time in the header, so compressing the same image twice
gives the same bytes.

``derivatives/``, ``sourcedata/`` and the other folders the standard leaves to
other tools are not touched: what is in them is named by their pipelines.

Qt-free. The Editor's dialog and ``bidsmgr-compress`` drive the same code.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import struct
import threading
import uuid
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..util.cancel import OperationCancelled

log = logging.getLogger(__name__)

GZ = ".nii.gz"
NII = ".nii"
#: zlib's own default: within a few percent of the smallest at a third of the time.
DEFAULT_LEVEL = 6
_CHUNK = 4 << 20

#: Reasons an image is left as it is.
SKIP_NOT_GZIP = "is named .nii.gz but its data is not compressed"
SKIP_ALREADY_GZIP = "is named .nii but its data is already compressed"
SKIP_NOT_NIFTI = "is not a NIfTI image"
SKIP_UNREADABLE = "could not be read"


def skip_both(name: str) -> str:
    return f"{name} is here too: two copies of one image, neither is changed"


@dataclass(frozen=True)
class ImageChange:
    """One image and the name it gets."""

    source: Path
    target: Path
    rel: str
    target_rel: str
    #: Bytes on disk now.
    size: int
    #: Bytes after: exact when decompressing (the gzip trailer records it),
    #: ``None`` when compressing (known only once done).
    size_after: Optional[int]


@dataclass(frozen=True)
class Reference:
    """A file naming one or more of the images that change."""

    path: Path
    rel: str
    #: What names them: a sidecar field, "scans.tsv filename", "saved scene",
    #: ".bidsignore".
    what: str
    #: The images it names, relative to the dataset.
    images: tuple[str, ...]


@dataclass
class CompressionPlan:
    """What :func:`apply_compression` would do, before any of it is done."""

    root: Path
    compress: bool
    changes: list[ImageChange] = field(default_factory=list)
    #: ``(rel, why)``: images in the scope that cannot be changed.
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: Images in the scope already in the asked form.
    already: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    #: Free bytes on the dataset's disk when planned (``None``: not known).
    free_bytes: Optional[int] = None

    def chosen(self, only: Optional[Iterable[str]] = None) -> list[ImageChange]:
        if only is None:
            return list(self.changes)
        wanted = set(only)
        return [c for c in self.changes if c.rel in wanted]

    def references_for(self, only: Optional[Iterable[str]] = None) -> list[Reference]:
        """The references the chosen images bring along (all when ``None``)."""
        if only is None:
            return list(self.references)
        wanted = set(only)
        return [r for r in self.references if wanted.intersection(r.images)]

    def bytes_now(self, only: Optional[Iterable[str]] = None) -> int:
        return sum(c.size for c in self.chosen(only))

    def bytes_needed(self, only: Optional[Iterable[str]] = None) -> int:
        """The space the new images take while the old ones are kept for
        Undo: their size when known, else (compressing) at most the size now."""
        return sum(c.size if c.size_after is None else c.size_after for c in self.chosen(only))

    def short_of_space(self, only: Optional[Iterable[str]] = None) -> bool:
        return self.free_bytes is not None and self.bytes_needed(only) > self.free_bytes


@dataclass
class CompressionOutcome:
    """What a run did."""

    changed: list[str] = field(default_factory=list)
    references: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    cancelled: bool = False
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.changed) and not self.failed and not self.cancelled


# --------------------------------------------------------------------------
# Reading


def is_image(name: str) -> bool:
    return name.endswith((GZ, NII))


def target_name(name: str, compress: bool) -> str:
    """``x.nii`` -> ``x.nii.gz`` (``compress``), ``x.nii.gz`` -> ``x.nii``."""
    if compress:
        return name if name.endswith(GZ) else name + ".gz"
    return name[: -len(".gz")] if name.endswith(GZ) else name


def _head(path: Path, gzipped: bool, n: int = 4) -> bytes:
    if gzipped:
        with gzip.open(path, "rb") as fh:
            return fh.read(n)
    with open(path, "rb") as fh:
        return fh.read(n)


def is_gzip(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def looks_like_nifti(head: bytes) -> bool:
    """A NIfTI-1 header is 348 bytes long and says so first, NIfTI-2 540,
    in either byte order."""
    if len(head) < 4:
        return False
    return any(struct.unpack(order + "i", head[:4])[0] in (348, 540) for order in "<>")


def gzip_size(path: Path) -> Optional[int]:
    """The uncompressed size a gzip file records in its last four bytes
    (modulo 4 GiB, so trusted only when it is not smaller than the file)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            value = struct.unpack("<I", fh.read(4))[0]
    except (OSError, struct.error):
        return None
    return value if value >= size else None


def find_images(root: Path, targets: Optional[Iterable[Path]] = None) -> list[Path]:
    """The NIfTI images under ``targets`` (the whole dataset when ``None``),
    outside the folders other tools own and the hidden ones."""
    from .rename import _skip, walk_dataset

    root = Path(root)
    if targets is None:
        return sorted(p for p in walk_dataset(root) if is_image(p.name))
    out: set[Path] = set()
    for target in targets:
        target = Path(target)
        if target.is_file():
            if is_image(target.name) and not _skip(target, root):
                out.add(target)
        elif target.is_dir():
            if target == root:
                out.update(p for p in walk_dataset(root) if is_image(p.name))
                continue
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                base = Path(dirpath)
                out.update(base / n for n in filenames if is_image(n))
    return sorted(p for p in out if not _skip(p, root))


def _rel(root: Path, path: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def plan_compression(root: Path, compress: bool,
                     targets: Optional[Iterable[Path]] = None) -> CompressionPlan:
    """Which images change, which cannot and why, and what names them."""
    root = Path(root)
    plan = CompressionPlan(root=root, compress=compress)
    for path in find_images(root, targets):
        rel = _rel(root, path)
        if path.name.endswith(GZ) == compress:
            plan.already.append(rel)
            continue
        target = path.with_name(target_name(path.name, compress))
        if target.exists():
            plan.skipped.append((rel, skip_both(target.name)))
            continue
        try:
            gzipped = is_gzip(path)
            if compress and gzipped:
                plan.skipped.append((rel, SKIP_ALREADY_GZIP))
                continue
            if not compress and not gzipped:
                plan.skipped.append((rel, SKIP_NOT_GZIP))
                continue
            if not looks_like_nifti(_head(path, gzipped)):
                plan.skipped.append((rel, SKIP_NOT_NIFTI))
                continue
            size = path.stat().st_size
        except (OSError, EOFError, zlib.error):
            plan.skipped.append((rel, SKIP_UNREADABLE))
            continue
        plan.changes.append(ImageChange(
            source=path, target=target, rel=rel, target_rel=_rel(root, target), size=size,
            size_after=None if compress else gzip_size(path)))
    plan.references = find_references(root, plan.changes)
    try:
        plan.free_bytes = shutil.disk_usage(root).free
    except OSError:
        plan.free_bytes = None
    return plan


def _spellings(root: Path, changes: list[ImageChange]) -> dict[str, str]:
    """Every spelling of where an image is (``build_ref_map``'s three) -> its
    path relative to the dataset."""
    from .rename import _subject_dir, scans_home

    out: dict[str, str] = {}
    for c in changes:
        out[c.rel] = c.rel
        for base in (_subject_dir(root, c.source), scans_home(c.source, root)):
            if base is not None:
                out[_rel(base, c.source)] = c.rel
    return out


def find_references(root: Path, changes: list[ImageChange]) -> list[Reference]:
    """Every file that names one of ``changes``, and which ones it names."""
    from .rename import PATH_FIELDS, _load_json, _split_uri, walk_dataset

    if not changes:
        return []
    root = Path(root)
    spelled = _spellings(root, changes)
    by_rel = {c.rel for c in changes}
    out: list[Reference] = []

    def named(values) -> tuple[str, ...]:
        hits = [spelled.get(_split_uri(str(v))[1]) for v in values]
        return tuple(dict.fromkeys(h for h in hits if h))

    for path in walk_dataset(root):
        name = path.name
        if name.endswith(".json"):
            data = _load_json(path)
            if not data:
                continue
            for key in PATH_FIELDS:
                if key in data:
                    value = data[key]
                    images = named(value if isinstance(value, list) else [value])
                    if images:
                        out.append(Reference(path, _rel(root, path), key, images))
        elif name.endswith("_scans.tsv"):
            from .tsv_edit import read_table

            table = read_table(path)
            if table is not None and "filename" in table.header:
                images = named(table.column("filename"))
                if images:
                    out.append(Reference(path, _rel(root, path), "scans.tsv filename", images))
    for path in _scene_files(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        images = tuple(dict.fromkeys(p for p in _scene_paths(data) if p in by_rel))
        if images:
            out.append(Reference(path, _rel(root, path), "saved scene", images))
    ignore = root / ".bidsignore"
    if ignore.is_file():
        try:
            lines = ignore.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        images = tuple(dict.fromkeys(
            line.strip().lstrip("!").lstrip("/") for line in lines
            if line.strip().lstrip("!").lstrip("/") in by_rel))
        if images:
            out.append(Reference(ignore, ".bidsignore", ".bidsignore", images))
    return out


def _scene_files(root: Path) -> list[Path]:
    from ..viz.scenes import scenes_dir

    folder = scenes_dir(root)
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def _scene_paths(data: dict) -> list[str]:
    out = [str(data.get("base") or "")]
    out += [str(o.get("path") or "") for o in data.get("overlays") or [] if isinstance(o, dict)]
    return [p for p in out if p]


# --------------------------------------------------------------------------
# Writing


def transcode(source: Path, target: Path, compress: bool, *, level: int = DEFAULT_LEVEL,
              stop: Optional[threading.Event] = None) -> int:
    """Write ``source`` to ``target`` compressed (``compress``) or not, and
    return the bytes written. Reproducible gzip: no name, no time. Reading a
    gzip file checks its CRC and length, so a damaged one raises here."""
    source, target = Path(source), Path(target)
    if compress:
        with open(source, "rb") as src, open(target, "wb") as raw, \
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=level,
                              mtime=0) as out:
            _copy(src, out, stop)
    else:
        with gzip.open(source, "rb") as src, open(target, "wb") as out:
            _copy(src, out, stop)
    shutil.copymode(source, target)
    return target.stat().st_size


def _copy(src, out, stop: Optional[threading.Event]) -> None:
    while True:
        if stop is not None and stop.is_set():
            raise OperationCancelled("stopped by the user")
        chunk = src.read(_CHUNK)
        if not chunk:
            return
        out.write(chunk)


def _temp_for(target: Path, token: str) -> Path:
    return target.with_name(f".{target.name}.{token}.part")


def default_jobs() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def label_for(plan: CompressionPlan, n: int) -> str:
    images = f"{n} image{'' if n == 1 else 's'}"
    return (f"Compress {images} (.nii to .nii.gz)" if plan.compress
            else f"Decompress {images} (.nii.gz to .nii)")


def apply_compression(
    root: Path,
    plan: CompressionPlan,
    *,
    only: Optional[Iterable[str]] = None,
    level: int = DEFAULT_LEVEL,
    jobs: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], None]] = None,
    label: Optional[str] = None,
) -> CompressionOutcome:
    """Change the chosen images (all of ``plan``'s when ``only`` is
    ``None``) and their references, as ONE operation.

    ``progress(done, total, rel)`` after each image is re-encoded (``rel`` ""
    at the end); ``cancel_check()`` raises :class:`OperationCancelled` to stop,
    which leaves the dataset as it was.
    """
    from ..project.operations import OperationError, begin_operation
    from .rename import PATH_FIELDS, _rewrite_column, _rewrite_path_field, build_ref_map

    root = Path(root)
    chosen = plan.chosen(only)
    outcome = CompressionOutcome()
    if not chosen:
        return outcome
    token = uuid.uuid4().hex[:8]
    temps = {c.rel: _temp_for(c.target, token) for c in chosen}
    stop = threading.Event()
    total = len(chosen)
    sizes: dict[str, int] = {}
    current = ""
    try:
        # 1. Re-encode beside each target, nothing in the dataset changed yet.
        with ThreadPoolExecutor(max_workers=jobs or default_jobs()) as pool:
            pending = {pool.submit(transcode, c.source, temps[c.rel], plan.compress,
                                   level=level, stop=stop): c for c in chosen}
            done = 0
            try:
                while pending:
                    finished, _rest = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    if cancel_check is not None:
                        cancel_check()
                    for fut in finished:
                        c = pending.pop(fut)
                        current = c.rel
                        sizes[c.rel] = fut.result()
                        done += 1
                        if progress is not None:
                            progress(done, total, c.rel)
            except BaseException:
                stop.set()
                raise
        current = ""
        # 2. One operation: the new images in, the old aside, the names followed.
        with begin_operation(root, label or label_for(plan, len(chosen))) as op:
            for c in chosen:
                current = c.rel
                op.adopt(c.target, temps[c.rel])
                op.set_aside(c.source, same_as=c.target)
            current = ""
            ref_map = build_ref_map(root, [(c.source, c.target) for c in chosen])
            by_rel = {c.rel: c.target_rel for c in chosen}
            for ref in plan.references_for([c.rel for c in chosen]):
                if ref.what in PATH_FIELDS:
                    _rewrite_path_field(op, ref.path, ref.what, "", "", ref_map=ref_map)
                elif ref.what == "scans.tsv filename":
                    _rewrite_column(op, ref.path, "filename", "", "", ref_map=ref_map)
                elif ref.what == "saved scene":
                    _rewrite_scene(op, ref.path, by_rel)
                elif ref.what == ".bidsignore":
                    _rewrite_bidsignore(op, ref.path, by_rel)
                outcome.references += 1
        if progress is not None:
            progress(total, total, "")
        outcome.changed = [c.rel for c in chosen]
        outcome.bytes_before = sum(c.size for c in chosen)
        outcome.bytes_after = sum(sizes.values())
    except OperationCancelled:
        outcome.cancelled = True
        outcome.references = 0
        log.info("compression stopped by the user; the dataset is as it was")
    except (OperationError, OSError, EOFError, zlib.error) as exc:
        outcome.references = 0
        outcome.failed.append((current or "the dataset", str(exc)))
        log.warning("compression rolled back at %s: %s", current or "the dataset", exc)
    finally:
        for tmp in temps.values():
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return outcome


def _rewrite_scene(op, path: Path, by_rel: dict[str, str]) -> None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if data.get("base") in by_rel:
        data["base"] = by_rel[data["base"]]
    for overlay in data.get("overlays") or []:
        if isinstance(overlay, dict) and overlay.get("path") in by_rel:
            overlay["path"] = by_rel[overlay["path"]]
    op.write_text(path, json.dumps(data, indent=2))


def _rewrite_bidsignore(op, path: Path, by_rel: dict[str, str]) -> None:
    text = Path(path).read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        body = line.strip()
        bang = "!" if body.startswith("!") else ""
        rest = body[len(bang):]
        slash = "/" if rest.startswith("/") else ""
        key = rest[len(slash):]
        out.append(f"{bang}{slash}{by_rel[key]}" if key in by_rel else line)
    op.write_text(path, "\n".join(out) + ("\n" if text.endswith("\n") else ""))


__all__ = [
    "CompressionOutcome", "CompressionPlan", "DEFAULT_LEVEL", "ImageChange", "Reference",
    "apply_compression", "default_jobs", "find_images", "find_references", "gzip_size",
    "is_gzip", "is_image", "label_for", "looks_like_nifti", "plan_compression", "target_name",
    "transcode",
]
