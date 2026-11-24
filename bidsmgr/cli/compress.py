"""``bidsmgr-compress``: compress or decompress a dataset's NIfTI images.

The Editor's "Compress or decompress images" without the window, over the same
engine (:mod:`bidsmgr.editor.compression`): ``.nii`` to ``.nii.gz``, or back
with ``--decompress``, and every scans table row, sidecar path field, saved
viewer scene and ``.bidsignore`` line that names an image follows it.

Plans first and prints the plan (``--dry-run`` stops there); done, the change
is one step of the Editor's history, undoable from the Editor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ..editor import compression as engine
from ..util.oplog import add_log_arguments, dataset_root_for, log_keywords, logged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-compress",
        description=(
            "Compress a dataset's NIfTI images (.nii to .nii.gz), or decompress them "
            "(--decompress). Both forms are valid BIDS and hold the same image. Every "
            "scans table row, sidecar field (IntendedFor, Sources, ...), saved viewer "
            "scene and .bidsignore line naming an image is updated with it, as one step "
            "of the Editor's history. derivatives/ and sourcedata/ are left as they are."),
    )
    parser.add_argument("root", type=Path, help="The dataset.")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="Images or folders inside the dataset (default: all of it).")
    parser.add_argument("--decompress", action="store_true",
                        help="Write .nii from .nii.gz (default: compress .nii to .nii.gz).")
    parser.add_argument("--level", type=int, default=engine.DEFAULT_LEVEL,
                        choices=range(1, 10), metavar="1-9",
                        help=f"gzip level when compressing (default {engine.DEFAULT_LEVEL}).")
    parser.add_argument("-j", "--jobs", type=int, default=engine.default_jobs(),
                        help="Images re-encoded at once (default %(default)s).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change and stop.")
    parser.add_argument("--json", action="store_true", help="The plan as JSON.")
    add_log_arguments(parser)
    return parser


def _relative(root: Path, path: Path) -> Path:
    return path if path.is_absolute() or path.exists() else root / path


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root
    if not root.is_dir():
        parser.error(f"{root} is not a folder")
    targets = [_relative(root, p) for p in args.paths] or None
    for target in targets or []:
        if not target.exists():
            parser.error(f"{target} does not exist")
        try:
            target.resolve().relative_to(root.resolve())
        except ValueError:
            parser.error(f"{target} is not inside {root}")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return run_compress_cli(root, targets, compress=not args.decompress, level=args.level,
                            jobs=args.jobs, dry_run=args.dry_run, as_json=args.json,
                            **log_keywords(args))


@logged("compress", lambda a: (
    dataset_root_for(Path(a["root"])),
    {"root": a["root"], "paths": [str(p) for p in a["targets"] or []],
     "direction": "compress" if a["compress"] else "decompress", "dry run": a["dry_run"]}))
def run_compress_cli(root: Path, targets=None, *, compress: bool = True,
                     level: int = engine.DEFAULT_LEVEL, jobs: Optional[int] = None,
                     dry_run: bool = False, as_json: bool = False) -> int:
    """Plan, print and (unless ``dry_run``) apply. Returns the exit code."""
    plan = engine.plan_compression(root, compress, targets)
    if as_json:
        print(json.dumps(_as_json(plan), indent=2, ensure_ascii=False))
    else:
        _print(plan)
    if dry_run or not plan.changes:
        return 0
    if plan.short_of_space():
        print(f"Not enough free space: {_human(plan.bytes_needed())} needed while the "
              f"originals are kept for Undo, {_human(plan.free_bytes or 0)} free.",
              file=sys.stderr)
        return 1

    def progress(done: int, total: int, rel: str) -> None:
        if rel and not as_json:
            print(f"  [{done}/{total}] {rel}", flush=True)

    outcome = engine.apply_compression(root, plan, level=level, jobs=jobs, progress=progress)
    if outcome.failed:
        where, why = outcome.failed[0]
        print(f"Nothing was changed: {where}: {why}", file=sys.stderr)
        return 1
    if not as_json:
        n = len(outcome.changed)
        verb = "Compressed" if compress else "Decompressed"
        print(f"{verb} {n} image{'s' if n != 1 else ''}: {_human(outcome.bytes_before)} -> "
              f"{_human(outcome.bytes_after)}; {outcome.references} file"
              f"{'s' if outcome.references != 1 else ''} naming them updated. The "
              "Editor's Undo takes it back.")
    return 0


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _print(plan: engine.CompressionPlan) -> None:
    verb = "compress" if plan.compress else "decompress"
    n = len(plan.changes)
    print(f"{plan.root.name}: {n} image{'s' if n != 1 else ''} to {verb} "
          f"({_human(plan.bytes_now())} now"
          + (f", {_human(plan.bytes_needed())} after" if not plan.compress else "") + ")"
          + (f"; {len(plan.already)} already {verb}ed" if plan.already else ""))
    for change in plan.changes:
        print(f"  {change.rel} -> {Path(change.target_rel).name}")
    for rel, why in plan.skipped:
        print(f"  left as it is: {rel}: {why}")
    for ref in plan.references:
        k = len(ref.images)
        print(f"  follows: {ref.rel} ({ref.what}, {k} image{'s' if k != 1 else ''})")


def _as_json(plan: engine.CompressionPlan) -> dict:
    return {
        "root": str(plan.root),
        "direction": "compress" if plan.compress else "decompress",
        "changes": [{"from": c.rel, "to": c.target_rel, "bytes": c.size,
                     "bytes_after": c.size_after} for c in plan.changes],
        "skipped": [{"path": rel, "why": why} for rel, why in plan.skipped],
        "already": list(plan.already),
        "references": [{"path": r.rel, "what": r.what, "images": list(r.images)}
                       for r in plan.references],
        "bytes_needed": plan.bytes_needed(),
        "free_bytes": plan.free_bytes,
    }


if __name__ == "__main__":
    sys.exit(main())
