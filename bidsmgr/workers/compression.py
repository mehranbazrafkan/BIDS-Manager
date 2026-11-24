"""Compress or decompress images off the GUI thread.

Re-encoding a dataset's images takes seconds to minutes; on the GUI thread the
window would stop repainting. The shape every long Editor job here uses (see
:mod:`bidsmgr.workers.deface`): a ``QThread``, a ``progress`` signal per image,
and a cooperative stop that leaves the dataset as it was.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .deface import _Base


class CompressionWorker(_Base):
    """Apply a :class:`~bidsmgr.editor.compression.CompressionPlan`."""

    def __init__(self, root: Path, plan, *, only: Optional[Iterable[str]] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._plan = plan
        self._only = list(only) if only is not None else None

    def run(self) -> None:  # noqa: D401 - QThread entry point
        from ..editor.compression import apply_compression

        self._run_guarded(lambda: apply_compression(
            self._root, self._plan, only=self._only,
            progress=self._emit_progress, cancel_check=self._cancel_check,
        ))
