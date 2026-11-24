"""Default MRI converter backend: invoke ``dcm2niix`` directly.

The orchestrator (``cli/convert.py``) drives parallelism — one
``ConvertTask`` per call, joblib pool sized by ``-j``. This backend is
synchronous and stateless; ``convert(task, staging_dir)`` runs one
dcm2niix invocation and reports back what landed on disk.

Layout written by ``convert``::

    <staging_dir>/
      _dicoms_<series_uid>/      (the series' DICOMs, linked or copied: util.staging)
      <datatype>/
        <basename>.nii.gz
        <basename>.json
        <basename>.bval           (DWI only)
        <basename>.bvec           (DWI only)
        <basename>_e1.nii.gz      (fmap multi-echo: renamed by fixups/fieldmaps)
        <basename>_e2.nii.gz
        <basename>_ph.nii.gz

``.nii`` instead of ``.nii.gz`` when the task asks for uncompressed images
(``ConvertTask.compress``, Settings > Convert).

The fmap suffix mapping (``_e1`` / ``_e2`` / ``_ph`` →
``magnitude1`` / ``magnitude2`` / ``phasediff``) is applied by
``fixups/fieldmaps.py`` *after* this backend returns, before the
atomic per-subject commit. ``IntendedFor`` lives in
``fixups/intended_for.py`` for the same reason.

Every OTHER series dcm2niix splits into one file per echo (a multi-echo GRE,
a multi-echo BOLD or MPRAGE) is named here, by :func:`_name_echoes`: its
``_e<n>`` / ``_e<n>_ph`` tokens become the ``echo`` (and ``part``) entities,
where the standard allows them for that suffix.

Reference: architecture.md §7. Decision log §13: dcm2niix invoked
directly, no wrapper layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from ... import schema
from ...classifier.dcm2niix_bidsguess import find_dcm2niix, run_dcm2niix
from ...util.paths import long_path
from ...util.staging import stage_files, staging_folder
from ..types import ConvertResult, ConvertTask

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1800
_STDERR_TAIL_BYTES = 4096


class Dcm2niixDirect:
    """Per-series dcm2niix backend.

    Constructed once by the registry; called once per task. The binary
    path is resolved at construction (so a missing ``dcm2niix`` fails
    fast, before any subject-level work begins).
    """

    name = "dcm2niix_direct"

    def __init__(self, dcm2niix_bin: Optional[Path] = None) -> None:
        self._bin = Path(dcm2niix_bin) if dcm2niix_bin else find_dcm2niix()

    @property
    def binary(self) -> Path:
        return self._bin

    def can_handle(self, task: ConvertTask) -> bool:
        if task.suffix == "physio":
            # Physio rows route to ``PhysioDcmBackend``
            # (converter.physio_cmrr). dcm2niix can't produce ``_physio.tsv.gz``.
            return False
        if task.datatype in {"eeg", "meg", "ieeg", "nirs"}:
            # EEG/MEG/iEEG/NIRS rows route to ``MneBidsBackend``.
            # dcm2niix doesn't read raw electrophysiology formats.
            return False
        return bool(task.source_files) and bool(task.basename)

    def convert(self, task: ConvertTask, staging_dir: Path) -> ConvertResult:
        """Run dcm2niix for one series; report what landed in staging.

        Failure modes captured in ``ConvertResult.error``:

        * ``"empty staging"`` — none of ``task.source_files`` exist.
        * ``"dcm2niix failed: rc=<n>"`` — dcm2niix exited non-zero.
        * ``"missing expected output: <ext>"`` — required extension
          (``.nii.gz`` or ``.json``) absent after a 0-exit run.

        Multi-output cases (fmap, multi-echo) are *not* failures here —
        the extra files are reported in ``staged_files`` and the
        downstream fmap fixup deals with the suffix mapping.
        """

        t0 = time.monotonic()
        try:
            return self._convert_inner(task, staging_dir, t0)
        except subprocess.TimeoutExpired as exc:
            return ConvertResult(
                task=task, success=False,
                error=f"dcm2niix timed out after {exc.timeout}s",
                duration_s=time.monotonic() - t0,
            )
        except Exception as exc:
            log.exception("dcm2niix_direct: unexpected error for %s", task.basename)
            return ConvertResult(
                task=task, success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
            )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _convert_inner(
        self, task: ConvertTask, staging_dir: Path, t0: float,
    ) -> ConvertResult:
        # Per-series symlink fan-out. Sibling staging dirs (one per
        # series) live alongside the datatype tree so nothing collides
        # when multiple tasks for the same subject run in parallel.
        # The dir name is a short hash of ``task.series_uid`` rather
        # than the UID itself: (a) fmap rows carry a ``|``-joined pair
        # of UIDs which is an illegal Windows path character, and
        # (b) a raw UID is ~64 chars (130 for a pair), which pushes
        # deep Windows BIDS trees past the 260-char ``MAX_PATH`` limit
        # and trips dcm2niix with ``rc=2``.
        # The basename identifies a row that has no UID; by this point it is
        # already known to be unique, because conversion refuses to start
        # otherwise.
        # On Windows, a staging tree on a network share stages the DICOMs in a
        # local temporary folder instead (util.staging explains why).
        dicoms_dir = staging_folder(staging_dir / _safe_dicoms_dirname(
            task.series_uid, task.basename,
        ))
        try:
            n_staged = stage_files(task.source_files, dicoms_dir)
            if n_staged == 0:
                return ConvertResult(
                    task=task, success=False,
                    error="empty staging: no source DICOMs were accessible",
                    duration_s=time.monotonic() - t0,
                )

            output_dir = staging_dir / task.datatype
            output_dir.mkdir(parents=True, exist_ok=True)

            proc = _run_dcm2niix(
                self._bin, dicoms_dir, output_dir, task.basename,
                compress=task.compress,
            )
        finally:
            # Best-effort cleanup of the per-series staging; not fatal.
            shutil.rmtree(long_path(dicoms_dir), ignore_errors=True)
        stderr_tail = (proc.stderr or "")[-_STDERR_TAIL_BYTES:]

        if proc.returncode != 0:
            return ConvertResult(
                task=task, success=False,
                error=f"dcm2niix failed: rc={proc.returncode}",
                dcm2niix_returncode=proc.returncode,
                dcm2niix_stderr_tail=stderr_tail,
                duration_s=time.monotonic() - t0,
            )

        staged = _collect_outputs(output_dir, task.basename)
        staged = _drop_disallowed_extensions(staged, task)
        staged = _name_phase_part(staged, task.basename)

        if task.skip_residuals:
            staged, dropped = _partition_residuals(staged, task.basename)
            for p in dropped:
                try:
                    p.unlink()
                except OSError:
                    pass  # best-effort; a leftover residual is harmless
            if dropped:
                log.info(
                    "skip_residuals: dropped %d dcm2niix residual output(s) "
                    "for %s: %s",
                    len(dropped), task.basename,
                    ", ".join(p.name for p in dropped),
                )
        # After the residuals: a name the echoes are given here no longer
        # starts with the basename dcm2niix was handed.
        staged = _name_echoes(staged, task)

        missing = _missing_expected(staged, task.expected_outputs, task.basename, output_dir)
        if missing:
            return ConvertResult(
                task=task, staged_files=tuple(staged), success=False,
                error=f"missing expected output(s): {', '.join(missing)}",
                dcm2niix_returncode=proc.returncode,
                dcm2niix_stderr_tail=stderr_tail,
                duration_s=time.monotonic() - t0,
            )

        return ConvertResult(
            task=task, staged_files=tuple(staged), success=True,
            dcm2niix_returncode=proc.returncode,
            dcm2niix_stderr_tail=stderr_tail,
            duration_s=time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (picklable for joblib workers; reused by tests)
# ---------------------------------------------------------------------------


def _safe_dicoms_dirname(series_uid: str, fallback: str = "") -> str:
    """Return the per-series staging dir name as ``_dicoms_<hash>``.

    The raw ``series_uid`` is unsuitable as a directory component:

    * fmap rows collapse a magnitude/phase pair into one row whose
      ``series_uid`` is the two UIDs joined by ``|`` — illegal in
      Windows path components.
    * A single UID is ~64 chars and a pair ~130; combined with deep
      Windows BIDS trees this pushes the full staging path past the
      260-char ``MAX_PATH`` limit and dcm2niix exits ``rc=2``.

    A 12-hex SHA-1 prefix is unique in practice (one subject's batch
    has on the order of 10² series — collision probability ≈ 2⁻⁴⁰)
    and keeps the dir at 20 chars including the ``_dicoms_`` prefix.

    ``fallback`` names the row when it has no UID to hash. Only DICOM carries a
    SeriesInstanceUID: every ECAT, EEG and MEG row has an EMPTY one, and hashing
    the empty string gives ``da39a3ee5e6b`` for all of them, so two such rows in
    one subject shared a staging directory and the second failed with
    FileExistsError. Three ECAT phantoms hit exactly that.
    """
    digest = hashlib.sha1(
        (series_uid or fallback or "").encode("utf-8")
    ).hexdigest()[:12]
    return f"_dicoms_{digest}"


def _run_dcm2niix(
    binary: Path,
    dicom_dir: Path,
    output_dir: Path,
    basename: str,
    *,
    compress: bool = True,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    """Invoke dcm2niix to write ``<basename>.nii.gz`` etc. into ``output_dir``.

    Flags: ``-b y`` (write JSON), ``-ba n`` (don't anonymise — keeps
    SeriesInstanceUID for provenance / dedup), ``-z y`` (gzip; ``-z n``
    with ``compress=False``, for ``.nii``), ``-f <basename>`` (we control
    the name; do not use ``%j`` here), and ``--terse`` is intentionally NOT
    passed so debugging stderr survives.
    """
    cmd = [
        str(binary),
        "-b", "y",
        "-ba", "n",
        "-z", "y" if compress else "n",
        "-o", str(output_dir),
        "-f", basename,
        str(dicom_dir),
    ]
    return run_dcm2niix(cmd, capture_output=True, text=True, timeout=timeout)


def _name_phase_part(staged: list[Path], basename: str) -> list[Path]:
    """A series the scan named ``part-phase`` IS the phase image, and
    dcm2niix still appends its ``_ph`` token (``..._part-phase_bold_ph``,
    a name BIDS does not define). Drop the token so the file lands at the
    planned name, and say the image's ``Units``, which BIDS requires of
    every ``part-phase`` file and dcm2niix does not write. Anything else is
    left as it is."""
    if "_part-phase_" not in f"_{basename}_":
        return staged
    out = []
    for path in staged:
        ext = _matching_ext(path.name, _KNOWN_OUTPUT_EXTS)
        if ext and path.name == f"{basename}_ph{ext}":
            target = path.with_name(basename + ext)
            if not target.exists():
                path.rename(target)
                path = target
        out.append(path)
    image = next((p for p in out if p.name in (f"{basename}.nii.gz", f"{basename}.nii")), None)
    sidecar = next((p for p in out if p.name == f"{basename}.json"), None)
    if image is not None and sidecar is not None:
        _write_phase_units(sidecar, image)
    return sorted(out)


_ECHO_TAIL_RE = re.compile(r"^_e(\d+)(_ph)?$")


def echo_basename(basename: str, datatype: str, suffix: str, echo: int,
                  part: str = "") -> Optional[str]:
    """The name of echo ``echo`` of the recording ``basename``.

    ``echo-<echo>`` (and ``part-<part>`` when given) set on the name's
    entities and placed where the standard orders them, so
    ``sub-01_acq-fa35_echo-1_part-mag_MEGRE`` gives
    ``sub-01_acq-fa35_echo-3_part-mag_MEGRE`` for echo 3. ``None`` where the
    suffix's rule has no ``echo``, and for fieldmaps, whose echoes
    ``fixups/fieldmaps`` names (``magnitude1/2``, ``phasediff``). The
    conversion names each file dcm2niix splits off by it, and the BIDS
    preview predicts those names by it.
    """
    if datatype == "fmap":
        return None
    try:
        allowed = set(schema.allowed_entities(datatype, suffix))
    except Exception:  # noqa: BLE001 - an unknown suffix is left as dcm2niix wrote it
        return None
    if "echo" not in allowed:
        return None
    entities = dict(schema.name_entities(f"{basename}.x"))
    entities["echo"] = str(int(echo))
    if part and "part" in allowed:
        entities["part"] = part
    return "_".join(f"{k}-{entities[k]}" for k in schema.entity_keys()
                    if entities.get(k)) + f"_{suffix}"


def _name_echoes(staged: list[Path], task) -> list[Path]:
    """Name each echo dcm2niix split off a series with the ``echo`` entity.

    A series acquired at several echo times (a multi-echo GRE, a multi-echo
    BOLD or MPRAGE) is ONE inventory row and comes out of dcm2niix as one file
    per echo, ``<basename>_e1``, ``_e2``, ... (``_e2_ph`` for the phase of a
    series that holds magnitude and phase). Those tails are dcm2niix's, not
    BIDS's: written as they are, every echo of a MEGRE series kept the row's
    ``echo-1`` and gained an ``_e2`` after its suffix, a name the standard
    does not define (96 validation errors on one QSM session).

    Here each tail becomes the entity it stands for (:func:`echo_basename`):
    the echo's number (the sidecar's ``EchoNumber``, else the tail's) in
    ``echo``, and ``_ph`` in ``part-phase`` (with the other files of the
    series then ``part-mag``). Only for a suffix whose rule allows ``echo``;
    fieldmaps are left to ``fixups/fieldmaps``, which maps the same tails to
    ``magnitude1/2`` and ``phasediff``. A file whose new name is already
    taken keeps dcm2niix's, so nothing is overwritten.
    """
    if not staged or echo_basename(task.basename, task.datatype, task.suffix, 1) is None:
        return staged
    basename = task.basename
    found = []
    for path in staged:
        ext = _matching_ext(path.name, _KNOWN_OUTPUT_EXTS)
        if not ext:
            continue
        stem = path.name[: -len(ext)]
        if not stem.startswith(basename):
            continue
        match = _ECHO_TAIL_RE.match(stem[len(basename):])
        if match:
            found.append((path, ext, int(match.group(1)), bool(match.group(2))))
    if not found:
        return staged

    has_part = "part" in schema.name_entities(f"{basename}.x")
    any_phase = any(phase for *_rest, phase in found)

    def numbered(echo: int, phase: bool) -> str:
        part = "phase" if phase else ("mag" if any_phase and not has_part else "")
        return echo_basename(basename, task.datatype, task.suffix, echo, part) or basename

    # The echo number of each (stem): the sidecar's EchoNumber where it says one.
    echo_of: dict[str, int] = {}
    for path, ext, echo, _phase in found:
        if ext == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("EchoNumber")
                if isinstance(value, (int, float)) and int(value) == value and value > 0:
                    echo_of[path.name[: -len(ext)]] = int(value)
            except (OSError, ValueError):
                pass

    taken = {p.name for p in staged}
    out = [p for p in staged if p not in {f[0] for f in found}]
    renamed: list[Path] = []
    for path, ext, echo, phase in found:
        stem = path.name[: -len(ext)]
        target = path.with_name(numbered(echo_of.get(stem, echo), phase) + ext)
        if target.name in taken or target.exists():
            log.warning("echo naming: %s would land on %s, which is taken; kept as written",
                        path.name, target.name)
            out.append(path)
            continue
        path.rename(target)
        taken.discard(path.name)
        taken.add(target.name)
        out.append(target)
        renamed.append(target)
    # A phase image says its units, as BIDS requires of every part-phase file.
    for image in renamed:
        if "_part-phase_" not in f"_{image.name}" or not image.name.endswith((".nii.gz", ".nii")):
            continue
        sidecar = image.with_name(image.name[: -len(".nii.gz" if image.name.endswith(".gz") else ".nii")]
                                  + ".json")
        if sidecar.exists():
            _write_phase_units(sidecar, image)
    if renamed:
        log.info("echo naming: %d file(s) of %s named by echo", len(renamed), basename)
    return sorted(out)


def _phase_units(image: Path) -> str:
    """``"rad"`` when the phase values lie within one turn, else
    ``"arbitrary"`` (Siemens stores raw phase as integers in +-4096). Reads
    the first volume only."""
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(image))
        data = img.dataobj[..., 0] if len(img.shape) > 3 else img.dataobj[...]
        values = np.asarray(data, dtype=float)
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    except Exception:  # noqa: BLE001 - a unit is a courtesy, never a failure
        return "arbitrary"
    return "rad" if -3.1416 * 1.01 <= low and high <= 3.1416 * 1.01 else "arbitrary"


def _write_phase_units(sidecar: Path, image: Path) -> None:
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(meta, dict) or meta.get("Units"):
        return
    meta["Units"] = _phase_units(image)
    sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _drop_disallowed_extensions(staged: list[Path], task) -> list[Path]:
    """Remove files whose extension the schema forbids for this suffix.

    dcm2niix writes ``.bval`` and ``.bvec`` for every series it recognises as
    diffusion, INCLUDING the scanner derivatives: FA, trace, colFA, ADC.
    Those are legitimate raw ``dwi/`` suffixes in BIDS 1.11, but they are
    scalar maps, not diffusion-weighted volumes, so the standard allows them
    only ``.nii[.gz]`` and ``.json``. A gradient table beside an FA map
    describes nothing: the directions were consumed to compute it.

    Left in place they are a validator error on every diffusion derivative a
    Siemens scanner produces, which is what this lab's own data was doing.

    **The schema decides, not a list.** ``list_extensions(datatype, suffix)``
    says ``.bvec`` and ``.bval`` belong to ``dwi/dwi`` and to nothing else,
    so this needs no maintenance when BIDS adds a suffix. A datatype or
    suffix the schema has no rule for is left completely alone: refusing to
    stage a file because the schema is silent about it would be the fixup
    deciding it knows better than the standard.
    """
    datatype = str(getattr(task, "datatype", "") or "")
    suffix = str(getattr(task, "suffix", "") or "")
    if not datatype or not suffix:
        return staged
    try:
        allowed = set(schema.list_extensions(datatype, suffix))
    except Exception:  # noqa: BLE001 - a staging pass must not fail on this
        return staged
    if not allowed:
        return staged

    keep: list[Path] = []
    for path in staged:
        ext = _matching_ext(path.name, allowed)
        if ext is None and _matching_ext(path.name, _KNOWN_OUTPUT_EXTS) is not None:
            # An extension dcm2niix writes, that the schema does not allow
            # for THIS suffix. Dropped, and said out loud: a file silently
            # vanishing is worse than one that should not be there.
            log.info(
                "%s: %s is not an allowed extension for %s/%s; not staged",
                task.basename, path.name, datatype, suffix,
            )
            try:
                path.unlink()
            except OSError:
                pass  # best-effort; an unstaged leftover is harmless
            continue
        keep.append(path)
    return keep


def _matching_ext(name: str, extensions) -> Optional[str]:
    """The longest entry of ``extensions`` that ``name`` ends with."""
    best = None
    for ext in extensions:
        if name.endswith(ext) and (best is None or len(ext) > len(best)):
            best = ext
    return best


def _collect_outputs(output_dir: Path, basename: str) -> list[Path]:
    """Return every file dcm2niix produced for this basename, sorted.

    dcm2niix occasionally writes additional siblings (``_e1``, ``_e2``,
    ``_ph`` for fmap; ``.bval`` / ``.bvec`` for DWI). Glob ``<basename>*``
    catches them all; deterministic sorting makes the result stable.
    """
    return sorted(output_dir.glob(f"{basename}*"))


# Extensions dcm2niix may write for one task. Stripped before the residual
# match so the marker is compared against the bare stem (longest first so
# ``.nii.gz`` wins over ``.nii``).
_KNOWN_OUTPUT_EXTS: tuple[str, ...] = (
    ".nii.gz", ".tsv.gz", ".nii", ".json", ".bval", ".bvec",
)

# A "residual" is an EXTRA output dcm2niix splits off a single input series:
# a derived / secondary volume that is not a real acquired image and has no
# valid BIDS suffix. dcm2niix names them by appending a marker to the
# user-supplied ``-f`` basename (the PRIMARY image keeps the exact basename):
#   * a single glued collision letter   ``..._bold`` -> ``..._bolda``
#   * ``_Eq_<n>``   an equidistant-resliced duplicate of a variable-spacing run
#   * ``_ROI<n>``   a Siemens on-scanner MoCo / ROI secondary series
#   * ``_i<instance>``  a raw per-instance dump (5+ digits)
# Legitimate multi-output is deliberately NOT matched here: fmap/multi-echo
# (``_e1`` / ``_e2`` / ``_ph`` / ``_e2_ph``), complex parts
# (``_real`` / ``_imaginary``), and DWI ``.bval`` / ``.bvec`` all survive
# (their tails start with ``_e`` / ``_real`` / ``_imag`` or strip to an empty
# tail), so the fmap and DWI fixups still see everything they expect.
_RESIDUAL_TAIL_RE = re.compile(r"^(?:[a-z]|_Eq_\d+|_ROI\d*|_i\d{5,})$")


def _strip_known_ext(name: str) -> str:
    for ext in _KNOWN_OUTPUT_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _is_residual_output(path: Path, basename: str) -> bool:
    """True if ``path`` is a dcm2niix residual/secondary output for ``basename``.

    The PRIMARY output (stem == ``basename``) and every legitimate sibling
    return False; only the appended-marker duplicates documented in
    :data:`_RESIDUAL_TAIL_RE` return True.
    """
    stem = _strip_known_ext(path.name)
    if not stem.startswith(basename):
        return False
    tail = stem[len(basename):]
    return bool(_RESIDUAL_TAIL_RE.match(tail))


def _partition_residuals(
    staged: list[Path], basename: str,
) -> tuple[list[Path], list[Path]]:
    """Split ``staged`` into (keep, residual) by :func:`_is_residual_output`."""
    keep: list[Path] = []
    residual: list[Path] = []
    for p in staged:
        (residual if _is_residual_output(p, basename) else keep).append(p)
    return keep, residual


def _missing_expected(
    staged: list[Path],
    expected_exts: tuple[str, ...],
    basename: str,
    output_dir: Path,
) -> list[str]:
    """Return any required extension that didn't materialise.

    For fmap multi-output, the *exact* ``<basename>.nii.gz`` may be absent
    while ``<basename>_e1.nii.gz`` etc. exist — that's fine, the fixup
    will rename them. Only flag missing if NEITHER the exact nor any
    suffixed sibling matches the extension.
    """
    missing: list[str] = []
    for ext in expected_exts:
        exact = output_dir / f"{basename}{ext}"
        if exact.exists():
            continue
        # Any sibling with this extension counts (e.g. <basename>_e1.nii.gz).
        if any(p.name.endswith(ext) for p in staged):
            continue
        missing.append(ext)
    return missing


__all__ = ["Dcm2niixDirect", "echo_basename"]
