"""Each echo dcm2niix splits off a series gets its own BIDS name.

dcm2niix writes a multi-echo series (a MEGRE for QSM, a multi-echo BOLD) as
one file per echo, ``<basename>_e1``, ``_e2``, ... and ``_e2_ph`` for the
phase of a series holding both. Written as they were, a MEGRE series kept the
row's ``echo-1`` on every echo with an ``_e2`` after the suffix, a name the
standard does not define: 96 validation errors on one QSM session (Tina
Schmitt's MEGRE data, 2026-10). The backend now turns each tail into the
entity it stands for, where the schema allows ``echo``, and nowhere else.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.converter.backends import dcm2niix_direct as backend_mod
from bidsmgr.converter.backends.dcm2niix_direct import (
    Dcm2niixDirect,
    _name_echoes,
    echo_basename,
)
from bidsmgr.converter.types import ConvertTask


def _task(tmp_path: Path, basename: str, datatype: str, suffix: str, *,
          compress: bool = True) -> ConvertTask:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    dcm = src / "IMG0001.dcm"
    dcm.write_bytes(b"DICM")
    image = ".nii.gz" if compress else ".nii"
    return ConvertTask(
        row_id="r1", series_uid="1.2.3", source_dicom_files=(dcm,), dataset="study",
        bids_root=tmp_path / "bids" / "study", subject="01", session=None,
        datatype=datatype, suffix=suffix, entities={"sub": "01"}, basename=basename,
        expected_outputs=(image, ".json"), compress=compress,
    )


def _write(folder: Path, name: str, meta: dict | None = None) -> Path:
    path = folder / name
    if name.endswith(".json"):
        path.write_text(json.dumps(meta or {}))
    else:
        path.write_bytes(b"image")
    return path


# -- the name of one echo -----------------------------------------------------


@pytest.mark.parametrize("basename, datatype, suffix, echo, part, expected", [
    ("sub-01_acq-fa35_echo-1_part-mag_MEGRE", "anat", "MEGRE", 3, "",
     "sub-01_acq-fa35_echo-3_part-mag_MEGRE"),
    ("sub-01_acq-fa35_rec-nd_echo-1_MEGRE", "anat", "MEGRE", 2, "phase",
     "sub-01_acq-fa35_rec-nd_echo-2_part-phase_MEGRE"),
    # echo lands where the standard orders it, not at the end
    ("sub-01_ses-1_task-rest_run-01_bold", "func", "bold", 2, "",
     "sub-01_ses-1_task-rest_run-01_echo-2_bold"),
    ("sub-01_acq-memprage_T1w", "anat", "T1w", 4, "", "sub-01_acq-memprage_echo-4_T1w"),
])
def test_an_echo_is_named_by_the_echo_entity(basename, datatype, suffix, echo, part, expected):
    assert echo_basename(basename, datatype, suffix, echo, part) == expected


@pytest.mark.parametrize("basename, datatype, suffix", [
    ("sub-01_dwi", "dwi", "dwi"),                    # no echo entity in its rule
    ("sub-01_magnitude1", "fmap", "magnitude1"),     # fixups/fieldmaps names these
    ("sub-01_phasediff", "fmap", "phasediff"),
    ("sub-01_nonsense", "anat", "nonsense"),         # no rule at all
])
def test_no_echo_name_where_the_standard_has_none(basename, datatype, suffix):
    assert echo_basename(basename, datatype, suffix, 2) is None


# -- naming the files dcm2niix wrote --------------------------------------------


def test_every_echo_of_a_megre_magnitude_gets_its_own_name(tmp_path):
    base = "sub-01_acq-fa35_echo-1_part-mag_MEGRE"
    task = _task(tmp_path, base, "anat", "MEGRE")
    staged = []
    for k in range(1, 6):
        staged.append(_write(tmp_path, f"{base}_e{k}.nii.gz"))
        staged.append(_write(tmp_path, f"{base}_e{k}.json", {"EchoNumber": k}))
    out = _name_echoes(staged, task)
    assert sorted(p.name for p in out) == sorted(
        f"sub-01_acq-fa35_echo-{k}_part-mag_MEGRE{ext}"
        for k in range(1, 6) for ext in (".nii.gz", ".json"))
    assert all(p.exists() for p in out)
    assert not list(tmp_path.glob("*_e[0-9]*"))


def test_the_sidecars_echo_number_wins_over_the_tail(tmp_path):
    """dcm2niix numbers the files it writes; the header numbers the echoes.
    They agree in practice; where they did not, the header is the fact."""
    base = "sub-01_echo-1_MEGRE"
    task = _task(tmp_path, base, "anat", "MEGRE")
    staged = [_write(tmp_path, f"{base}_e1.nii.gz"), _write(tmp_path, f"{base}_e1.json", {"EchoNumber": 3})]
    out = _name_echoes(staged, task)
    assert sorted(p.name for p in out) == ["sub-01_echo-3_MEGRE.json", "sub-01_echo-3_MEGRE.nii.gz"]


def test_a_series_holding_magnitude_and_phase_gets_both_parts(tmp_path):
    """``_e2_ph`` is the phase of echo 2; the files without ``_ph`` are then
    the magnitude, and both say so. The phase sidecar says its units, which
    BIDS requires of every part-phase file."""
    base = "sub-01_echo-1_MEGRE"
    task = _task(tmp_path, base, "anat", "MEGRE")
    import nibabel as nib

    staged = []
    for k in (1, 2):
        for tail in ("", "_ph"):
            img = tmp_path / f"{base}_e{k}{tail}.nii.gz"
            data = np.full((2, 2, 2), 4000 if tail else 10, dtype=np.int16)
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(img))
            staged.append(img)
            staged.append(_write(tmp_path, f"{base}_e{k}{tail}.json", {"EchoNumber": k}))
    out = {p.name for p in _name_echoes(staged, task)}
    assert out == {f"sub-01_echo-{k}_part-{part}_MEGRE{ext}"
                   for k in (1, 2) for part in ("mag", "phase") for ext in (".nii.gz", ".json")}
    phase = json.loads((tmp_path / "sub-01_echo-2_part-phase_MEGRE.json").read_text())
    assert phase["Units"] == "arbitrary"
    assert "Units" not in json.loads((tmp_path / "sub-01_echo-2_part-mag_MEGRE.json").read_text())


def test_uncompressed_echoes_are_named_too(tmp_path):
    base = "sub-01_task-rest_bold"
    task = _task(tmp_path, base, "func", "bold", compress=False)
    staged = [_write(tmp_path, f"{base}_e{k}{ext}", {"EchoNumber": k} if ext == ".json" else None)
              for k in (1, 2, 3) for ext in (".nii", ".json")]
    out = sorted(p.name for p in _name_echoes(staged, task))
    assert out == sorted(f"sub-01_task-rest_echo-{k}_bold{ext}"
                         for k in (1, 2, 3) for ext in (".nii", ".json"))


def test_a_name_already_taken_keeps_what_dcm2niix_wrote(tmp_path):
    base = "sub-01_echo-1_MEGRE"
    task = _task(tmp_path, base, "anat", "MEGRE")
    taken = _write(tmp_path, "sub-01_echo-2_MEGRE.nii.gz")
    staged = [taken, _write(tmp_path, f"{base}_e2.nii.gz")]
    out = {p.name for p in _name_echoes(staged, task)}
    assert out == {"sub-01_echo-2_MEGRE.nii.gz", f"{base}_e2.nii.gz"}
    assert taken.read_bytes() == b"image"


@pytest.mark.parametrize("basename, datatype, suffix", [
    ("sub-01_magnitude1", "fmap", "magnitude1"),
    ("sub-01_dwi", "dwi", "dwi"),
])
def test_files_the_standard_cannot_name_by_echo_are_left_alone(tmp_path, basename, datatype, suffix):
    task = _task(tmp_path, basename, datatype, suffix)
    staged = [_write(tmp_path, f"{basename}_e1.nii.gz"), _write(tmp_path, f"{basename}_e2.nii.gz")]
    assert _name_echoes(list(staged), task) == staged
    assert all(p.exists() for p in staged)


def test_a_single_echo_series_is_not_touched(tmp_path):
    base = "sub-01_T1w"
    task = _task(tmp_path, base, "anat", "T1w")
    staged = [_write(tmp_path, f"{base}.nii.gz"), _write(tmp_path, f"{base}.json")]
    assert _name_echoes(list(staged), task) == staged


# -- through the backend ---------------------------------------------------------


def _fake_run(tails: list[str], seen: list):
    def fake(cmd, *a, **kw):
        seen.append(list(cmd))
        out = Path(cmd[cmd.index("-o") + 1])
        base = cmd[cmd.index("-f") + 1]
        ext = ".nii.gz" if cmd[cmd.index("-z") + 1] == "y" else ".nii"
        for k, tail in enumerate(tails, start=1):
            (out / f"{base}{tail}{ext}").write_bytes(b"image")
            (out / f"{base}{tail}.json").write_text(json.dumps({"EchoNumber": k}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return fake


@pytest.mark.parametrize("compress", [True, False])
def test_a_multi_echo_conversion_lands_at_bids_names(tmp_path, monkeypatch, compress):
    seen: list = []
    monkeypatch.setattr(backend_mod, "run_dcm2niix", _fake_run(["_e1", "_e2"], seen))
    binary = tmp_path / "dcm2niix"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    task = _task(tmp_path, "sub-01_acq-fa14_echo-1_part-mag_MEGRE", "anat", "MEGRE",
                 compress=compress)
    result = Dcm2niixDirect(dcm2niix_bin=binary).convert(task, tmp_path / "stage")
    assert result.success, result.error
    ext = ".nii.gz" if compress else ".nii"
    assert sorted(p.name for p in result.staged_files) == sorted(
        f"sub-01_acq-fa14_echo-{k}_part-mag_MEGRE{e}" for k in (1, 2) for e in (ext, ".json"))
    cmd = seen[0]
    assert cmd[cmd.index("-z") + 1] == ("y" if compress else "n")
