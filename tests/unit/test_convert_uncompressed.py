"""``.nii`` instead of ``.nii.gz``, from the setting to the task.

Both are valid BIDS; the conversion writes ``.nii.gz`` unless asked otherwise
(Settings > Convert "NIfTI images", ``bidsmgr-convert --no-compress``). The
choice travels as ``run_convert(compress=)`` -> ``TaskOptions.compress`` ->
``ConvertTask.compress``, which the dcm2niix and ECAT backends read.
"""

from __future__ import annotations


import pandas as pd
import pytest

from bidsmgr.cli import convert as convert_mod
from bidsmgr.converter.types import ConvertTask


def _row(**extra) -> pd.Series:
    row = {"series_uid": "1.2.3", "participant_id": "sub-01", "session": "",
           "datatype": "anat", "bids_guess_suffix": "T1w", "bids_name": "sub-01_T1w",
           "include": "1"}
    row.update(extra)
    return pd.Series(row)


@pytest.mark.parametrize("compress, image", [(True, ".nii.gz"), (False, ".nii")])
def test_an_mri_task_carries_the_choice(tmp_path, compress, image):
    task = convert_mod._row_to_task(
        _row(), tmp_path, {"1.2.3": [str(tmp_path / "a.dcm")]},
        options=convert_mod.TaskOptions(compress=compress))
    assert task is not None and task.compress is compress
    assert image in task.expected_outputs


def test_a_dwi_task_keeps_its_gradient_tables_uncompressed(tmp_path):
    task = convert_mod._row_to_task(
        _row(datatype="dwi", bids_guess_suffix="dwi", bids_name="sub-01_dwi"), tmp_path,
        {"1.2.3": [str(tmp_path / "a.dcm")]}, options=convert_mod.TaskOptions(compress=False))
    assert set(task.expected_outputs) >= {".nii", ".json", ".bval", ".bvec"}


def test_compressed_is_the_default_everywhere():
    assert ConvertTask.model_fields["compress"].default is True
    assert convert_mod.TaskOptions().compress is True
    import inspect

    assert inspect.signature(convert_mod.run_convert).parameters["compress"].default is True


@pytest.mark.parametrize("argv, expected", [([], True), (["--no-compress"], False)])
def test_the_command_line_flag_reaches_the_conversion(tmp_path, monkeypatch, argv, expected):
    seen = {}

    def fake(*args, **kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(convert_mod, "run_convert", fake)
    tsv = tmp_path / "inv.tsv"
    tsv.write_text("subject\n")
    convert_mod.main([str(tsv), str(tmp_path / "out"), "--no-log-file", *argv])
    assert seen["compress"] is expected
