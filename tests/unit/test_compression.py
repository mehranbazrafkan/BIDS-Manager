"""Compressing and decompressing a dataset's images, and what names them.

The image is the same either way; what can go wrong is everything AROUND it:
a scans table still listing ``x.nii.gz`` after it became ``x.nii``, a
fieldmap's IntendedFor pointing at nothing, a saved view that no longer opens,
a half-done run, an Undo that cannot put it back. Each is pinned here.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bidsmgr.cli import compress as cli
from bidsmgr.editor import compression as engine
from bidsmgr.project import operations as ops
from bidsmgr.util.cancel import OperationCancelled

T1 = "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"
BOLD = "sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz"
FMAP = "sub-01/ses-1/fmap/sub-01_ses-1_dir-AP_epi.nii.gz"
T1_02 = "sub-02/anat/sub-02_T1w.nii.gz"


def _image(path: Path, seed: int, shape=(6, 5, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).integers(0, 1000, size=shape).astype(np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    path.with_name(path.name.split(".")[0] + ".json").write_text(json.dumps({"RepetitionTime": 2.0}))


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset_description.json").write_text(json.dumps({"Name": "t", "BIDSVersion": "1.11.1"}))
    for i, rel in enumerate((T1, BOLD, FMAP, T1_02)):
        _image(root / rel, i, (6, 5, 4, 3) if "bold" in rel else (6, 5, 4))
    # A fieldmap names its run in both spellings BIDS uses.
    sidecar = root / FMAP.replace(".nii.gz", ".json")
    sidecar.write_text(json.dumps({
        "IntendedFor": ["bids::" + BOLD, "ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz"],
        "PhaseEncodingDirection": "j-"}))
    (root / "sub-01/ses-1/sub-01_ses-1_scans.tsv").write_text(
        "filename\tacq_time\n"
        "anat/sub-01_ses-1_T1w.nii.gz\t2026-01-01T10:00:00\n"
        "func/sub-01_ses-1_task-rest_bold.nii.gz\t2026-01-01T10:10:00\n"
        "fmap/sub-01_ses-1_dir-AP_epi.nii.gz\t2026-01-01T10:09:00\n")
    (root / "sub-02/sub-02_scans.tsv").write_text("filename\tacq_time\nanat/sub-02_T1w.nii.gz\tn/a\n")
    scenes = root / ".bidsmgr/viz/scenes"
    scenes.mkdir(parents=True)
    (scenes / "check.json").write_text(json.dumps({
        "schema": 1, "name": "check", "base": T1, "overlays": [{"path": T1_02}], "view": {}}))
    (root / ".bidsignore").write_text("extra/\n/" + T1_02 + "\n")
    # Images under derivatives/ belong to their pipeline.
    _image(root / "derivatives/pipe/sub-01/anat/sub-01_desc-brain_T1w.nii.gz", 9)
    return root


def _data(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def _snapshot(root: Path) -> dict[str, bytes]:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and not rel.startswith(".bidsmgr/editor"):
            out[rel] = p.read_bytes()
    return out


def _nii(rel: str) -> str:
    return rel[: -len(".gz")]


# -- planning -------------------------------------------------------------------


def test_the_plan_lists_every_image_and_what_names_it(tmp_path):
    root = _dataset(tmp_path)
    plan = engine.plan_compression(root, compress=False)
    assert [c.rel for c in plan.changes] == sorted([T1, BOLD, FMAP, T1_02])
    assert all(c.target_rel == _nii(c.rel) for c in plan.changes)
    assert all(c.size_after and c.size_after > c.size for c in plan.changes)
    refs = {(r.rel, r.what): set(r.images) for r in plan.references}
    assert refs[("sub-01/ses-1/fmap/sub-01_ses-1_dir-AP_epi.json", "IntendedFor")] == {BOLD}
    assert refs[("sub-01/ses-1/sub-01_ses-1_scans.tsv", "scans.tsv filename")] == {T1, BOLD, FMAP}
    assert refs[(".bidsmgr/viz/scenes/check.json", "saved scene")] == {T1, T1_02}
    assert refs[(".bidsignore", ".bidsignore")] == {T1_02}
    assert not any("derivatives" in c.rel for c in plan.changes)
    assert plan.free_bytes is not None and not plan.short_of_space()


def test_images_already_in_the_form_and_ones_that_cannot_change(tmp_path):
    root = _dataset(tmp_path)
    plain = root / _nii(T1_02)
    with gzip.open(root / T1_02, "rb") as fh:
        plain.write_bytes(fh.read())               # both forms of one image
    misnamed = root / "sub-02/anat/sub-02_FLAIR.nii.gz"
    misnamed.write_bytes(plain.read_bytes())      # named .nii.gz, not compressed
    stray = root / "sub-02/anat/sub-02_PDw.nii.gz"
    with gzip.open(stray, "wb") as fh:
        fh.write(b"not an image at all")
    plan = engine.plan_compression(root, compress=False)
    why = dict(plan.skipped)
    assert "here too" in why[T1_02]
    assert why["sub-02/anat/sub-02_FLAIR.nii.gz"] == engine.SKIP_NOT_GZIP
    assert why["sub-02/anat/sub-02_PDw.nii.gz"] == engine.SKIP_NOT_NIFTI
    assert _nii(T1_02) in plan.already
    assert T1_02 not in [c.rel for c in plan.changes]


def test_a_scope_narrows_the_plan_and_its_references(tmp_path):
    root = _dataset(tmp_path)
    plan = engine.plan_compression(root, compress=False, targets=[root / "sub-01/ses-1/anat"])
    assert [c.rel for c in plan.changes] == [T1]
    assert {r.what for r in plan.references} == {"scans.tsv filename", "saved scene"}


def test_short_of_space_is_known_before_anything_is_written(tmp_path):
    root = _dataset(tmp_path)
    plan = engine.plan_compression(root, compress=False)
    plan.free_bytes = 10
    assert plan.short_of_space()
    assert not plan.short_of_space(only=[])


# -- applying ---------------------------------------------------------------------


def test_decompressing_keeps_the_data_and_every_reference_follows(tmp_path):
    root = _dataset(tmp_path)
    before = {rel: _data(root / rel) for rel in (T1, BOLD, FMAP, T1_02)}
    plan = engine.plan_compression(root, compress=False)
    outcome = engine.apply_compression(root, plan, jobs=2)
    assert outcome.ok and sorted(outcome.changed) == sorted(before)
    assert outcome.references == 5
    for rel, data in before.items():
        assert not (root / rel).exists()
        assert not engine.is_gzip(root / _nii(rel))
        assert np.array_equal(_data(root / _nii(rel)), data)
    fmap = json.loads((root / FMAP.replace(".nii.gz", ".json")).read_text())
    assert fmap["IntendedFor"] == ["bids::" + _nii(BOLD),
                                   "ses-1/func/sub-01_ses-1_task-rest_bold.nii"]
    assert fmap["PhaseEncodingDirection"] == "j-", "the rest of the sidecar is kept"
    scans = (root / "sub-01/ses-1/sub-01_ses-1_scans.tsv").read_text()
    assert ".nii.gz" not in scans and "anat/sub-01_ses-1_T1w.nii\t2026-01-01T10:00:00" in scans
    scene = json.loads((root / ".bidsmgr/viz/scenes/check.json").read_text())
    assert scene["base"] == _nii(T1) and scene["overlays"][0]["path"] == _nii(T1_02)
    assert (root / ".bidsignore").read_text() == "extra/\n/" + _nii(T1_02) + "\n"
    assert (root / "derivatives/pipe/sub-01/anat/sub-01_desc-brain_T1w.nii.gz").exists()
    assert not list(root.rglob("*.part")), "no temporary file is left"


def test_it_is_one_undo_and_one_redo(tmp_path):
    root = _dataset(tmp_path)
    original = _snapshot(root)
    engine.apply_compression(root, engine.plan_compression(root, compress=False))
    after = _snapshot(root)
    step = ops.peek_undo(root)
    assert step is not None and step.label == "Decompress 4 images (.nii.gz to .nii)"
    ops.undo_last(root)
    assert _snapshot(root) == original
    ops.redo_last(root)
    assert _snapshot(root) == after


def test_compressing_back_gives_the_same_bytes_and_a_reproducible_gzip(tmp_path):
    root = _dataset(tmp_path)
    engine.apply_compression(root, engine.plan_compression(root, compress=False))
    plan = engine.plan_compression(root, compress=True)
    assert [c.size_after for c in plan.changes] == [None] * 4
    first = engine.apply_compression(root, plan)
    assert first.ok and first.bytes_after < first.bytes_before
    raw = (root / T1).read_bytes()
    assert raw[3] & 0x08 == 0, "no file name in the gzip header"
    assert raw[4:8] == b"\x00\x00\x00\x00", "no time in the gzip header"
    again = tmp_path / "again.nii.gz"
    with gzip.open(root / T1, "rb") as fh:
        (tmp_path / "x.nii").write_bytes(fh.read())
    engine.transcode(tmp_path / "x.nii", again, compress=True)
    assert again.read_bytes() == raw


def test_only_the_ticked_images_change_and_only_their_references(tmp_path):
    root = _dataset(tmp_path)
    plan = engine.plan_compression(root, compress=False)
    outcome = engine.apply_compression(root, plan, only=[T1])
    assert outcome.changed == [T1]
    assert (root / _nii(T1)).exists() and (root / BOLD).exists()
    scans = (root / "sub-01/ses-1/sub-01_ses-1_scans.tsv").read_text()
    assert "anat/sub-01_ses-1_T1w.nii\t" in scans
    assert "func/sub-01_ses-1_task-rest_bold.nii.gz\t" in scans
    fmap = json.loads((root / FMAP.replace(".nii.gz", ".json")).read_text())
    assert fmap["IntendedFor"][0] == "bids::" + BOLD, "a run left alone keeps its link"


def test_stopping_leaves_the_dataset_as_it_was(tmp_path):
    root = _dataset(tmp_path)
    original = _snapshot(root)
    plan = engine.plan_compression(root, compress=False)

    def stop():
        raise OperationCancelled("stop")

    outcome = engine.apply_compression(root, plan, cancel_check=stop)
    assert outcome.cancelled and not outcome.changed
    assert _snapshot(root) == original
    assert ops.peek_undo(root) is None
    assert not list(root.rglob("*.part"))


def test_a_damaged_image_stops_the_run_with_nothing_changed(tmp_path):
    root = _dataset(tmp_path)
    good = (root / BOLD).read_bytes()
    (root / BOLD).write_bytes(good[: len(good) // 2])      # cut off: header readable, end not
    original = _snapshot(root)
    plan = engine.plan_compression(root, compress=False)
    assert BOLD in [c.rel for c in plan.changes]
    outcome = engine.apply_compression(root, plan)
    assert outcome.failed and not outcome.changed
    assert _snapshot(root) == original
    assert not list(root.rglob("*.part"))


def test_uncompressed_images_compress_in_place_of_themselves(tmp_path):
    root = _dataset(tmp_path)
    engine.apply_compression(root, engine.plan_compression(root, compress=False))
    data = _data(root / _nii(T1))
    outcome = engine.apply_compression(root, engine.plan_compression(root, compress=True))
    assert outcome.ok and engine.is_gzip(root / T1)
    assert np.array_equal(_data(root / T1), data)
    fmap = json.loads((root / FMAP.replace(".nii.gz", ".json")).read_text())
    assert fmap["IntendedFor"][0] == "bids::" + BOLD


# -- the operation steps it is made of --------------------------------------------


def test_set_aside_moves_and_records_what_holds_the_same_data(tmp_path):
    root = _dataset(tmp_path)
    image = root / T1
    before = image.read_bytes()
    with ops.begin_operation(root, "test") as op:
        new = image.with_name("new.part")
        new.write_bytes(b"replacement")
        op.adopt(root / _nii(T1), new)
        op.set_aside(image, same_as=root / _nii(T1))
    record = ops.read_log(root)[-1]
    kinds = {(c["kind"], c["path"], c.get("same_as", "")) for c in record["children"]}
    assert ("delete", T1, _nii(T1)) in kinds
    assert ("create", _nii(T1), "") in kinds
    kept = root / ops.ORIGINALS_DIR / record["op_id"] / T1
    assert kept.read_bytes() == before, "moved, the original bytes kept"
    ops.undo_last(root)
    assert image.read_bytes() == before and not (root / _nii(T1)).exists()


def test_a_failed_operation_puts_the_set_aside_file_back(tmp_path):
    root = _dataset(tmp_path)
    before = (root / T1).read_bytes()
    with pytest.raises(RuntimeError):
        with ops.begin_operation(root, "test") as op:
            op.set_aside(root / T1)
            raise RuntimeError("boom")
    assert (root / T1).read_bytes() == before


# -- the command line ------------------------------------------------------------------


def test_the_command_plans_then_changes(tmp_path, capsys):
    root = _dataset(tmp_path)
    original = _snapshot(root)
    assert cli.main([str(root), "--decompress", "--dry-run", "--no-log-file"]) == 0
    assert _snapshot(root) == original
    assert "4 images to decompress" in capsys.readouterr().out
    assert cli.main([str(root), "--decompress", "--no-log-file", "-j", "1"]) == 0
    assert (root / _nii(T1)).exists()
    assert "Decompressed 4 images" in capsys.readouterr().out
    assert cli.main([str(root), str(root / "sub-02"), "--json", "--dry-run", "--no-log-file"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["direction"] == "compress" and [c["from"] for c in data["changes"]] == [_nii(T1_02)]


def test_the_command_refuses_a_path_outside_the_dataset(tmp_path):
    root = _dataset(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(SystemExit):
        cli.main([str(root), str(outside), "--no-log-file"])


def test_a_copy_of_the_dataset_elsewhere_is_unaffected(tmp_path):
    root = _dataset(tmp_path)
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    engine.apply_compression(root, engine.plan_compression(root, compress=False))
    assert (copy / T1).exists() and not (copy / _nii(T1)).exists()
