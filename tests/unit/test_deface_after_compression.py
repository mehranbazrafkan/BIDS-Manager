"""The undefaced original is still found after an image changes form.

``deface.compare`` finds an image's undefaced copy by its path. Compressing or
decompressing changes the path (``x.nii.gz`` to ``x.nii``) and keeps a copy of
its own for Undo, which is the SAME image, defaced or not. So: the original is
looked up under every name the image has had, the copy a change of form kept
never counts as one, and restoring re-encodes when the forms differ.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from bidsmgr.deface import compare, revert, status
from bidsmgr.deface.engines import ALLINEATE, TEMPLATE
from bidsmgr.editor import compression
from bidsmgr.project.operations import begin_operation

GZ = "sub-01/anat/sub-01_T1w.nii.gz"
NII = "sub-01/anat/sub-01_T1w.nii"


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "t", "BIDSVersion": "1.11.1"}))
    shutil.copyfile(TEMPLATE, root / GZ)
    return root


def _deface_in_the_editor(root: Path) -> np.ndarray:
    """What defacing from Tools leaves: the undefaced copy in the history,
    a different image in the dataset, the sidecar saying so."""
    undefaced = np.asarray(nib.load(str(root / GZ)).dataobj)
    img = nib.load(str(root / GZ))
    defaced = np.zeros(img.shape, dtype=np.int16)
    out = root.parent / "defaced.nii.gz"
    nib.save(nib.Nifti1Image(defaced, img.affine), str(out))
    with begin_operation(root, "Deface 1 image (allineate)") as op:
        op.replace_from(root / GZ, out)
        op.write_json(status.sidecar_for(root / GZ), status.record({}, ALLINEATE))
    return undefaced


def _decompress(root: Path) -> None:
    outcome = compression.apply_compression(
        root, compression.plan_compression(root, compress=False))
    assert outcome.ok


def test_the_original_follows_a_decompressed_image(tmp_path):
    root = _dataset(tmp_path)
    _deface_in_the_editor(root)
    _decompress(root)
    found = compare.original_for(root, NII)
    assert found is not None and found.source == "history"
    assert found.label.startswith("Deface"), "the defacing's copy, not the decompression's"
    assert compare.comparable(root) == [NII]
    assert compare.names_of(root, NII) == [NII, GZ]
    assert compare.current_name(root, GZ) == NII


def test_the_copy_a_change_of_form_kept_is_not_an_original(tmp_path):
    """Defaced at conversion: there never was an undefaced copy, and the
    decompression's own copy (defaced too) must not pretend to be one."""
    root = _dataset(tmp_path)
    status.sidecar_for(root / GZ).write_text(json.dumps(status.record({}, ALLINEATE)))
    _decompress(root)
    assert compare.original_for(root, NII) is None
    assert compare.comparable(root) == []
    assert "conversion" in compare.explain_missing(root, NII)


def test_a_sourcedata_mirror_under_the_earlier_name_is_found(tmp_path):
    root = _dataset(tmp_path)
    mirror = root / "sourcedata" / GZ
    mirror.parent.mkdir(parents=True)
    shutil.copyfile(TEMPLATE, mirror)
    _decompress(root)
    found = compare.original_for(root, NII)
    assert found is not None and found.path == mirror
    assert NII in compare.comparable(root)


def test_putting_the_face_back_writes_the_images_own_form(tmp_path):
    root = _dataset(tmp_path)
    undefaced = _deface_in_the_editor(root)
    _decompress(root)
    items = revert.revertable(root)
    assert [i.relative for i in items] == [NII]
    outcome = revert.revert_dataset(root, items=items)
    assert outcome.ok, outcome.failed
    image = root / NII
    with open(image, "rb") as fh:
        assert fh.read(2) != b"\x1f\x8b", "a .nii holds uncompressed data"
    assert np.array_equal(np.asarray(nib.load(str(image)).dataobj), undefaced)
    assert not (root / GZ).exists()


def test_an_image_that_kept_its_form_is_restored_as_before(tmp_path):
    root = _dataset(tmp_path)
    undefaced = _deface_in_the_editor(root)
    outcome = revert.revert_dataset(root)
    assert outcome.ok
    with gzip.open(root / GZ, "rb") as fh:
        assert fh.read(4)
    assert np.array_equal(np.asarray(nib.load(str(root / GZ)).dataobj), undefaced)
