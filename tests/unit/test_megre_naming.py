"""Multi-echo GRE protocols told apart by what differs, not by run.

A QSM session (Tina Schmitt's, 2026-10) holds two MEGRE protocols at 35 and
14 degrees, each reconstructed with and without distortion correction, each
as magnitude and phase: eight series. BIDS has no ``flip`` entity for MEGRE,
so the scan named them ``run-1`` to ``run-4``, and a run means "the same
acquisition repeated", which these are not. ``cli.scan._name_megre_protocols``
gives the flip angle to ``acq`` and the uncorrected reconstruction ``rec-nd``,
and touches no other suffix.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from bidsmgr.classifier.acquisition.features import SeriesFeatures, _distortion_marker, build
from bidsmgr.classifier.types import Classification
from bidsmgr.cli.scan import _flip_label, _name_megre_protocols, _normalize_runs
from bidsmgr.inventory.types import InventoryRow


def _row(uid: str, description: str, *, subject="001", session="") -> InventoryRow:
    return InventoryRow(modality="mri", source=Path("/tmp/x"), subject_hint=subject,
                        session_hint=session, series_description=description, series_uid=uid,
                        acq_time=uid)


def _megre(row: InventoryRow, part: str, *, suffix="MEGRE", classifier="dcm2niix_bidsguess",
           **entities) -> Classification:
    return Classification(row_id=row.row_id, classifier=classifier, datatype="anat", suffix=suffix,
                          candidate_entities={"acquisition": "fl3p2", "echo": "1", "part": part,
                                              **entities}, confidence=0.85)


def _session(**overrides):
    """The eight series of the QSM session, as the scan sees them."""
    series = [
        ("s05", "MEGRE_1p0_T1w_qsm_ND", 35.0, "ND", "mag"),
        ("s06", "MEGRE_1p0_T1w_qsm", 35.0, "DIS2D", "mag"),
        ("s07", "MEGRE_1p0_T1w_qsm_ND", 35.0, "ND", "phase"),
        ("s08", "MEGRE_1p0_T1w_qsm", 35.0, "DIS2D", "phase"),
        ("s09", "QSM_MultiEchoGRE_1p0mm_ND", 14.0, "ND", "mag"),
        ("s10", "QSM_MultiEchoGRE_1p0mm", 14.0, "DIS2D", "mag"),
        ("s11", "QSM_MultiEchoGRE_1p0mm_ND", 14.0, "ND", "phase"),
        ("s12", "QSM_MultiEchoGRE_1p0mm", 14.0, "DIS2D", "phase"),
    ]
    rows, chosen, features = [], {}, {}
    for uid, description, flip, mark, part in series:
        row = _row(uid, description)
        rows.append(row)
        chosen[row.row_id.hex] = _megre(row, part)
        features[uid] = SeriesFeatures(flip_angle=flip, distortion=overrides.get("mark", mark),
                                       description=description).to_dict()
    df = pd.DataFrame({"x": [0]})
    df.attrs["features_by_uid"] = features
    return rows, chosen, df


def _names(rows, chosen) -> dict[str, dict]:
    return {r.series_uid: chosen[r.row_id.hex].candidate_entities for r in rows}


def test_the_qsm_session_gets_flip_angle_and_reconstruction():
    rows, chosen, df = _session()
    chosen = _normalize_runs(rows, _name_megre_protocols(rows, chosen, df))
    names = _names(rows, chosen)
    assert names["s06"] == {"acquisition": "fa35", "echo": "1", "part": "mag"}
    assert names["s05"] == {"acquisition": "fa35", "echo": "1", "part": "mag",
                            "reconstruction": "nd"}
    assert names["s12"] == {"acquisition": "fa14", "echo": "1", "part": "phase"}
    assert names["s11"]["reconstruction"] == "nd"
    assert not any("run" in e for e in names.values()), "eight different series, no repeats"
    assert "flip angle" in chosen[rows[0].row_id.hex].rationale


def test_without_the_header_marker_the_description_decides():
    rows, chosen, df = _session(mark="")
    chosen = _name_megre_protocols(rows, chosen, df)
    names = _names(rows, chosen)
    assert names["s05"].get("reconstruction") == "nd"
    assert "reconstruction" not in names["s06"]


def test_one_protocol_keeps_its_acquisition_label():
    rows, chosen, df = _session()
    keep = {"s05", "s06", "s07", "s08"}
    rows = [r for r in rows if r.series_uid in keep]
    chosen = {r.row_id.hex: chosen[r.row_id.hex] for r in rows}
    chosen = _name_megre_protocols(rows, chosen, df)
    names = _names(rows, chosen)
    assert {e["acquisition"] for e in names.values()} == {"fl3p2"}
    assert names["s05"]["reconstruction"] == "nd"


def test_an_uncorrected_series_without_its_twin_is_left_as_it_is():
    rows, chosen, df = _session()
    rows = [r for r in rows if r.series_uid in {"s05", "s09"}]
    chosen = {r.row_id.hex: chosen[r.row_id.hex] for r in rows}
    chosen = _name_megre_protocols(rows, chosen, df)
    assert not any("reconstruction" in e for e in _names(rows, chosen).values())


def test_other_suffixes_and_scan_rules_are_not_touched():
    rows, chosen, df = _session()
    t1 = [r for r in rows if r.series_uid in {"s05", "s06", "s09", "s10"}]
    for r in t1[:2]:
        chosen[r.row_id.hex] = _megre(r, "mag", suffix="T1w")
    for r in t1[2:]:
        chosen[r.row_id.hex] = _megre(r, "mag", classifier="user_hint")
    before = {k: dict(c.candidate_entities) for k, c in chosen.items() if k in
              {r.row_id.hex for r in t1}}
    chosen = _name_megre_protocols(rows, chosen, df)
    for key, entities in before.items():
        assert chosen[key].candidate_entities == entities


def test_sessions_and_subjects_are_apart():
    """Two flip angles in two sessions are one protocol in each."""
    rows = [_row("a", "GRE", session="1"), _row("b", "GRE", session="2")]
    chosen = {r.row_id.hex: _megre(r, "mag") for r in rows}
    df = pd.DataFrame({"x": [0]})
    df.attrs["features_by_uid"] = {"a": SeriesFeatures(flip_angle=35.0).to_dict(),
                                   "b": SeriesFeatures(flip_angle=14.0).to_dict()}
    chosen = _name_megre_protocols(rows, chosen, df)
    assert {c.candidate_entities["acquisition"] for c in chosen.values()} == {"fl3p2"}


@pytest.mark.parametrize("angle, label", [(35.0, "fa35"), (14, "fa14"), (7.5, "fa7p5")])
def test_a_flip_angle_label_is_alphanumeric(angle, label):
    assert _flip_label(angle) == label


# -- the reconstruction marker, from the header ------------------------------------


def test_classic_image_type_carries_the_marker():
    ds = Dataset()
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"]
    assert _distortion_marker(ds) == "ND"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "DIS2D"]
    assert _distortion_marker(ds) == "DIS2D"


def test_xa_enhanced_dicom_keeps_it_in_a_private_frame_group():
    """Siemens XA writes ImageType without it; the marker is (0021,1175)
    inside the private sequence (0021,11FE) of the first frame group."""
    item = Dataset()
    item.add_new(0x00211175, "CS", ["ORIGINAL", "PRIMARY", "M", "DIS2D"])
    private = Dataset()
    private.add_new(0x002111FE, "SQ", Sequence([item]))
    ds = Dataset()
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "NONE"]
    ds.PerFrameFunctionalGroupsSequence = Sequence([private])
    assert _distortion_marker(ds) == "DIS2D"


def test_no_marker_reads_as_unknown():
    ds = Dataset()
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "NONE"]
    assert _distortion_marker(ds) == ""


def test_the_marker_reaches_the_series_features():
    feats = build([], {"distortion": "ND"}, n_files=0)
    assert feats.distortion == "ND"
    assert SeriesFeatures.from_dict(feats.to_dict()).distortion == "ND"
