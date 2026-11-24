"""Compress or decompress images: the setting, the dialog, the Editor.

The dialog plans on a thread with its spinner where the images will appear
(GUI_LOOK_AND_FEEL 4.34), lists every image nested as the dataset is with the
files that name them, and applies on another thread with Stop. The Editor
offers it in Tools and on the tree's right-click, and the file it was showing
follows to its new name.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from bidsmgr.editor import compression as engine
from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.compression_dialog import CompressionDialog, human
from bidsmgr.project.operations import peek_undo

pytestmark = pytest.mark.gui

T1 = "sub-01/anat/sub-01_T1w.nii.gz"
BOLD = "sub-01/func/sub-01_task-rest_bold.nii.gz"
T1_02 = "sub-02/anat/sub-02_T1w.nii.gz"


def _image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).integers(0, 500, size=(6, 6, 4)).astype(np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    path.with_name(path.name.split(".")[0] + ".json").write_text("{}")


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset_description.json").write_text(json.dumps({"Name": "d", "BIDSVersion": "1.11.1"}))
    for i, rel in enumerate((T1, BOLD, T1_02)):
        _image(root / rel, i)
    (root / "sub-01/sub-01_scans.tsv").write_text(
        "filename\tacq_time\nanat/sub-01_T1w.nii.gz\tn/a\nfunc/sub-01_task-rest_bold.nii.gz\tn/a\n")
    return root


def _open(qtbot, root: Path, targets=None) -> CompressionDialog:
    dlg = CompressionDialog(root, targets)
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: not dlg.planning(), timeout=20_000)
    return dlg


def _leaves(dlg) -> dict[str, object]:
    out = {}
    for item in dlg._tree._leaves():
        key = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if key:
            out[key] = item
    return out


def _section(dlg, title: str) -> list[tuple[str, str]]:
    for i in range(dlg._tree.topLevelItemCount()):
        top = dlg._tree.topLevelItem(i)
        if top.text(0) == title:
            return [(top.child(j).text(0), top.child(j).text(1)) for j in range(top.childCount())]
    return []


def test_it_opens_in_the_direction_the_images_can_go(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    assert dlg._direction.current() == "decompress"
    assert sorted(_leaves(dlg)) == sorted([T1, BOLD, T1_02])
    assert _leaves(dlg)[T1].text(1) == "sub-01_T1w.nii"
    assert _section(dlg, "Follows automatically") == [
        ("sub-01/sub-01_scans.tsv", "the filename column follows (2 images)")]
    assert dlg._stat_images.number.text() == "3"
    assert dlg._stat_refs.number.text() == "1"
    assert dlg._ok.isEnabled() and dlg._ok.text() == "Decompress 3 images"
    assert dlg._busy.isHidden() and dlg._progress.isHidden()


def test_unticking_an_image_changes_the_count_and_the_button(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    _leaves(dlg)[T1_02].setCheckState(0, Qt.CheckState.Unchecked)
    assert dlg._stat_images.number.text() == "2"
    assert dlg._ok.text() == "Decompress 2 images"
    dlg._set_all(Qt.CheckState.Unchecked)
    assert not dlg._ok.isEnabled() and dlg._status.text() == "No image ticked."


def test_the_other_way_says_there_is_nothing_to_do(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    dlg._direction.buttons["compress"].click()
    assert dlg.planning() and not dlg._ok.isEnabled(), "the old plan cannot be applied"
    dlg._on_apply()
    assert dlg._worker is None
    qtbot.waitUntil(lambda: not dlg.planning(), timeout=20_000)
    assert dlg._plan.compress is True
    assert not dlg._ok.isEnabled()
    assert dlg._status.text() == "Every image here is already .nii.gz."
    assert _section(dlg, "Left as they are") == [("3 images", "already .nii.gz")]


def test_it_opens_on_what_was_picked(qtbot, ds) -> None:
    dlg = _open(qtbot, ds, [ds / T1])
    assert sorted(_leaves(dlg)) == [T1]


def test_applying_runs_on_a_thread_and_is_one_undo(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    _leaves(dlg)[T1_02].setCheckState(0, Qt.CheckState.Unchecked)
    with qtbot.waitSignal(dlg.accepted, timeout=30_000):
        dlg._on_apply()
    assert (ds / "sub-01/anat/sub-01_T1w.nii").is_file() and (ds / T1_02).is_file()
    assert dlg.outcome().ok and sorted(dlg.outcome().changed) == sorted([T1, BOLD])
    assert dlg.renamed() == {ds / T1: ds / "sub-01/anat/sub-01_T1w.nii",
                             ds / BOLD: ds / "sub-01/func/sub-01_task-rest_bold.nii"}
    assert peek_undo(ds).label == "Decompress 2 images (.nii.gz to .nii)"


def test_stop_leaves_the_dataset_as_it_was(qtbot, ds, monkeypatch) -> None:
    started = threading.Event()
    real = engine.transcode

    def slow(source, target, compress, *, level=6, stop=None):
        started.set()
        while stop is not None and not stop.is_set():
            stop.wait(0.02)
        return real(source, target, compress, level=level, stop=stop)

    monkeypatch.setattr(engine, "transcode", slow)
    dlg = _open(qtbot, ds)
    dlg._on_apply()
    assert not dlg._progress.isHidden() and not dlg._stop.isHidden() and not dlg._tree.isEnabled()
    qtbot.waitUntil(started.is_set, timeout=10_000)
    dlg._on_stop()
    qtbot.waitUntil(lambda: dlg._worker is None, timeout=20_000)
    assert dlg._status.text() == "Stopped. Nothing was changed."
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert (ds / T1).is_file() and not (ds / "sub-01/anat/sub-01_T1w.nii").exists()
    assert peek_undo(ds) is None
    assert not list(ds.rglob("*.part"))
    assert dlg._tree.isEnabled() and dlg._progress.isHidden()


def test_too_little_space_is_said_before_anything_is_written(qtbot, ds, monkeypatch) -> None:
    import shutil

    real = shutil.disk_usage
    monkeypatch.setattr(engine.shutil, "disk_usage",
                        lambda p: real(p)._replace(free=100))
    dlg = _open(qtbot, ds)
    assert not dlg._ok.isEnabled()
    assert dlg._status.text().startswith("Not enough free space")
    assert dlg._stat_free.number.objectName() == "stat-value-err"


def test_sizes_read_the_way_a_file_manager_says_them() -> None:
    assert [human(n) for n in (0, 1536, 5 * 1024 ** 2, 3 * 1024 ** 3)] == [
        "0 B", "1.5 KB", "5.0 MB", "3.0 GB"]


# -- the setting --------------------------------------------------------------------


def test_compressed_is_the_default_and_the_choice_is_kept(qtbot) -> None:
    from bidsmgr.gui.settings_dialog import SettingsDialog

    assert AppSettings.load().convert_compress_nifti is True
    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    combo = dlg._convert_compress_nifti
    assert combo.currentIndex() == 0 and combo.currentText().startswith("Compressed (.nii.gz)")
    combo.setCurrentIndex(1)
    dlg._on_save()
    assert AppSettings.load().convert_compress_nifti is False
    again = SettingsDialog(AppSettings.load())
    qtbot.addWidget(again)
    assert again._convert_compress_nifti.currentIndex() == 1
    again._on_restore_defaults()
    assert again._convert_compress_nifti.currentIndex() == 0


def test_the_conversion_is_handed_the_setting(qtbot, tmp_path, monkeypatch) -> None:
    """Settings > Convert reaches ``run_convert`` through the worker."""
    import pandas as pd

    from bidsmgr.cli import convert as convert_mod
    from bidsmgr.workers import ConvertWorker

    seen = {}
    monkeypatch.setattr(convert_mod, "run_convert", lambda *a, **kw: seen.update(kw) or 0)
    tsv = tmp_path / "inv.tsv"
    df = pd.DataFrame([{"participant_id": "sub-01", "dataset": "d", "include": 1}])
    worker = ConvertWorker(df, tsv, tmp_path / "out", compress=False)
    with qtbot.waitSignal(worker.finished_with_result, timeout=30_000):
        worker.start()
    worker.wait()
    assert seen["compress"] is False


# -- in the Editor ---------------------------------------------------------------------


def test_the_editor_offers_it_with_the_names(qtbot) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    labels = [a.text() for a in panel._tools_menu.actions()]
    assert labels.index("Compress or decompress images...") == labels.index("Index widths...") + 1


def _right_click(panel, path: Path, monkeypatch) -> list:
    from PyQt6.QtWidgets import QMenu

    from bidsmgr.gui.widgets.bids_tree_pane import PATH_ROLE

    tree = panel._tree_pane
    item = None
    stack = [tree._tree.topLevelItem(i) for i in range(tree._tree.topLevelItemCount())]
    while stack:
        node = stack.pop()
        if node.data(0, PATH_ROLE) == str(path):
            item = node
            break
        node.setExpanded(True)
        stack.extend(node.child(j) for j in range(node.childCount()))
    assert item is not None, path
    shown: list = []
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: shown.append(self))
    tree._tree.setCurrentItem(item)
    tree._on_show_context_menu(tree._tree.visualItemRect(item).center())
    return [a for a in shown[-1].actions()]


def test_the_tree_offers_it_on_what_was_clicked(qtbot, ds, monkeypatch) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._tree_pane.set_root(ds)
    opened = []

    def fake_exec(dialog):
        opened.append([p.name for p in dialog._scope_bar.targets()])
        dialog.deleteLater()
        return 0

    monkeypatch.setattr(CompressionDialog, "exec", fake_exec)
    actions = _right_click(panel, ds / T1, monkeypatch)
    entry = next(a for a in actions if a.text() == "Decompress this image (.nii)...")
    entry.trigger()
    assert opened == [["sub-01_T1w.nii.gz"]]
    folder = _right_click(panel, ds / "sub-01", monkeypatch)
    assert any(a.text() == "Compress or decompress images..." for a in folder)


def test_the_open_image_follows_to_its_new_name(qtbot, ds, monkeypatch) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._tree_pane.set_root(ds)
    monkeypatch.setattr(panel, "current_root", lambda: ds)
    panel._on_file_selected(ds / T1)

    def run_it(dialog):
        qtbot.waitUntil(lambda: not dialog.planning(), timeout=20_000)
        with qtbot.waitSignal(dialog.accepted, timeout=30_000):
            dialog._on_apply()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CompressionDialog, "exec", run_it)
    panel._on_compression([ds / "sub-01"])
    assert panel._open_path == ds / "sub-01/anat/sub-01_T1w.nii"
    assert (ds / T1_02).is_file(), "outside the scope, left alone"


def test_the_predicted_path_follows_the_setting(qtbot) -> None:
    """Properties shows the name a row is converted to: ``.nii`` when
    Settings > Convert writes uncompressed images. Recordings that are not
    images keep their own extension."""
    from bidsmgr.gui.properties_panel import PropertiesPanel

    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    assert panel._preview_extension(0, "anat", "T1w") == ".nii.gz"
    settings = AppSettings.load()
    settings.convert_compress_nifti = False
    settings.save()
    assert panel._preview_extension(0, "anat", "T1w") == ".nii"
    assert panel._preview_extension(0, "func", "physio") == ".tsv.gz"
