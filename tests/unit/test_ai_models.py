"""The models folder scan: what counts as a downloaded model.

Mirrors the honest-enough check the agent makes on its own cache: one
``models--<org>--<name>`` folder per model, ``refs`` pointing at a
revision, that revision's snapshot holding at least one file. A model
copied in by hand must count, and a half-fetched download must not.
Nothing here touches a model or the network — one directory listing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bidsmgr.gui import ai_models


@pytest.fixture()
def _models_dir_in_tmp(tmp_path: Path, monkeypatch) -> Path:
    """Scan tests point at a fresh tmp folder, never at the user's real
    cache. Not autouse: the models_dir tests below check the *real*
    resolution."""
    target = tmp_path / "models"
    target.mkdir(parents=True)
    monkeypatch.setattr(ai_models, "models_dir", lambda override="": target)
    return target


def _make_model(
    root: Path,
    name: str = "Qwen/Qwen3-1.7B",
    *,
    revision: str = "abc123",
    files: tuple[str, ...] = ("config.json", "model.safetensors"),
    incomplete: bool = False,
) -> None:
    """A synthetic cache folder in the standard HF layout."""
    repo = root / f"models--{name.replace('/', '--')}"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for filename in files:
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 10)
    if incomplete:
        blobs = repo / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "deadbeef.incomplete").write_bytes(b"partial")


class TestDownloadedModelIds:
    def test_a_complete_cache_folder_counts(self, _models_dir_in_tmp):
        _make_model(_models_dir_in_tmp)
        assert ai_models.downloaded_model_ids() == {"Qwen/Qwen3-1.7B"}

    def test_the_folder_spelling_maps_to_the_repo_id(self, _models_dir_in_tmp):
        _make_model(
            _models_dir_in_tmp,
            "MehranBazrafkan/gemma-3-270m-it-bids-finetuned",
        )
        ids = ai_models.downloaded_model_ids()
        assert ids == {"MehranBazrafkan/gemma-3-270m-it-bids-finetuned"}

    def test_several_models_are_all_found(self, _models_dir_in_tmp):
        _make_model(_models_dir_in_tmp, "Qwen/Qwen3-1.7B")
        _make_model(_models_dir_in_tmp, "Qwen/Qwen3-0.6B")
        assert ai_models.downloaded_model_ids() == {
            "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B",
        }

    def test_an_empty_folder_returns_nothing(self, _models_dir_in_tmp):
        assert ai_models.downloaded_model_ids() == set()

    def test_a_missing_folder_returns_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ai_models, "models_dir", lambda override="": tmp_path / "gone",
        )
        assert ai_models.downloaded_model_ids() == set()

    def test_no_refs_means_not_ready(self, _models_dir_in_tmp):
        _make_model(_models_dir_in_tmp)
        shutil.rmtree(
            _models_dir_in_tmp / "models--Qwen--Qwen3-1.7B" / "refs"
        )
        assert ai_models.downloaded_model_ids() == set()

    def test_an_empty_snapshot_does_not_count(self, _models_dir_in_tmp):
        _make_model(_models_dir_in_tmp, files=())
        assert ai_models.downloaded_model_ids() == set()

    def test_an_interrupted_download_does_not_count(self, _models_dir_in_tmp):
        _make_model(_models_dir_in_tmp, incomplete=True)
        assert ai_models.downloaded_model_ids() == set()

    def test_folders_that_are_not_models_are_ignored(
        self, _models_dir_in_tmp,
    ):
        junk = _models_dir_in_tmp / "some-other-folder"
        junk.mkdir()
        (junk / "file.txt").write_text("not a model", encoding="utf-8")
        assert ai_models.downloaded_model_ids() == set()


class TestPartialModelIds:
    """A started-but-unfinished download is a state of its own.

    The dropdown shows it with a different mark: "press Download to
    resume" is a different story from "nothing fetched yet", and an
    interrupted fetch is exactly the case that used to look broken.
    """

    def test_an_interrupted_download_is_partial(self, _models_dir_in_tmp):
        _make_model(_models_dir_in_tmp, incomplete=True)
        assert ai_models.partial_model_ids() == {"Qwen/Qwen3-1.7B"}

    def test_a_complete_model_is_not_partial(self, _models_dir_in_tmp):
        _make_model(_models_dir_in_tmp)
        assert ai_models.partial_model_ids() == set()

    def test_an_empty_folder_returns_nothing(self, _models_dir_in_tmp):
        assert ai_models.partial_model_ids() == set()

    def test_a_missing_folder_returns_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ai_models, "models_dir", lambda override="": tmp_path / "gone",
        )
        assert ai_models.partial_model_ids() == set()

    def test_a_half_fetched_folder_without_blobs_is_not_partial(
        self, _models_dir_in_tmp,
    ):
        """No ``.incomplete`` file means nothing was started, not that a
        download is half done."""
        _make_model(_models_dir_in_tmp)
        shutil.rmtree(
            _models_dir_in_tmp / "models--Qwen--Qwen3-1.7B" / "refs"
        )
        assert ai_models.partial_model_ids() == set()


class TestModelsDir:
    def test_an_override_wins(self, tmp_path, monkeypatch):
        chosen = tmp_path / "elsewhere" / "models"
        assert ai_models.models_dir(str(chosen)) == chosen

    def test_no_override_falls_back_to_the_service_default(self, monkeypatch):
        # The default is the service's own resolution — the same value
        # the agent is spawned with, so the two can never disagree.
        monkeypatch.setattr(
            ai_models.agent_service, "models_dir",
            lambda override: Path("/resolved/by/service"),
        )
        assert ai_models.models_dir() == Path("/resolved/by/service")

    def test_a_blank_override_counts_as_no_override(self, tmp_path):
        from bidsmgr.agent_service import models_dir as service_models_dir

        assert ai_models.models_dir("   ") == service_models_dir()
