"""Tests for the full-data export (Settings → AI Agent).

No agent and no model: the builder is a plain callable here, which is the
point of the split — collecting, naming, evidence and writing are tested
without a server, and the server side has its own API tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from bidsmgr.editor.types import (
    FileVerdict,
    Issue,
    Severity,
    ValidationReport,
)
from bidsmgr.gui.ai_explainer import AgentError
from bidsmgr.gui.ai_full_data import (
    FORMAT,
    collect_requests,
    export_folder,
    export_records,
)


def _issue(
    severity: Severity = Severity.ERR,
    message: str = "Bad field",
    rule_id: str = "JSON_SCHEMA_VALIDATION_ERROR",
    field: str = "X",
    mirrored: bool = False,
) -> Issue:
    return Issue(
        severity=severity,
        rule_id=rule_id,
        message=message,
        field=field,
        mirrored=mirrored,
    )


def _report() -> ValidationReport:
    return ValidationReport(
        files=[
            FileVerdict(
                path=Path("sub-01/anat/sub-01_T1w.json"),
                datatype="anat",
                suffix="T1w",
                issues=[
                    _issue(),
                    _issue(Severity.OK, message="Entity set is valid."),
                    _issue(message="Mirror", mirrored=True),
                ],
            ),
        ],
        folder_issues={"sub-01/anat": [_issue(Severity.WARN, message="Folder")]},
        dataset_issues=[_issue(message="Dataset")],
    )


def _build(calls: list, *, system: str = "SYSTEM") -> Callable[..., dict]:
    def build(user_input, context, dataset_structure=None, file_content=None):
        calls.append({
            "user_input": user_input,
            "context": dict(context),
            "dataset_structure": dataset_structure,
            "file_content": file_content,
        })
        return {
            "system_prompt": system,
            "user_prompt": user_input,
            "intent": "explain",
            "classification_source": "manual",
            "prompt_available": True,
            "retrieval_found": True,
            "retrieval_quality": "partial",
        }

    return build


def test_only_errors_and_warnings_are_exported():
    requests = collect_requests(_report())

    assert len(requests) == 3
    assert [r.payload["message"] for r in requests] == [
        "Bad field", "Folder", "Dataset",
    ]


def test_a_file_finding_carries_its_file_context():
    request = collect_requests(_report())[0]

    assert request.payload["path"] == "sub-01/anat/sub-01_T1w.json"
    assert request.payload["datatype"] == "anat"
    assert request.payload["suffix"] == "T1w"
    assert request.prompt.startswith("Explain this error")


def test_folder_and_dataset_findings_say_which_scope_they_are():
    folder, dataset = collect_requests(_report())[1:]

    assert folder.payload["scope"] == "folder"
    assert folder.payload["folder"] == "sub-01/anat"
    assert "path" not in folder.payload
    assert dataset.payload["scope"] == "dataset"
    assert "path" not in dataset.payload


def test_two_dataset_findings_with_the_same_rule_still_get_their_own_key():
    """Dataset issues share a rule and field; only the message differs."""
    report = ValidationReport(dataset_issues=[
        _issue(message="First"),
        _issue(message="Second"),
    ])

    first, second = collect_requests(report)
    assert first.key != second.key


def test_each_export_gets_a_new_timestamped_folder(tmp_path):
    first = export_folder(str(tmp_path / "logs"), tmp_path / "ds")
    first.mkdir(parents=True)
    second = export_folder(str(tmp_path / "logs"), tmp_path / "ds")

    assert first.parent.name == "full-data"
    assert first.name.startswith("ds-")
    assert first != second


def test_export_writes_one_record_per_finding_and_a_manifest(tmp_path):
    report = _report()
    (tmp_path / "sub-01" / "anat").mkdir(parents=True)
    (tmp_path / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        '{"RepetitionTime": 2.0}', encoding="utf-8",
    )
    calls: list = []

    result = export_records(
        collect_requests(report), tmp_path, str(tmp_path / "logs"),
        _build(calls),
    )

    assert result["saved"] == 3
    assert result["failed"] == []
    assert result["aborted"] == ""

    folder = Path(result["folder"])
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in folder.glob("*.json")
        if p.name != "manifest.json"
    ]
    assert len(records) == 3
    assert all(record["format"] == FORMAT for record in records)
    assert all(record["output"] is None for record in records)
    assert all("response" not in record for record in records)
    assert all(record["system_prompt"] == "SYSTEM" for record in records)

    manifest = json.loads(
        (folder / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["findings"] == 3
    assert manifest["saved"] == 3
    assert manifest["dataset"] == str(tmp_path)


def test_the_file_finding_gets_the_file_digest(tmp_path):
    report = _report()
    (tmp_path / "sub-01" / "anat").mkdir(parents=True)
    (tmp_path / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        '{"RepetitionTime": 2.0}', encoding="utf-8",
    )
    calls: list = []

    export_records(
        collect_requests(report), tmp_path, str(tmp_path / "logs"),
        _build(calls),
    )

    by_path = {
        call["context"].get("path"): call for call in calls
    }
    file_call = by_path["sub-01/anat/sub-01_T1w.json"]
    assert "RepetitionTime" in file_call["file_content"]
    assert "sub-01" in file_call["dataset_structure"]


def test_the_dataset_tree_can_be_left_out_of_the_export(tmp_path):
    report = _report()
    (tmp_path / "sub-01" / "anat").mkdir(parents=True)
    (tmp_path / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        '{"RepetitionTime": 2.0}', encoding="utf-8",
    )
    calls: list = []

    export_records(
        collect_requests(report), tmp_path, str(tmp_path / "logs"),
        _build(calls), include_dataset_structure=False,
    )

    assert all(call["dataset_structure"] == "" for call in calls)
    by_path = {call["context"].get("path"): call for call in calls}
    assert "RepetitionTime" in (
        by_path["sub-01/anat/sub-01_T1w.json"]["file_content"]
    )


def test_the_file_digest_can_be_left_out_of_the_export(tmp_path):
    report = _report()
    (tmp_path / "sub-01" / "anat").mkdir(parents=True)
    (tmp_path / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        '{"RepetitionTime": 2.0}', encoding="utf-8",
    )
    calls: list = []

    export_records(
        collect_requests(report), tmp_path, str(tmp_path / "logs"),
        _build(calls), include_file_content=False,
    )

    assert all(call["file_content"] == "" for call in calls)
    by_path = {call["context"].get("path"): call for call in calls}
    assert "sub-01" in (
        by_path["sub-01/anat/sub-01_T1w.json"]["dataset_structure"]
    )


def test_an_unreachable_agent_keeps_what_was_written_and_stops(tmp_path):
    calls = {"count": 0}

    def build(user_input, context, dataset_structure=None, file_content=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise AgentError("The AI agent is not reachable.")
        return {
            "system_prompt": "SYSTEM",
            "user_prompt": user_input,
            "intent": "explain",
            "classification_source": "manual",
            "prompt_available": True,
        }

    result = export_records(
        collect_requests(_report()), tmp_path, str(tmp_path / "logs"), build,
    )

    assert result["saved"] == 1
    assert "not reachable" in result["aborted"]
    folder = Path(result["folder"])
    assert len(list(folder.glob("*.json"))) == 2  # the record + the manifest


def test_one_rejected_request_does_not_stop_the_rest(tmp_path):
    calls = {"count": 0}

    def build(user_input, context, dataset_structure=None, file_content=None):
        calls["count"] += 1
        if calls["count"] == 2:
            raise AgentError("500 from the retriever", status=500)
        return {
            "system_prompt": "SYSTEM",
            "user_prompt": user_input,
            "intent": "explain",
            "classification_source": "manual",
            "prompt_available": True,
        }

    result = export_records(
        collect_requests(_report()), tmp_path, str(tmp_path / "logs"), build,
    )

    assert result["saved"] == 2
    assert len(result["failed"]) == 1
    assert result["aborted"] == ""
