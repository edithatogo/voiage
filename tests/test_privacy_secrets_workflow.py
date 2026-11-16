"""Contracts for the shared, isolated privacy and secrets workflow."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/privacy-secrets.yml"
ACTION_COMMIT = "dd3a6d89a89a59fd1fcb231382a0dddbb868d363"


def _workflow() -> dict[str | bool, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_privacy_workflow_covers_changes_and_scheduled_audits() -> None:
    workflow = _workflow()
    events = workflow[True]  # PyYAML's safe loader interprets the YAML 1.1 key "on".
    assert {
        "pull_request",
        "push",
        "merge_group",
        "schedule",
        "workflow_dispatch",
    } <= set(events)
    assert "main" in events["push"]["branches"]
    assert events["schedule"]
    assert workflow["permissions"] == {"contents": "read"}


def test_privacy_workflow_uses_immutable_qualified_action_and_complete_history() -> (
    None
):
    steps = _workflow()["jobs"]["privacy-and-secrets"]["steps"]
    checkout, scanner = steps
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["submodules"] is False
    assert scanner["uses"] == (
        "edithatogo/repository-standards/.github/actions/"
        f"privacy-secrets@{ACTION_COMMIT}"
    )
    assert scanner["with"]["profile"] == "${{ inputs.profile || 'repository' }}"
