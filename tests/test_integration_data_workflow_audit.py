"""Executable contract for the pre-submission integration and data audit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
AUDIT = (
    ROOT
    / "specs"
    / "submission-readiness"
    / "integration-data-workflow-audit-20260829.json"
)


def _audit() -> dict[str, object]:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_audit_inventory_matches_repository() -> None:
    """Counts in the dated audit must remain reproducible."""
    audit = _audit()
    inventory = audit["inventory"]
    assert isinstance(inventory, dict)

    dataset_registry = json.loads(
        (ROOT / "specs/dataset-registry/registry.json").read_text(encoding="utf-8")
    )
    domain_registry = json.loads(
        (ROOT / "specs/domain-templates/registry.json").read_text(encoding="utf-8")
    )
    templates = domain_registry["templates"]

    assert inventory["registered_datasets"] == len(dataset_registry)
    assert inventory["committed_dataset_snapshots"] == len(
        list((ROOT / "specs/dataset-registry/snapshots").glob("*.json"))
    )
    assert inventory["dataset_transform_python_scripts"] == len(
        list((ROOT / "specs/dataset-registry/transforms").glob("*.py"))
    )
    assert inventory["domain_templates"] == len(templates)
    assert inventory["domain_templates_with_examples"] == sum(
        bool(template["examples"]) for template in templates
    )
    assert inventory["unique_domain_example_scripts"] == len(
        {example for template in templates for example in template["examples"]}
    )
    assert inventory["python_example_scripts"] == len(
        list((ROOT / "examples").glob("**/*.py"))
    )
    assert inventory["notebook_examples"] == len(
        list((ROOT / "examples").glob("*.ipynb"))
    )


def test_audit_keeps_all_detected_gaps_open_for_disposition() -> None:
    """Discovery does not silently turn a gap into a completed repair."""
    audit = _audit()
    findings = audit["findings"]
    assert isinstance(findings, list)
    by_id = {finding["id"]: finding for finding in findings}

    assert set(by_id) == {f"INTDATA-{number:03d}" for number in range(1, 9)}
    assert {finding["state"] for finding in findings} == {"open"}
    assert {finding["severity"] for finding in findings} == {"high", "medium"}
    assert all(finding["evidence"] for finding in findings)
    assert all(finding["required_disposition"] for finding in findings)
