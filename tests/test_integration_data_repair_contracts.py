"""Regression contracts for accepted analytical, data, and integration repairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

from scripts.run_example_smoke import validate_manifest
from scripts.validate_research_workflow_corpus import validate as validate_corpus

ROOT = Path(__file__).parents[1]


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_dataset_registry_uses_real_fixture_digests_and_honest_status() -> None:
    registry = _json("specs/dataset-registry/registry.json")
    status = _json("specs/dataset-registry/status.json")

    assert status["mode"] == "fixture_only"
    assert status["live_refresh_implemented"] is False
    assert status["source_bytes_included"] is False
    assert status["network_default"] == "deny"
    for record in registry.values():
        snapshot = record["snapshot"]
        path = ROOT / snapshot["path"]
        assert snapshot["classification"] == "illustrative_metadata_fixture"
        assert snapshot["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["snapshot_hash"] == f"sha256:{snapshot['sha256']}"


def test_dataset_docs_do_not_advertise_nonexistent_refresh_scripts() -> None:
    docs = (ROOT / "docs/astro-site/src/content/docs/dataset-registry.mdx").read_text(
        encoding="utf-8"
    )
    transforms = (ROOT / "specs/dataset-registry/transforms/README.md").read_text(
        encoding="utf-8"
    )

    assert "fixture-only" in docs
    assert "does not provide live refresh scripts" in docs
    assert "no executable transforms" in transforms
    assert "refresh_all.py" not in docs + transforms


def test_ecosystem_interchange_profiles_match_actual_formats_and_extras() -> None:
    profiles = _json("specs/integrations/ecosystem-interchange-profiles-v1.json")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert profiles["treeage_xml"]["compatibility"] == "voiage_xml_profile_only"
    assert profiles["treeage_xml"]["treeage_product_roundtrip_verified"] is False
    assert profiles["rds"]["status"] == "unsupported"
    assert profiles["rds"]["implementation"] == "none"
    assert profiles["excel"]["status"] == "optional"
    assert profiles["excel"]["extra"] == "excel"
    assert any(
        dependency.startswith("openpyxl>=")
        for dependency in project["project"]["optional-dependencies"]["excel"]
    )


def test_domain_templates_distinguish_validated_workflows_from_templates() -> None:
    registry = _json("specs/domain-templates/registry.json")
    stable = [item for item in registry["templates"] if item["maturity"] == "stable"]

    assert {item["template_id"] for item in stable} == {
        "churn_retention",
        "market_entry",
    }
    for item in registry["templates"]:
        assert item["validation_level"] in {
            "registry_driven_worked_example",
            "template_only",
        }
        if item["maturity"] == "stable":
            assert item["validation_level"] == "registry_driven_worked_example"
            assert len(item["examples"]) == 1
            assert (ROOT / item["examples"][0]).is_file()
        else:
            assert item["validation_level"] == "template_only"


def test_example_smoke_manifest_covers_every_script_and_notebook() -> None:
    manifest = _json("specs/examples/smoke-manifest-v1.json")
    discovered = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "examples").rglob("*")
        if path.suffix in {".py", ".ipynb"}
    }
    entries = {entry["path"]: entry for entry in manifest["entries"]}

    assert entries.keys() == discovered
    assert all(
        entry["disposition"] in {"execute", "quarantine"} for entry in entries.values()
    )
    assert all(
        entry.get("reason")
        for entry in entries.values()
        if entry["disposition"] == "quarantine"
    )
    assert all(
        entry.get("command")
        for entry in entries.values()
        if entry["disposition"] == "execute"
    )
    assert manifest["network_default"] == "deny"
    assert manifest["hardware_default"] == "cpu"
    assert validate_manifest(manifest) == []


def test_enterprise_adapters_are_sdk_free_profiles_not_product_claims() -> None:
    profiles = _json("specs/integrations/enterprise/interchange-profiles-v1.json")

    assert profiles["scope"] == "sdk_free_interchange_constructors"
    assert profiles["product_compatibility_claimed"] is False
    assert profiles["producer_roundtrip_verified"] is False
    assert set(profiles["profiles"]) == {
        "mlflow",
        "openlineage",
        "dbt",
        "experimentation",
        "cate",
        "forecast",
    }
    assert all(
        item["status"] == "voiage_profile" for item in profiles["profiles"].values()
    )


def test_research_workflow_corpus_covers_four_representative_domains() -> None:
    corpus = _json("specs/workflows/research-workflow-corpus-v1.json")

    assert corpus["rights"] == "repository-authored synthetic fixtures"
    assert corpus["network_required"] is False
    assert {workflow["domain"] for workflow in corpus["workflows"]} == {
        "health",
        "environmental",
        "financial",
        "enterprise",
    }
    for workflow in corpus["workflows"]:
        assert (ROOT / workflow["input"]).is_file()
        assert (ROOT / workflow["expected_result"]).is_file()
        assert workflow["stages"] == [
            "source",
            "ingestion",
            "analysis",
            "serialization",
            "report",
        ]
        assert workflow["verification"]
    assert validate_corpus() == []


def test_analytical_candidates_have_explicit_nonstable_decisions() -> None:
    decisions = _json(
        "specs/submission-readiness/analytical-candidate-decisions-20260829.json"
    )
    records = {record["id"]: record for record in decisions["candidates"]}

    assert records.keys() == {
        "ANALYTIC-001",
        "ANALYTIC-002",
        "ANALYTIC-003",
        "ANALYTIC-004",
    }
    assert {records[key]["decision"] for key in records if key != "ANALYTIC-004"} == {
        "retain_research_preview"
    }
    assert records["ANALYTIC-004"]["decision"] == "reviewed_exclusion"
    assert all(record["stable_api_added"] is False for record in records.values())
    assert all(
        record["rationale"] and record["revisit_gate"] for record in records.values()
    )
