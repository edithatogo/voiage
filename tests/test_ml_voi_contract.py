"""Contract tests distinguishing information gain from decision VOI."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tomllib

from jsonschema import Draft202012Validator
import pytest

from voiage.contracts.concerns import EvidenceReference

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "specs/frontier/ai-assisted-evidence-triage/v1/fixtures/manifest.json"
FIXTURE = (
    ROOT
    / "specs/frontier/ai-assisted-evidence-triage/v1/fixtures/normative"
    / "eig-versus-voi.json"
)
SCHEMA = FIXTURE.parents[2] / "schemas" / "eig-versus-decision-voi.schema.json"


def test_eig_fixture_keeps_information_and_decision_values_distinct() -> None:
    """Entropy reduction must not be mislabeled as economic VOI."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["protocol_version"] == "1.0.0"
    assert payload["predictive_model"]
    assert payload["information_action"]
    assert payload["utility_definition"]
    assert payload["stopping_rule"]
    assert payload["privacy_budget"] >= 0
    assert payload["provenance"] == {
        "source": "synthetic",
        "network_required": False,
        "private_data": False,
    }
    assert payload["expected_information_gain_nats"] > 0
    assert payload["expected_decision_voi"] == (
        payload["posterior_expected_utility"]
        - payload["current_expected_utility"]
        - payload["information_cost"]
    )
    assert "utility" in payload["interpretation"]
    assert "cost" in payload["interpretation"]
    assert payload["expected_decision_voi"] != payload["expected_information_gain_nats"]


def test_eig_fixture_schema_declares_formal_decision_contract() -> None:
    """The fixture's required fields remain machine-readable and versioned."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    required = set(schema["required"])
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    assert required <= payload.keys()
    assert schema["properties"]["provenance"]["properties"]["network_required"] == {
        "const": False
    }


def test_eig_fixture_manifest_links_schema_and_input() -> None:
    """The normative fixture remains discoverable from the canonical manifest."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["normative"]
        if item["name"].startswith("expected information")
    )
    fixture_root = MANIFEST.parent
    assert (fixture_root / entry["input_artifact"]).is_file()
    assert (fixture_root / entry["schema_artifact"]).resolve().is_file()


def test_ml_contract_requires_offline_cpu_and_optional_backends() -> None:
    """ML/LLM methods must not require providers or private-data transport."""
    spec = (ROOT / "conductor/tracks/ml_llm_agent_voi_20260723/spec.md").read_text(
        encoding="utf-8"
    )
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = "\n".join(metadata["project"]["dependencies"]).lower()

    spec = spec.lower()
    assert "offline tables" in spec
    assert "cpu deterministic" in spec
    assert "no network or" in spec
    assert "private-data transmission" in spec
    assert "pyro" in spec
    assert "botorch" in spec
    assert "pyro" not in base
    assert "botorch" not in base


def test_ml_contract_keeps_private_locators_and_fallbacks_explicit() -> None:
    """Private evidence and degraded execution must fail closed or be labelled."""
    with pytest.raises(ValueError, match="local_private"):
        EvidenceReference(
            id="private-fixture",
            title="private fixture",
            summary="private fixture",
            status="verified",
            evidence_kind="source",
            locator_kind="local_path",
            locator="/private/input.json",
            observed_at=datetime(2026, 7, 28, tzinfo=UTC),
            visibility="public",
        )

    kernel = (ROOT / "voiage/contracts/kernel.py").read_text(encoding="utf-8")
    assert "allow_fallback" in kernel
    assert 'status="degraded"' in kernel
    assert 'code="backend_fallback"' in kernel
