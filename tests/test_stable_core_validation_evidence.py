"""Governance tests for stable-core numerical validation evidence."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

CONTRACT_PATH = Path("specs/v1/stable-core-validation-evidence.json")
SCHEMA_PATH = Path("specs/v1/stable-core-validation-evidence.schema.json")
ASSURANCE_PATH = Path("specs/v1/stable-estimator-assurance.json")
STABLE_API_PATH = Path("specs/v1/stable-api.json")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_validation_evidence_conforms_to_its_schema() -> None:
    Draft202012Validator(_load(SCHEMA_PATH)).validate(_load(CONTRACT_PATH))


def test_validation_evidence_covers_every_stable_estimator_profile() -> None:
    contract = _load(CONTRACT_PATH)
    assurance = _load(ASSURANCE_PATH)
    method_ids = [entry["method_id"] for entry in contract["methods"]]

    assert len(method_ids) == len(set(method_ids))
    assert set(method_ids) == {
        profile["method_id"] for profile in assurance["profiles"]
    }
    assert _load(STABLE_API_PATH)["validation_evidence_contract"] == str(CONTRACT_PATH)


def test_every_method_has_analytical_independent_and_metamorphic_evidence() -> None:
    contract = _load(CONTRACT_PATH)

    for method in contract["methods"]:
        kinds = {item["kind"] for item in method["evidence"]}
        assert {"analytical", "independent-reference", "metamorphic"} <= kinds
        for item in method["evidence"]:
            evidence_path = Path(item["path"])
            assert evidence_path.is_file()
            assert item["test_id"] in evidence_path.read_text(encoding="utf-8")


def test_differential_evidence_is_never_mislabelled_as_independent() -> None:
    contract = _load(CONTRACT_PATH)

    for method in contract["methods"]:
        for item in method["evidence"]:
            if item["kind"] == "independent-reference":
                assert item["reference_boundary"] in {
                    "closed-form-definition",
                    "hand-derived-fixture",
                    "separately-implemented-oracle",
                }
                assert item["reference_boundary"] != "same-runtime-differential"


def test_reference_claims_disclose_limitations_and_promotion_effect() -> None:
    contract = _load(CONTRACT_PATH)

    assert contract["claim_policy"]["published_parity"] == (
        "no-external-package-parity-claim-from-this-contract"
    )
    assert contract["claim_policy"]["scientific_validation"] == (
        "supports-repository-promotion-only-with-human-review"
    )
    assert contract["claim_policy"]["differential_role"] == (
        "supplementary-not-a-substitute-for-independent-evidence"
    )
