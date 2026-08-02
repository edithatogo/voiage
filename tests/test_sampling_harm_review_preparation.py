"""Fail-closed tests for the H8-C frozen review preparation."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from voiage.sampling_harm_review_preparation import (
    SamplingHarmReviewPreparationError,
    validate_sampling_harm_review_preparation,
)
from voiage.scientific_review_evidence import (
    ScientificReviewEvidenceError,
    validate_scientific_review_bundle,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/sampling-acquisition-harm/v1"
CANDIDATE = "8d6c67879050f161258ed95d878a72e2bb6b22dd"
PACKAGE = "PACKAGE_COMMIT_PENDING"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _validate(payload: dict[str, object]) -> None:
    validate_sampling_harm_review_preparation(
        payload,
        repository_root=ROOT,
        expected_candidate_commit=CANDIDATE,
        expected_package_commit=PACKAGE,
        now=datetime(2026, 8, 3, tzinfo=UTC),
    )


def test_frozen_review_preparation_validates_exact_candidate_tree() -> None:
    _validate(_json(CONTRACT / "review-preparation.json"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: item["candidate_tree"].__setitem__("value", "f" * 40),
            "candidate tree",
        ),
        (
            lambda item: item["artifact_manifest"].__setitem__("sha256", "f" * 64),
            "manifest canonical digest",
        ),
        (
            lambda item: item["review_packet"].__setitem__("path", "../packet.json"),
            "review preparation invalid",
        ),
        (
            lambda item: item["authority_boundary"].__setitem__(
                "scientific_review_completed", True
            ),
            "review preparation invalid",
        ),
        (
            lambda item: item["required_independent_review_roles"].pop(),
            "review preparation invalid",
        ),
    ],
)
def test_preparation_rejects_integrity_or_authority_mutation(
    mutation: object, message: str
) -> None:
    payload = deepcopy(_json(CONTRACT / "review-preparation.json"))
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        _validate(payload)


def test_preparation_rejects_wrong_candidate_or_expired_snapshot() -> None:
    payload = _json(CONTRACT / "review-preparation.json")
    with pytest.raises(SamplingHarmReviewPreparationError, match="unexpected candidate"):
        validate_sampling_harm_review_preparation(
            payload,
            repository_root=ROOT,
            expected_candidate_commit="f" * 40,
            expected_package_commit=PACKAGE,
            now=datetime(2026, 8, 3, tzinfo=UTC),
        )
    with pytest.raises(SamplingHarmReviewPreparationError, match="snapshot is expired"):
        validate_sampling_harm_review_preparation(
            payload,
            repository_root=ROOT,
            expected_candidate_commit=CANDIDATE,
            expected_package_commit=PACKAGE,
            now=datetime(2026, 8, 10, tzinfo=UTC),
        )


def test_preparation_is_not_a_complete_scientific_review_bundle() -> None:
    manifest = _json(CONTRACT / "review-artifact-manifest.json")
    packet = _json(CONTRACT / "review-packet.json")
    incomplete = {
        "schema_version": "1.1.0",
        "expected_finding_ids": [],
        "expected_disagreement_ids": [],
        "evidence": {"artifact-manifest": manifest, "review-packet": packet},
    }
    with pytest.raises(ScientificReviewEvidenceError):
        validate_scientific_review_bundle(incomplete, repository_root=ROOT)
