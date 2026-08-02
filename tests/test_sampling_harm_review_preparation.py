"""Fail-closed tests for the H8-C frozen review preparation."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

import voiage.sampling_harm_review_preparation as preparation_module
from voiage.sampling_harm_review_preparation import (
    SamplingHarmReviewPreparationError,
    validate_sampling_harm_review_preparation,
)
from voiage.scientific_review_evidence import (
    ScientificReviewEvidenceError,
    canonical_json_sha256,
    validate_scientific_review_bundle,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/sampling-acquisition-harm/v1"
CANDIDATE = "8d6c67879050f161258ed95d878a72e2bb6b22dd"
PACKAGE = "d00e0e20752f44c52581dbb7ee45ce27c9b7d6dd"


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
            "trusted package commit",
        ),
        (
            lambda item: item["artifact_manifest"].__setitem__("sha256", "f" * 64),
            "trusted package commit",
        ),
        (
            lambda item: item["review_packet"].__setitem__("path", "../packet.json"),
            "trusted package commit",
        ),
        (
            lambda item: item["authority_boundary"].__setitem__(
                "scientific_review_completed", True
            ),
            "trusted package commit",
        ),
        (
            lambda item: item["required_independent_review_roles"].pop(),
            "trusted package commit",
        ),
    ],
)
def test_preparation_rejects_integrity_or_authority_mutation(
    mutation: Callable[[dict[str, Any]], object], message: str
) -> None:
    payload = deepcopy(_json(CONTRACT / "review-preparation.json"))
    mutation(payload)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        _validate(payload)


def test_preparation_rejects_wrong_candidate_or_expired_snapshot() -> None:
    payload = _json(CONTRACT / "review-preparation.json")
    with pytest.raises(
        SamplingHarmReviewPreparationError, match="unexpected candidate"
    ):
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


def test_self_consistent_package_substitution_still_fails_exact_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    manifest = deepcopy(_json(CONTRACT / "review-artifact-manifest.json"))
    packet = deepcopy(_json(CONTRACT / "review-packet.json"))
    manifest["artifacts"][0]["role"] = "substituted_role"
    manifest["manifest_sha256"] = canonical_json_sha256(
        manifest, excluded_json_pointers={"/manifest_sha256"}
    )
    packet["artifact_manifest_sha256"] = manifest["manifest_sha256"]
    packet["packet_sha256"] = canonical_json_sha256(
        packet, excluded_json_pointers={"/packet_sha256"}
    )
    envelope["artifact_manifest"]["sha256"] = manifest["manifest_sha256"]
    envelope["review_packet"]["sha256"] = packet["packet_sha256"]

    original = preparation_module._git_json

    def fake_git_json(root: Path, commit: str, path: object) -> dict[str, Any]:
        if commit == PACKAGE:
            if str(path) == str(preparation_module.ENVELOPE_PATH):
                return envelope
            if str(path) == str(preparation_module.MANIFEST_PATH):
                return manifest
            if str(path) == str(preparation_module.PACKET_PATH):
                return packet
        return original(root, commit, path)  # type: ignore[arg-type]

    monkeypatch.setattr(preparation_module, "_git_json", fake_git_json)
    with pytest.raises(
        SamplingHarmReviewPreparationError, match="path and role inventory"
    ):
        _validate(envelope)


def test_self_consistent_artifact_hash_substitution_fails_frozen_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    manifest = deepcopy(_json(CONTRACT / "review-artifact-manifest.json"))
    packet = deepcopy(_json(CONTRACT / "review-packet.json"))
    manifest["artifacts"][0]["sha256"] = "f" * 64
    manifest["manifest_sha256"] = canonical_json_sha256(
        manifest, excluded_json_pointers={"/manifest_sha256"}
    )
    packet["artifact_manifest_sha256"] = manifest["manifest_sha256"]
    packet["packet_sha256"] = canonical_json_sha256(
        packet, excluded_json_pointers={"/packet_sha256"}
    )
    envelope["artifact_manifest"]["sha256"] = manifest["manifest_sha256"]
    envelope["review_packet"]["sha256"] = packet["packet_sha256"]
    original = preparation_module._git_json

    def fake_git_json(root: Path, commit: str, path: object) -> dict[str, Any]:
        if commit == PACKAGE:
            replacements = {
                str(preparation_module.ENVELOPE_PATH): envelope,
                str(preparation_module.MANIFEST_PATH): manifest,
                str(preparation_module.PACKET_PATH): packet,
            }
            if str(path) in replacements:
                return replacements[str(path)]
        return original(root, commit, path)  # type: ignore[arg-type]

    monkeypatch.setattr(preparation_module, "_git_json", fake_git_json)
    with pytest.raises(
        SamplingHarmReviewPreparationError, match="frozen artifact bytes"
    ):
        _validate(envelope)


def test_preparation_rejects_dirty_packaging_or_naive_validation_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _json(CONTRACT / "review-preparation.json")
    original = preparation_module._git_output

    def dirty_git_output(root: Path, *args: str) -> bytes:
        if args and args[0] == "status":
            return (
                b" M specs/frontier/sampling-acquisition-harm/v1/review-packet.json\n"
            )
        return original(root, *args)

    monkeypatch.setattr(preparation_module, "_git_output", dirty_git_output)
    with pytest.raises(SamplingHarmReviewPreparationError, match="substitutions"):
        _validate(envelope)
    monkeypatch.setattr(preparation_module, "_git_output", original)
    with pytest.raises(SamplingHarmReviewPreparationError, match="timezone-aware"):
        validate_sampling_harm_review_preparation(
            envelope,
            repository_root=ROOT,
            expected_candidate_commit=CANDIDATE,
            expected_package_commit=PACKAGE,
            now=datetime(2026, 8, 3),
        )


def test_canonical_cli_emits_receipt_and_rejects_wrong_package(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_sampling_harm_review_preparation.py"),
        str(CONTRACT / "review-preparation.json"),
        "--repository-root",
        str(ROOT),
        "--expected-candidate-commit",
        CANDIDATE,
        "--expected-package-commit",
        PACKAGE,
    ]
    valid = subprocess.run(command, check=False, capture_output=True, text=True)
    assert valid.returncode == 0, valid.stderr
    receipt = json.loads(valid.stdout)
    assert receipt["status"] == "valid"
    assert receipt["candidate_commit"] == CANDIDATE
    assert receipt["package_commit"] == PACKAGE

    invalid = subprocess.run(
        [*command[:-1], "f" * 40], check=False, capture_output=True, text=True
    )
    assert invalid.returncode != 0
    assert "unexpected package commit" in invalid.stderr

    reserialized = tmp_path / "review-preparation.json"
    reserialized.write_text(
        json.dumps(_json(CONTRACT / "review-preparation.json")), encoding="utf-8"
    )
    substituted = subprocess.run(
        [sys.executable, command[1], str(reserialized), *command[3:]],
        check=False,
        capture_output=True,
        text=True,
    )
    assert substituted.returncode != 0
    assert "canonical package path" in substituted.stderr
