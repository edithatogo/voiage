"""Fail-closed tests for the H8-C frozen review preparation."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

import voiage.sampling_harm_review_preparation as preparation_module
from voiage.scientific_review_evidence import (
    ScientificReviewEvidenceError,
    canonical_json_sha256,
    validate_scientific_review_bundle,
)

SamplingHarmReviewPreparationError = (
    preparation_module.SamplingHarmReviewPreparationError
)
validate_sampling_harm_review_preparation = (
    preparation_module.validate_sampling_harm_review_preparation
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


def _patch_git_json(
    monkeypatch: pytest.MonkeyPatch,
    replacements: dict[tuple[str, str], dict[str, Any]],
) -> None:
    original = preparation_module._git_json

    def fake_git_json(root: Path, commit: str, path: object) -> dict[str, Any]:
        replacement = replacements.get((commit, str(path)))
        if replacement is not None:
            return replacement
        return original(root, commit, path)  # type: ignore[arg-type]

    monkeypatch.setattr(preparation_module, "_git_json", fake_git_json)


def _disable_schema_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preparation_module, "_validate_json", lambda *args, **kwargs: None
    )


def _candidate_artifact_path(name: str) -> str:
    return str(preparation_module.CONTRACT_ROOT / name)


def _rebind_package(
    envelope: dict[str, Any], manifest: dict[str, Any], packet: dict[str, Any]
) -> None:
    manifest["manifest_sha256"] = canonical_json_sha256(
        manifest, excluded_json_pointers={"/manifest_sha256"}
    )
    packet["artifact_manifest_sha256"] = manifest["manifest_sha256"]
    packet["packet_sha256"] = canonical_json_sha256(
        packet, excluded_json_pointers={"/packet_sha256"}
    )
    envelope["artifact_manifest"]["sha256"] = manifest["manifest_sha256"]
    envelope["review_packet"]["sha256"] = packet["packet_sha256"]


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


def test_preparation_rejects_wrong_package() -> None:
    with pytest.raises(SamplingHarmReviewPreparationError, match="unexpected package"):
        validate_sampling_harm_review_preparation(
            _json(CONTRACT / "review-preparation.json"),
            repository_root=ROOT,
            expected_candidate_commit=CANDIDATE,
            expected_package_commit="f" * 40,
            now=datetime(2026, 8, 3, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("artifact_name", "mutation", "message"),
    [
        (
            "review-candidate.json",
            lambda item: item["scope"].__setitem__(
                "scientific_disposition", "reviewed"
            ),
            "disposition is not pending",
        ),
        (
            "review-candidate.json",
            lambda item: item["scope"].__setitem__("runtime_available", True),
            "unexpectedly claims runtime",
        ),
        (
            "review-candidate.json",
            lambda item: item["required_independent_review_roles"].pop(),
            "reviewer roles are incomplete",
        ),
        (
            "review-candidate.json",
            lambda item: item["next_tasks"].pop(),
            "pending-task binding mismatch",
        ),
        (
            "review-candidate.json",
            lambda item: item["adjacent_methods_not_aliased"].pop(),
            "adjacent-method boundary mismatch",
        ),
        (
            "review-candidate.json",
            lambda item: item["preserved_candidate_classes"].pop(),
            "preserved candidate classes mismatch",
        ),
        (
            "estimand-boundary.json",
            lambda item: item["preserved_research"].pop(),
            "estimand preservation boundary mismatch",
        ),
        (
            "estimand-boundary.json",
            lambda item: item.__setitem__("scientific_disposition", "reviewed"),
            "estimand boundary claims completion",
        ),
        (
            "estimand-boundary.json",
            lambda item: item.__setitem__("runtime_available", True),
            "estimand boundary claims completion",
        ),
        (
            "source-and-retrieval-register.json",
            lambda item: item.__setitem__("exact_source_review_status", "complete"),
            "source-review status mismatch",
        ),
        (
            "prior-findings.json",
            lambda item: item.__setitem__(
                "candidate_bound_independent_verification", True
            ),
            "claim independent review",
        ),
        (
            "capabilities.json",
            lambda item: item.__setitem__("runtime_available", True),
            "unexpectedly claims maturity",
        ),
        (
            "capabilities.json",
            lambda item: item.__setitem__("stable_claim_allowed", True),
            "unexpectedly claims maturity",
        ),
        (
            "research-disposition.json",
            lambda item: item.__setitem__("runtime_prohibited", False),
            "permits runtime",
        ),
        (
            "scope-selection.json",
            lambda item: item.__setitem__("scientific_disposition", "reviewed"),
            "claims disposition",
        ),
        (
            "governance-snapshot.json",
            lambda item: item["authority_boundary"].__setitem__(
                "scientific_review_completed", True
            ),
            "claims scientific review completion",
        ),
        (
            "review-candidate.json",
            lambda item: item["authority_boundary"].__setitem__(
                "release_authorized", True
            ),
            "authority boundary is not false",
        ),
        (
            "source-and-retrieval-register.json",
            lambda item: item["authority_boundary"].__setitem__(
                "release_authorized", True
            ),
            "authority boundary is not false",
        ),
        (
            "governance-snapshot.json",
            lambda item: item["authority_boundary"].__setitem__(
                "release_authorized", True
            ),
            "authority boundary is not false",
        ),
    ],
)
def test_candidate_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    mutation: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    artifact = deepcopy(_json(CONTRACT / artifact_name))
    mutation(artifact)
    _patch_git_json(
        monkeypatch,
        {(CANDIDATE, _candidate_artifact_path(artifact_name)): artifact},
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        _validate(_json(CONTRACT / "review-preparation.json"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: item["required_independent_review_roles"].pop(),
            "preparation reviewer roles are incomplete",
        ),
        (
            lambda item: item["authority_boundary"].__setitem__(
                "release_authorized", True
            ),
            "authority boundary is not false",
        ),
    ],
)
def test_envelope_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    mutation(envelope)
    _patch_git_json(
        monkeypatch,
        {(PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope},
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        _validate(envelope)


def test_snapshot_expiry_binding_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = deepcopy(_json(CONTRACT / "governance-snapshot.json"))
    snapshot["expires_at"] = "2026-08-11T00:00:00Z"
    _patch_git_json(
        monkeypatch,
        {
            (
                CANDIDATE,
                _candidate_artifact_path("governance-snapshot.json"),
            ): snapshot
        },
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match="expiry binding"):
        _validate(_json(CONTRACT / "review-preparation.json"))


@pytest.mark.parametrize(
    ("target", "field", "message"),
    [
        ("manifest", "candidate_commit", "candidate commit binding mismatch"),
        ("manifest", "candidate_tree", "candidate tree binding mismatch"),
        ("packet", "candidate_commit", "candidate commit binding mismatch"),
        ("packet", "candidate_tree", "candidate tree binding mismatch"),
    ],
)
def test_package_candidate_bindings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    message: str,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    manifest = deepcopy(_json(CONTRACT / "review-artifact-manifest.json"))
    packet = deepcopy(_json(CONTRACT / "review-packet.json"))
    value = manifest if target == "manifest" else packet
    value[field]["value"] = "f" * 40
    _rebind_package(envelope, manifest, packet)
    _patch_git_json(
        monkeypatch,
        {
            (PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope,
            (PACKAGE, str(preparation_module.MANIFEST_PATH)): manifest,
            (PACKAGE, str(preparation_module.PACKET_PATH)): packet,
        },
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        _validate(envelope)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda envelope, manifest, packet: manifest.__setitem__(
                "manifest_sha256", "f" * 64
            ),
            "manifest canonical digest mismatch",
        ),
        (
            lambda envelope, manifest, packet: envelope[
                "artifact_manifest"
            ].__setitem__("sha256", "f" * 64),
            "manifest canonical digest mismatch",
        ),
        (
            lambda envelope, manifest, packet: packet.__setitem__(
                "packet_sha256", "f" * 64
            ),
            "packet canonical digest mismatch",
        ),
        (
            lambda envelope, manifest, packet: envelope["review_packet"].__setitem__(
                "sha256", "f" * 64
            ),
            "packet canonical digest mismatch",
        ),
    ],
)
def test_package_digest_mismatches_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], object],
    message: str,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    manifest = deepcopy(_json(CONTRACT / "review-artifact-manifest.json"))
    packet = deepcopy(_json(CONTRACT / "review-packet.json"))
    mutation(envelope, manifest, packet)
    _patch_git_json(
        monkeypatch,
        {
            (PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope,
            (PACKAGE, str(preparation_module.MANIFEST_PATH)): manifest,
            (PACKAGE, str(preparation_module.PACKET_PATH)): packet,
        },
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        _validate(envelope)


def test_packet_manifest_binding_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    manifest = deepcopy(_json(CONTRACT / "review-artifact-manifest.json"))
    packet = deepcopy(_json(CONTRACT / "review-packet.json"))
    packet["artifact_manifest_sha256"] = "f" * 64
    packet["packet_sha256"] = canonical_json_sha256(
        packet, excluded_json_pointers={"/packet_sha256"}
    )
    envelope["review_packet"]["sha256"] = packet["packet_sha256"]
    _patch_git_json(
        monkeypatch,
        {
            (PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope,
            (PACKAGE, str(preparation_module.MANIFEST_PATH)): manifest,
            (PACKAGE, str(preparation_module.PACKET_PATH)): packet,
        },
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match="does not bind"):
        _validate(envelope)


def test_duplicate_manifest_path_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    manifest = deepcopy(_json(CONTRACT / "review-artifact-manifest.json"))
    packet = deepcopy(_json(CONTRACT / "review-packet.json"))
    manifest["artifacts"][1]["path"] = manifest["artifacts"][0]["path"]
    _rebind_package(envelope, manifest, packet)
    _patch_git_json(
        monkeypatch,
        {
            (PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope,
            (PACKAGE, str(preparation_module.MANIFEST_PATH)): manifest,
            (PACKAGE, str(preparation_module.PACKET_PATH)): packet,
        },
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match="must be unique"):
        _validate(envelope)


@pytest.mark.parametrize("reference", ["candidate_input", "governance_snapshot"])
def test_envelope_artifact_reference_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    envelope[reference]["sha256"] = "f" * 64
    _patch_git_json(
        monkeypatch,
        {(PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope},
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match="artifact binding"):
        _validate(envelope)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("candidate_commit", "unexpected candidate commit"),
        ("candidate_tree", "candidate tree does not match commit"),
    ],
)
def test_embedded_candidate_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    envelope[field]["value"] = "f" * 40
    _patch_git_json(
        monkeypatch,
        {(PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope},
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        _validate(envelope)


def test_manifest_artifact_count_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    envelope["artifact_manifest"]["artifact_count"] -= 1
    _patch_git_json(
        monkeypatch,
        {(PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope},
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match="artifact count"):
        _validate(envelope)


@pytest.mark.parametrize("reference", ["artifact_manifest", "review_packet"])
def test_noncanonical_packaging_path_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
) -> None:
    envelope = deepcopy(_json(CONTRACT / "review-preparation.json"))
    original_path = envelope[reference]["path"]
    alternate_path = f"{original_path}.alternate"
    envelope[reference]["path"] = alternate_path
    replacement = (
        deepcopy(_json(CONTRACT / "review-artifact-manifest.json"))
        if reference == "artifact_manifest"
        else deepcopy(_json(CONTRACT / "review-packet.json"))
    )
    _patch_git_json(
        monkeypatch,
        {
            (PACKAGE, str(preparation_module.ENVELOPE_PATH)): envelope,
            (PACKAGE, alternate_path): replacement,
        },
    )
    _disable_schema_validation(monkeypatch)
    with pytest.raises(SamplingHarmReviewPreparationError, match="packaging path"):
        _validate(envelope)


def test_schema_validation_failure_is_normalized() -> None:
    with pytest.raises(
        SamplingHarmReviewPreparationError, match=r"example invalid at x"
    ):
        preparation_module._validate_json(
            {"x": 1},
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
            label="example",
        )


@pytest.mark.parametrize("value", [None, "/absolute", "../parent", "a//b"])
def test_safe_path_rejects_invalid_values(value: object) -> None:
    with pytest.raises(SamplingHarmReviewPreparationError, match="artifact path"):
        preparation_module._safe_path(value)


def test_git_output_rejects_unavailable_or_failed_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preparation_module, "GIT_EXECUTABLE", None)
    with pytest.raises(SamplingHarmReviewPreparationError, match="git is unavailable"):
        preparation_module._git_output(ROOT, "status")
    monkeypatch.setattr(preparation_module, "GIT_EXECUTABLE", "git")
    monkeypatch.setattr(
        preparation_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr=b"expected failure", stdout=b""
        ),
    )
    with pytest.raises(SamplingHarmReviewPreparationError, match="expected failure"):
        preparation_module._git_output(ROOT, "status")


@pytest.mark.parametrize(
    ("payload", "message"),
    [(b"not-json", "not valid JSON"), (b"[]", "not an object")],
)
def test_git_json_rejects_invalid_frozen_artifact(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(preparation_module, "_git_output", lambda *args: payload)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        preparation_module._git_json(ROOT, CANDIDATE, preparation_module.CANDIDATE_PATH)


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


@pytest.mark.parametrize(
    ("candidate", "package", "message"),
    [
        ("f" * 40, PACKAGE, "unexpected candidate"),
        (CANDIDATE, "f" * 40, "unexpected package"),
    ],
)
def test_loader_rejects_wrong_pins(
    candidate: str,
    package: str,
    message: str,
) -> None:
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        preparation_module.load_and_validate_sampling_harm_review_preparation(
            CONTRACT / "review-preparation.json",
            repository_root=ROOT,
            expected_candidate_commit=candidate,
            expected_package_commit=package,
        )


def test_loader_accepts_canonical_frozen_package() -> None:
    receipt = preparation_module.load_and_validate_sampling_harm_review_preparation(
        CONTRACT / "review-preparation.json",
        repository_root=ROOT,
        expected_candidate_commit=CANDIDATE,
        expected_package_commit=PACKAGE,
        now=datetime(2026, 8, 3, tzinfo=UTC),
    )
    assert receipt["candidate_commit"] == CANDIDATE
    assert receipt["package_commit"] == PACKAGE


def test_loader_rejects_noncanonical_path(tmp_path: Path) -> None:
    alternate = tmp_path / "review-preparation.json"
    alternate.write_bytes((CONTRACT / "review-preparation.json").read_bytes())
    with pytest.raises(
        SamplingHarmReviewPreparationError, match="canonical package path"
    ):
        preparation_module.load_and_validate_sampling_harm_review_preparation(
            alternate,
            repository_root=ROOT,
            expected_candidate_commit=CANDIDATE,
            expected_package_commit=PACKAGE,
        )


def test_loader_normalizes_read_failure() -> None:
    canonical = (ROOT / preparation_module.ENVELOPE_PATH).resolve()

    class UnreadableCanonicalPath:
        def resolve(self) -> Path:
            return canonical

        def read_bytes(self) -> bytes:
            raise OSError("expected read failure")

    with pytest.raises(
        SamplingHarmReviewPreparationError, match="expected read failure"
    ):
        preparation_module.load_and_validate_sampling_harm_review_preparation(  # type: ignore[arg-type]
            UnreadableCanonicalPath(),
            repository_root=ROOT,
            expected_candidate_commit=CANDIDATE,
            expected_package_commit=PACKAGE,
        )


def test_loader_rejects_raw_byte_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preparation_module, "_git_output", lambda *args: b"different")
    with pytest.raises(SamplingHarmReviewPreparationError, match="bytes differ"):
        preparation_module.load_and_validate_sampling_harm_review_preparation(
            CONTRACT / "review-preparation.json",
            repository_root=ROOT,
            expected_candidate_commit=CANDIDATE,
            expected_package_commit=PACKAGE,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [(b"not-json", "cannot load"), (b"[]", "must contain an object")],
)
def test_loader_rejects_invalid_frozen_json(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda self: payload)
    monkeypatch.setattr(preparation_module, "_git_output", lambda *args: payload)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        preparation_module.load_and_validate_sampling_harm_review_preparation(
            CONTRACT / "review-preparation.json",
            repository_root=ROOT,
            expected_candidate_commit=CANDIDATE,
            expected_package_commit=PACKAGE,
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
