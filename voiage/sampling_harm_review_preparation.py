"""Validation for an H8-C sampling-harm review preparation package."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess  # nosec B404
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from voiage.scientific_review_evidence import (
    ScientificReviewEvidenceError,
    canonical_json_sha256,
    validate_scientific_review_evidence,
)

CONTRACT_ROOT = Path("specs/frontier/sampling-acquisition-harm/v1")
ENVELOPE_SCHEMA = CONTRACT_ROOT / "schemas/review-preparation.schema.json"
CANDIDATE_PATH = CONTRACT_ROOT / "review-candidate.json"
SNAPSHOT_PATH = CONTRACT_ROOT / "governance-snapshot.json"
REQUIRED_ROLES = (
    "estimand_domain",
    "estimator_assurance",
    "cross_language_api",
    "governance_publication",
    "domain_specialist",
)
GIT_EXECUTABLE = shutil.which("git")


class SamplingHarmReviewPreparationError(ValueError):
    """Raised when a frozen review preparation is incomplete or stale."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SamplingHarmReviewPreparationError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise SamplingHarmReviewPreparationError(f"{path} must contain an object")
    return value


def _safe_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise SamplingHarmReviewPreparationError("artifact path must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise SamplingHarmReviewPreparationError(f"artifact path is unsafe: {value}")
    return path


def _git_output(root: Path, *args: str) -> bytes:
    if GIT_EXECUTABLE is None:
        raise SamplingHarmReviewPreparationError("git is unavailable")
    result = subprocess.run(  # noqa: S603  # nosec B603 -- fixed Git executable
        [GIT_EXECUTABLE, "-C", str(root), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SamplingHarmReviewPreparationError(f"git verification failed: {detail}")
    return result.stdout


def _git_json(root: Path, commit: str, path: PurePosixPath) -> dict[str, Any]:
    try:
        value = json.loads(_git_output(root, "show", f"{commit}:{path}").decode())
    except json.JSONDecodeError as error:
        raise SamplingHarmReviewPreparationError(
            f"frozen artifact is not valid JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise SamplingHarmReviewPreparationError(f"frozen artifact is not an object: {path}")
    return value


def validate_sampling_harm_review_preparation(
    envelope: Mapping[str, Any],
    *,
    repository_root: Path,
    expected_candidate_commit: str | None = None,
    now: datetime | None = None,
) -> None:
    """Validate a preparation package against exact frozen Git-tree bytes."""
    root = repository_root.resolve()
    schema = _load_json(root / ENVELOPE_SCHEMA)
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(envelope)
    except JsonSchemaValidationError as error:
        location = "/".join(str(item) for item in error.absolute_path) or "$"
        raise SamplingHarmReviewPreparationError(
            f"review preparation invalid at {location}: {error.message}"
        ) from error

    commit = str(envelope["candidate_commit"]["value"])
    tree = str(envelope["candidate_tree"]["value"])
    if expected_candidate_commit is not None and commit != expected_candidate_commit:
        raise SamplingHarmReviewPreparationError("unexpected candidate commit")
    actual_tree = _git_output(root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if actual_tree != tree:
        raise SamplingHarmReviewPreparationError("candidate tree does not match commit")

    manifest_ref = envelope["artifact_manifest"]
    packet_ref = envelope["review_packet"]
    manifest = _load_json(root / _safe_path(manifest_ref["path"]))
    packet = _load_json(root / _safe_path(packet_ref["path"]))
    try:
        validate_scientific_review_evidence("artifact-manifest", manifest)
        validate_scientific_review_evidence("review-packet", packet)
    except ScientificReviewEvidenceError as error:
        raise SamplingHarmReviewPreparationError(str(error)) from error

    manifest_digest = canonical_json_sha256(
        manifest, excluded_json_pointers={"/manifest_sha256"}
    )
    packet_digest = canonical_json_sha256(
        packet, excluded_json_pointers={"/packet_sha256"}
    )
    if manifest_digest != manifest["manifest_sha256"] or manifest_digest != manifest_ref["sha256"]:
        raise SamplingHarmReviewPreparationError("artifact manifest canonical digest mismatch")
    if packet_digest != packet["packet_sha256"] or packet_digest != packet_ref["sha256"]:
        raise SamplingHarmReviewPreparationError("review packet canonical digest mismatch")
    if packet["artifact_manifest_sha256"] != manifest_digest:
        raise SamplingHarmReviewPreparationError("packet does not bind artifact manifest")
    for value in (manifest, packet):
        if value["candidate_commit"] != envelope["candidate_commit"]:
            raise SamplingHarmReviewPreparationError("candidate commit binding mismatch")
        if value["candidate_tree"] != envelope["candidate_tree"]:
            raise SamplingHarmReviewPreparationError("candidate tree binding mismatch")

    artifacts = manifest["artifacts"]
    paths = [str(_safe_path(item["path"])) for item in artifacts]
    if len(paths) != len(set(paths)):
        raise SamplingHarmReviewPreparationError("manifest artifact paths must be unique")
    if len(paths) != manifest_ref["artifact_count"]:
        raise SamplingHarmReviewPreparationError("manifest artifact count mismatch")
    for item, path in zip(artifacts, paths, strict=True):
        content = _git_output(root, "show", f"{commit}:{path}")
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise SamplingHarmReviewPreparationError(
                f"frozen artifact bytes do not match manifest: {path}"
            )

    candidate_ref = envelope["candidate_input"]
    candidate_path = _safe_path(candidate_ref["path"])
    snapshot_ref = envelope["governance_snapshot"]
    snapshot_path = _safe_path(snapshot_ref["path"])
    manifest_by_path = {item["path"]: item["sha256"] for item in artifacts}
    for ref, path in ((candidate_ref, candidate_path), (snapshot_ref, snapshot_path)):
        if manifest_by_path.get(str(path)) != ref["sha256"]:
            raise SamplingHarmReviewPreparationError("envelope artifact binding mismatch")

    candidate = _git_json(root, commit, candidate_path)
    snapshot = _git_json(root, commit, snapshot_path)
    if candidate["scope"]["scientific_disposition"] != "pending":
        raise SamplingHarmReviewPreparationError("candidate disposition is not pending")
    if candidate["scope"]["runtime_available"] is not False:
        raise SamplingHarmReviewPreparationError("candidate unexpectedly claims runtime")
    if tuple(candidate["required_independent_review_roles"]) != REQUIRED_ROLES:
        raise SamplingHarmReviewPreparationError("candidate reviewer roles are incomplete")
    if tuple(envelope["required_independent_review_roles"]) != REQUIRED_ROLES:
        raise SamplingHarmReviewPreparationError("preparation reviewer roles are incomplete")
    if snapshot["authority_boundary"]["scientific_review_completed"] is not False:
        raise SamplingHarmReviewPreparationError("snapshot claims scientific review completion")

    expires = datetime.fromisoformat(snapshot["expires_at"])
    if snapshot_ref["expires_at"] != snapshot["expires_at"]:
        raise SamplingHarmReviewPreparationError("snapshot expiry binding mismatch")
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise SamplingHarmReviewPreparationError("validation time must be timezone-aware")
    if current > expires:
        raise SamplingHarmReviewPreparationError("governance snapshot is expired")


def load_and_validate_sampling_harm_review_preparation(
    envelope_path: Path,
    *,
    repository_root: Path,
    expected_candidate_commit: str | None = None,
    now: datetime | None = None,
) -> None:
    """Load and validate one review-preparation envelope."""
    envelope = _load_json(envelope_path)
    validate_sampling_harm_review_preparation(
        envelope,
        repository_root=repository_root,
        expected_candidate_commit=expected_candidate_commit,
        now=now,
    )
