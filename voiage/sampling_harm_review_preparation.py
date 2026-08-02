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
from referencing import Registry, Resource

from voiage.scientific_review_evidence import canonical_json_sha256

CONTRACT_ROOT = PurePosixPath("specs/frontier/sampling-acquisition-harm/v1")
ENVELOPE_SCHEMA = CONTRACT_ROOT / "schemas/review-preparation.schema.json"
CANDIDATE_PATH = CONTRACT_ROOT / "review-candidate.json"
SNAPSHOT_PATH = CONTRACT_ROOT / "governance-snapshot.json"
ENVELOPE_PATH = CONTRACT_ROOT / "review-preparation.json"
MANIFEST_PATH = CONTRACT_ROOT / "review-artifact-manifest.json"
PACKET_PATH = CONTRACT_ROOT / "review-packet.json"
FROZEN_CANDIDATE_COMMIT = "8d6c67879050f161258ed95d878a72e2bb6b22dd"
TRUSTED_PACKAGE_COMMIT = "d00e0e20752f44c52581dbb7ee45ce27c9b7d6dd"
EXPECTED_INVENTORY_SHA256 = (
    "7a3bafee13b44d21f095c528bec64472a13b81acdab20aa6b439d8d0cbe90a6e"
)
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


def _validate_json(
    value: object,
    schema: dict[str, Any],
    *,
    label: str,
    registry: Registry[Any] | None = None,
) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(
            schema, registry=registry or Registry(), format_checker=FormatChecker()
        ).validate(value)
    except JsonSchemaValidationError as error:
        location = "/".join(str(item) for item in error.absolute_path) or "$"
        raise SamplingHarmReviewPreparationError(
            f"{label} invalid at {location}: {error.message}"
        ) from error


def _frozen_schema_registry(root: Path, commit: str) -> Registry[Any]:
    common_path = PurePosixPath(
        "specs/frontier/governance/scientific-review/v1/schemas/common.schema.json"
    )
    common = _git_json(root, commit, common_path)
    return Registry().with_resource(str(common["$id"]), Resource.from_contents(common))


def validate_sampling_harm_review_preparation(
    envelope: Mapping[str, Any],
    *,
    repository_root: Path,
    expected_candidate_commit: str,
    expected_package_commit: str,
    now: datetime | None = None,
) -> dict[str, str]:
    """Validate a preparation package against exact frozen Git-tree bytes."""
    root = repository_root.resolve()
    if expected_candidate_commit != FROZEN_CANDIDATE_COMMIT:
        raise SamplingHarmReviewPreparationError("unexpected candidate commit")
    if expected_package_commit != TRUSTED_PACKAGE_COMMIT:
        raise SamplingHarmReviewPreparationError("unexpected package commit")
    if _git_output(
        root,
        "merge-base",
        "--is-ancestor",
        expected_candidate_commit,
        expected_package_commit,
    ):
        pass

    frozen_envelope = _git_json(root, expected_package_commit, ENVELOPE_PATH)
    if dict(envelope) != frozen_envelope:
        raise SamplingHarmReviewPreparationError(
            "working envelope differs from trusted package commit"
        )
    schema = _git_json(root, expected_package_commit, ENVELOPE_SCHEMA)
    _validate_json(envelope, schema, label="review preparation")
    dirty = _git_output(
        root,
        "status",
        "--porcelain",
        "--",
        str(ENVELOPE_PATH),
        str(MANIFEST_PATH),
        str(PACKET_PATH),
    ).decode()
    if dirty:
        raise SamplingHarmReviewPreparationError(
            "canonical packaging artifacts have working-tree substitutions"
        )

    commit = str(envelope["candidate_commit"]["value"])
    tree = str(envelope["candidate_tree"]["value"])
    if commit != expected_candidate_commit:
        raise SamplingHarmReviewPreparationError("unexpected candidate commit")
    actual_tree = _git_output(root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if actual_tree != tree:
        raise SamplingHarmReviewPreparationError("candidate tree does not match commit")

    manifest_ref = envelope["artifact_manifest"]
    packet_ref = envelope["review_packet"]
    manifest_path = _safe_path(manifest_ref["path"])
    packet_path = _safe_path(packet_ref["path"])
    if manifest_path != MANIFEST_PATH or packet_path != PACKET_PATH:
        raise SamplingHarmReviewPreparationError("canonical packaging path mismatch")
    manifest = _git_json(root, expected_package_commit, manifest_path)
    packet = _git_json(root, expected_package_commit, packet_path)
    schema_root = PurePosixPath(
        "specs/frontier/governance/scientific-review/v1/schemas"
    )
    registry = _frozen_schema_registry(root, commit)
    _validate_json(
        manifest,
        _git_json(root, commit, schema_root / "artifact-manifest.schema.json"),
        label="artifact manifest",
        registry=registry,
    )
    _validate_json(
        packet,
        _git_json(root, commit, schema_root / "review-packet.schema.json"),
        label="review packet",
        registry=registry,
    )

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
    inventory_digest = canonical_json_sha256(
        {
            "artifacts": sorted(
                ({"path": item["path"], "role": item["role"]} for item in artifacts),
                key=lambda item: item["path"],
            )
        }
    )
    if inventory_digest != EXPECTED_INVENTORY_SHA256:
        raise SamplingHarmReviewPreparationError("required path and role inventory mismatch")
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

    candidate_contracts = {
        "capabilities.json": "capability.schema.json",
        "estimand-boundary.json": "estimand-boundary.schema.json",
        "governance-snapshot.json": "governance-snapshot.schema.json",
        "prior-findings.json": "prior-findings.schema.json",
        "research-disposition.json": "research-disposition.schema.json",
        "review-candidate.json": "review-candidate.schema.json",
        "scope-selection.json": "scope-selection.schema.json",
        "source-and-retrieval-register.json": (
            "source-and-retrieval-register.schema.json"
        ),
    }
    frozen: dict[str, dict[str, Any]] = {}
    for artifact_name, schema_name in candidate_contracts.items():
        artifact_path = CONTRACT_ROOT / artifact_name
        artifact = _git_json(root, commit, artifact_path)
        artifact_schema = _git_json(root, commit, CONTRACT_ROOT / "schemas" / schema_name)
        _validate_json(artifact, artifact_schema, label=artifact_name)
        frozen[artifact_name] = artifact

    candidate = frozen["review-candidate.json"]
    snapshot = frozen["governance-snapshot.json"]
    boundary = frozen["estimand-boundary.json"]
    sources = frozen["source-and-retrieval-register.json"]
    findings = frozen["prior-findings.json"]
    capabilities = frozen["capabilities.json"]
    disposition = frozen["research-disposition.json"]
    selection = frozen["scope-selection.json"]
    if candidate["scope"]["scientific_disposition"] != "pending":
        raise SamplingHarmReviewPreparationError("candidate disposition is not pending")
    if candidate["scope"]["runtime_available"] is not False:
        raise SamplingHarmReviewPreparationError("candidate unexpectedly claims runtime")
    if tuple(candidate["required_independent_review_roles"]) != REQUIRED_ROLES:
        raise SamplingHarmReviewPreparationError("candidate reviewer roles are incomplete")
    if tuple(envelope["required_independent_review_roles"]) != REQUIRED_ROLES:
        raise SamplingHarmReviewPreparationError("preparation reviewer roles are incomplete")
    if candidate["next_tasks"] != envelope["pending_tasks"]:
        raise SamplingHarmReviewPreparationError("candidate pending-task binding mismatch")
    if candidate["adjacent_methods_not_aliased"] != [570, 571, 595, 598]:
        raise SamplingHarmReviewPreparationError("adjacent-method boundary mismatch")
    preserved = [
        "candidate_specific_non_authorizing_scalar_with_declared_commensurate_ledger",
        "parameterized_non_authorizing_constrained_candidate",
        "parameterized_non_authorizing_vector_candidate",
    ]
    if candidate["preserved_candidate_classes"] != preserved:
        raise SamplingHarmReviewPreparationError("preserved candidate classes mismatch")
    if boundary["preserved_research"] != preserved:
        raise SamplingHarmReviewPreparationError("estimand preservation boundary mismatch")
    if boundary["scientific_disposition"] != "pending" or boundary["runtime_available"]:
        raise SamplingHarmReviewPreparationError("estimand boundary claims completion")
    if sources["exact_source_review_status"] != envelope["source_review_status"]:
        raise SamplingHarmReviewPreparationError("source-review status mismatch")
    if findings["candidate_bound_independent_verification"] is not False:
        raise SamplingHarmReviewPreparationError("prior findings claim independent review")
    if capabilities["runtime_available"] or capabilities["stable_claim_allowed"]:
        raise SamplingHarmReviewPreparationError("capability unexpectedly claims maturity")
    if disposition["runtime_prohibited"] is not True:
        raise SamplingHarmReviewPreparationError("research disposition permits runtime")
    if selection["scientific_disposition"] != "pending":
        raise SamplingHarmReviewPreparationError("scope selection claims disposition")
    if snapshot["authority_boundary"]["scientific_review_completed"] is not False:
        raise SamplingHarmReviewPreparationError("snapshot claims scientific review completion")
    false_authority_groups = (
        candidate["authority_boundary"],
        snapshot["authority_boundary"],
        sources["authority_boundary"],
        envelope["authority_boundary"],
    )
    for group in false_authority_groups:
        for key, value in group.items():
            if key == "preparation_only":
                continue
            if value is not False:
                raise SamplingHarmReviewPreparationError(
                    f"authority boundary is not false: {key}"
                )

    expires = datetime.fromisoformat(snapshot["expires_at"])
    if snapshot_ref["expires_at"] != snapshot["expires_at"]:
        raise SamplingHarmReviewPreparationError("snapshot expiry binding mismatch")
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise SamplingHarmReviewPreparationError("validation time must be timezone-aware")
    if current > expires:
        raise SamplingHarmReviewPreparationError("governance snapshot is expired")

    return {
        "package_commit": expected_package_commit,
        "candidate_commit": commit,
        "candidate_tree": tree,
        "manifest_sha256": manifest_digest,
        "packet_sha256": packet_digest,
    }


def load_and_validate_sampling_harm_review_preparation(
    envelope_path: Path,
    *,
    repository_root: Path,
    expected_candidate_commit: str,
    expected_package_commit: str,
    now: datetime | None = None,
) -> dict[str, str]:
    """Load and validate one review-preparation envelope."""
    envelope = _load_json(envelope_path)
    return validate_sampling_harm_review_preparation(
        envelope,
        repository_root=repository_root,
        expected_candidate_commit=expected_candidate_commit,
        expected_package_commit=expected_package_commit,
        now=now,
    )
