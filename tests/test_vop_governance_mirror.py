"""Conformance tests for the vendored VOP governance schemas."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re

from jsonschema import Draft202012Validator
import pytest

from voiage.contracts.concerns import ConcernSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
MIRROR_ROOT = REPO_ROOT / "specs/integration/vop-voiage/governance"
SCHEMA_ROOT = MIRROR_ROOT / "schemas"
DEFAULT_CANONICAL_ROOT = Path("C:/repos/vop_poc_nz-codex-v7/schemas/governance")
CANONICAL_ROOT = Path(
    os.environ.get("VOP_GOVERNANCE_SCHEMA_ROOT", DEFAULT_CANONICAL_ROOT)
)


def _manifest() -> dict[str, object]:
    return json.loads((MIRROR_ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))


def test_vendored_files_match_manifest_and_canonical_bytes() -> None:
    manifest = _manifest()
    expected = manifest["files"]
    assert isinstance(expected, dict)
    assert {path.name for path in SCHEMA_ROOT.glob("*.json")} == set(expected)

    for filename, digest in expected.items():
        mirror = SCHEMA_ROOT / filename
        content = mirror.read_bytes()
        assert sha256(content).hexdigest() == digest
        if CANONICAL_ROOT.exists():
            assert content == (CANONICAL_ROOT / filename).read_bytes()


def test_schema_and_example_are_draft_2020_12_valid() -> None:
    for path in sorted(SCHEMA_ROOT.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)

    concern_schema = json.loads(
        (SCHEMA_ROOT / "concern.schema.json").read_text(encoding="utf-8")
    )
    example = json.loads(
        (MIRROR_ROOT / "examples/concern.example.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(concern_schema).validate(example)
    runtime = ConcernSpec.model_validate(example)
    Draft202012Validator(concern_schema).validate(runtime.model_dump(mode="json"))


def test_privacy_and_stable_marker_policy_fail_closed() -> None:
    policy = _manifest()["github_projection_policy"]
    assert isinstance(policy, dict)
    marker = "vop-voiage-governance-id:RSK-SHR-0001"
    assert re.fullmatch(str(policy["stable_marker_regex"]), marker)
    assert policy["excluded_visibilities"] == ["local_private"]
    assert policy["network_mutation"] is False
    assert policy["source_of_truth"] == "local_governance_ledger"

    evidence_schema = (SCHEMA_ROOT / "evidence-reference.schema.json").read_text(
        encoding="utf-8"
    )
    assert '"local_private"' in evidence_schema
    assert ".conductor/local" not in "\n".join(
        path.read_text(encoding="utf-8") for path in MIRROR_ROOT.rglob("*.json")
    )


@pytest.mark.skipif(CANONICAL_ROOT.exists(), reason="canonical sibling is available")
def test_missing_sibling_is_covered_by_pinned_digests() -> None:
    """Hosted CI remains deterministic when the sibling checkout is absent."""
    assert len(_manifest()["canonical_git_commit"]) == 40  # type: ignore[arg-type]
