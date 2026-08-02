from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import voiage.sampling_harm_automated_challenge as challenge_module
from voiage.scientific_review_evidence import canonical_json_sha256

SYNTHESIS_PATH = challenge_module.SYNTHESIS_PATH
SamplingHarmAutomatedChallengeError = (
    challenge_module.SamplingHarmAutomatedChallengeError
)
load_and_validate_sampling_harm_automated_challenge = (
    challenge_module.load_and_validate_sampling_harm_automated_challenge
)
validate_sampling_harm_automated_challenge = (
    challenge_module.validate_sampling_harm_automated_challenge
)

ROOT = Path(__file__).resolve().parents[1]


def _synthesis() -> dict[str, object]:
    return json.loads((ROOT / SYNTHESIS_PATH).read_bytes())


def _resign(value: dict[str, object]) -> dict[str, object]:
    value["synthesis_sha256"] = canonical_json_sha256(
        value, excluded_json_pointers={"/synthesis_sha256"}
    )
    return value


def test_canonical_automated_challenge_is_valid_and_non_authorizing() -> None:
    receipt = load_and_validate_sampling_harm_automated_challenge(
        ROOT / SYNTHESIS_PATH, repository_root=ROOT
    )
    assert receipt == {
        "synthesis_sha256": "29fdd684d356fba6d57d2dc164a2ba31c1d5e37d2318c8b484851ba649908820",
        "role_report_count": 5,
        "finding_count": 19,
        "h8d_satisfied": False,
        "h8e_satisfied": False,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(synthesis_sha256="f" * 64), "digest"),
        (
            lambda value: value["bindings"]["candidate_commit"].update(value="f" * 40),
            "candidate_commit",
        ),
        (
            lambda value: value["bindings"]["artifact_manifest"].update(
                sha256="f" * 64
            ),
            "artifact manifest",
        ),
        (
            lambda value: value["bindings"]["review_packet"].update(sha256="f" * 64),
            "review packet",
        ),
        (
            lambda value: value["role_reports"][0].update(report_sha256="f" * 64),
            "role-report digest",
        ),
        (
            lambda value: value["role_reports"][0].update(path="../outside.json"),
            "invalid",
        ),
        (
            lambda value: value["findings"].pop(),
            "invalid",
        ),
        (
            lambda value: value["findings"][0].update(disposition="fixed"),
            "invalid",
        ),
        (
            lambda value: value["gate_status"].update(h8d_satisfied=True),
            "invalid",
        ),
        (
            lambda value: value["authority"].update(runtime_authority=True),
            "invalid",
        ),
    ],
)
def test_automated_challenge_rejects_tampering(mutation, match: str) -> None:
    value = deepcopy(_synthesis())
    mutation(value)
    if value["synthesis_sha256"] != "f" * 64:
        _resign(value)
    with pytest.raises(SamplingHarmAutomatedChallengeError, match=match):
        validate_sampling_harm_automated_challenge(value, repository_root=ROOT)


def test_automated_challenge_cli_emits_non_authorizing_receipt() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_sampling_harm_automated_challenge.py"),
            str(ROOT / SYNTHESIS_PATH),
            "--repository-root",
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "valid"
    assert receipt["h8d_satisfied"] is False
    assert receipt["h8e_satisfied"] is False


@pytest.mark.parametrize("content", [b"not-json", b"[]"])
def test_load_object_rejects_invalid_json_objects(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(content)
    with pytest.raises(SamplingHarmAutomatedChallengeError, match="cannot load|object"):
        challenge_module._load_object(path)


def test_load_object_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SamplingHarmAutomatedChallengeError, match="cannot load"):
        challenge_module._load_object(tmp_path / "missing.json")


def test_schema_error_reports_root_location() -> None:
    with pytest.raises(SamplingHarmAutomatedChallengeError, match=r"invalid at \$"):
        challenge_module._validate_schema({}, {"type": "array"}, label="root")


@pytest.mark.parametrize(
    "value",
    [
        1,
        "/absolute.json",
        "specs/frontier/sampling-acquisition-harm/v1/reviews/../report.json",
        "specs/frontier/sampling-acquisition-harm/v1/reviews/nested/report.json",
    ],
)
def test_safe_report_path_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(SamplingHarmAutomatedChallengeError, match="path"):
        challenge_module._safe_report_path(value)


def _without_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        challenge_module, "_validate_schema", lambda *args, **kwargs: None
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.update(
                required_roles=list(reversed(value["required_roles"]))
            ),
            "required role inventory",
        ),
        (
            lambda value: value["role_reports"].reverse(),
            "role-report order",
        ),
        (
            lambda value: value["role_reports"][0].update(digest_verified=False),
            "digest_verified",
        ),
        (
            lambda value: value["role_reports"][0].update(report_id="wrong"),
            "projection",
        ),
        (
            lambda value: value["role_reports"][0].update(reviewer_eligible=True),
            "projection",
        ),
        (
            lambda value: value["findings"][0].update(
                finding_id=value["findings"][1]["finding_id"]
            ),
            "finding union",
        ),
        (
            lambda value: value["findings"][0].update(source_role="domain_specialist"),
            "provenance",
        ),
        (
            lambda value: value["findings"][0].update(disposition="fixed"),
            "cannot disposition",
        ),
        (
            lambda value: value["finding_summary"].update(total=18),
            "finding summary",
        ),
        (
            lambda value: (
                value["findings"][0].update(normalized_severity="Medium"),
                value["finding_summary"].update(high=14, medium=4),
            ),
            "unexpected challenge finding profile",
        ),
        (
            lambda value: value["actor"].update(human=True),
            "overclaims authority",
        ),
        (
            lambda value: value["gate_status"].update(
                required_role_reports_complete=False
            ),
            "coverage is incomplete",
        ),
        (
            lambda value: value["gate_status"].update(h8d_satisfied=True),
            "h8d_satisfied must remain false",
        ),
        (
            lambda value: value["reviewer_eligibility"].update(
                independent_eligibility_satisfied=True
            ),
            "reviewer eligibility",
        ),
        (
            lambda value: value["source_review"].update(source_review_satisfied=True),
            "source review",
        ),
        (
            lambda value: value["authority"].update(runtime_authority=True),
            "authority flags",
        ),
    ],
)
def test_semantic_validator_rejects_schema_bypass_mutations(
    monkeypatch: pytest.MonkeyPatch, mutation, match: str
) -> None:
    _without_schema(monkeypatch)
    value = deepcopy(_synthesis())
    mutation(value)
    _resign(value)
    with pytest.raises(SamplingHarmAutomatedChallengeError, match=match):
        validate_sampling_harm_automated_challenge(value, repository_root=ROOT)


@pytest.mark.parametrize(
    ("constant", "binding", "replacement", "match"),
    [
        ("FROZEN_CANDIDATE_COMMIT", "candidate_commit", "f" * 40, "commit"),
        ("FROZEN_CANDIDATE_TREE", "candidate_tree", "f" * 40, "tree"),
        ("FROZEN_PACKET_SHA256", "review_packet", "f" * 64, "packet"),
    ],
)
def test_role_reports_must_match_frozen_bindings(
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    binding: str,
    replacement: str,
    match: str,
) -> None:
    value = deepcopy(_synthesis())
    monkeypatch.setattr(challenge_module, constant, replacement)
    value["bindings"][binding]["value" if replacement.__len__() == 40 else "sha256"] = (
        replacement
    )
    _resign(value)
    with pytest.raises(SamplingHarmAutomatedChallengeError, match=match):
        validate_sampling_harm_automated_challenge(value, repository_root=ROOT)


def test_duplicate_source_finding_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _without_schema(monkeypatch)
    original_load = challenge_module._load_object

    def altered_load(path: Path):
        value = original_load(path)
        if path.name == "h8d-estimator-assurance-automated-20260803.json":
            value["finding_ids"].append("H8D-ED-01")
        return value

    monkeypatch.setattr(challenge_module, "_load_object", altered_load)
    monkeypatch.setattr(
        challenge_module,
        "canonical_json_sha256",
        lambda value, **kwargs: value.get(
            "report_sha256", value.get("synthesis_sha256")
        ),
    )
    with pytest.raises(SamplingHarmAutomatedChallengeError, match="duplicate"):
        validate_sampling_harm_automated_challenge(
            deepcopy(_synthesis()), repository_root=ROOT
        )
