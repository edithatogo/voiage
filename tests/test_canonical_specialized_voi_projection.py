from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/validate_canonical_specialized_voi_projection.py"
SPEC = importlib.util.spec_from_file_location("specialized_voi_projection", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_c16_projection_matches_voiage_tracks() -> None:
    MODULE.validate(
        ROOT / "conductor/canonical-projections/specialized-voi-v1.2.0.json", ROOT
    )


def test_c17_projection_matches_mcda_track_and_remains_experimental() -> None:
    projection_path = (
        ROOT / "conductor/canonical-projections/specialized-voi-v1.3.0.json"
    )
    MODULE.validate(projection_path, ROOT)

    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    by_number = {item["number"]: item for item in projection["issues"]}
    issue = by_number[560]
    assert projection["canonical_track"] == "C17"
    assert set(issue["requirement_ids"]) == {"M17", "M21"}
    assert issue["subissues"] == [746, 747, 748, 749, 750]
    assert issue["implementation_pr"] == 751
    assert issue["maturity"] == "experimental"
    assert issue["stable_claim_allowed"] is False
    assert projection["github_project"]["fields"]["Evidence State"] == "Unverified"
    assert projection["github_project"]["fields"]["Sync State"] == "Clean"


def test_c16_projection_maps_voc_family_to_dedicated_track() -> None:
    """#595 and its native delivery issues share one non-duplicate track."""
    projection = json.loads(
        (
            ROOT / "conductor/canonical-projections/specialized-voi-v1.2.0.json"
        ).read_text(encoding="utf-8")
    )
    by_number = {item["number"]: item for item in projection["issues"]}
    issue = by_number[595]
    assert issue["track_id"] == "risk_adjusted_information_pricing_20260731"
    assert set(issue["requirement_ids"]) == {"M16", "M17"}
    assert issue["subissues"] == [694, 695, 696, 697]
    assert issue["capability_contract"].endswith(
        "risk_adjusted_information_pricing_20260731/contract.md"
    )


def test_c16_projection_rejects_missing_consumer_registration(tmp_path: Path) -> None:
    projection = json.loads(
        (
            ROOT / "conductor/canonical-projections/specialized-voi-v1.2.0.json"
        ).read_text(encoding="utf-8")
    )
    projection["registered_repositories"] = []
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(projection), encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly managed"):
        MODULE.validate(path, ROOT)


def test_projection_rejects_an_unregistered_version(tmp_path: Path) -> None:
    projection = json.loads(
        (
            ROOT / "conductor/canonical-projections/specialized-voi-v1.3.0.json"
        ).read_text(encoding="utf-8")
    )
    projection["projection_id"] = "specialized-voi-v9.9.9"
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(projection), encoding="utf-8")

    with pytest.raises(ValueError, match="is not registered"):
        MODULE.validate(path, ROOT)


def test_projection_rejects_a_broadened_sync_policy(tmp_path: Path) -> None:
    projection = json.loads(
        (
            ROOT / "conductor/canonical-projections/specialized-voi-v1.3.0.json"
        ).read_text(encoding="utf-8")
    )
    projection["sync_policy"]["automatic_merge"] = True
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(projection), encoding="utf-8")

    with pytest.raises(ValueError, match="fail-closed"):
        MODULE.validate(path, ROOT)


def test_projection_rejects_content_for_a_different_selected_path() -> None:
    with pytest.raises(ValueError, match="does not match selected path"):
        MODULE.validate(
            ROOT / "conductor/canonical-projections/specialized-voi-v1.2.0.json",
            ROOT,
            "specialized-voi-v1.3.0",
        )


def test_projection_rejects_project_field_drift(tmp_path: Path) -> None:
    projection = json.loads(
        (
            ROOT / "conductor/canonical-projections/specialized-voi-v1.3.0.json"
        ).read_text(encoding="utf-8")
    )
    projection["github_project"]["fields"]["Evidence State"] = "Verified"
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(projection), encoding="utf-8")

    with pytest.raises(ValueError, match="Project 28"):
        MODULE.validate(path, ROOT)


@pytest.mark.parametrize("invalid_issue", ["not-an-object", {"number": 560}])
def test_projection_rejects_malformed_or_duplicate_issues(
    tmp_path: Path, invalid_issue: object
) -> None:
    projection = json.loads(
        (
            ROOT / "conductor/canonical-projections/specialized-voi-v1.3.0.json"
        ).read_text(encoding="utf-8")
    )
    if isinstance(invalid_issue, dict):
        projection["issues"].append(projection["issues"][1].copy())
    else:
        projection["issues"].append(invalid_issue)
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(projection), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="issue"):
        MODULE.validate(path, ROOT)
