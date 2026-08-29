"""Contracts for the dated test and CI performance baseline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
BASELINE = (
    ROOT
    / "specs"
    / "submission-readiness"
    / "test-ci-performance-baseline-20260829.json"
)
PROFILE = ROOT / "specs" / "submission-readiness" / "scalene-test-profile-20260829.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_baseline_covers_every_required_execution_surface() -> None:
    baseline = _json(BASELINE)
    observations = {entry["id"]: entry for entry in baseline["local_observations"]}

    assert {
        "collection-serial",
        "pytest-serial",
        "pytest-xdist-worksteal-8",
        "repository-harness",
        "rust-cold-after-lock-refresh",
        "rust-warm",
        "r-source-check-current",
        "julia-cold",
        "julia-warm",
        "tox-docs-cold",
    } == observations.keys()
    assert observations["pytest-serial"]["passed"] > 4000
    assert (
        observations["pytest-xdist-worksteal-8"]["wall_reduction_percent_from_serial"]
        > 80
    )
    assert observations["r-source-check-current"]["outcome"] == "passed"
    assert observations["julia-warm"]["outcome"] == "passed"
    assert observations["tox-docs-cold"]["built_pages"] > 1000


def test_hosted_baseline_is_explicitly_representative_and_source_bound() -> None:
    hosted = _json(BASELINE)["hosted_reference"]

    assert "not exact-head" in hosted["interpretation"]
    assert hosted["ci"]["source_head"] == hosted["operational_assurance"]["source_head"]
    assert len(hosted["ci"]["source_head"]) == 40
    assert hosted["bindings"]["source_head"] != hosted["ci"]["source_head"]
    assert all(
        section["url"].startswith("https://github.com/edithatogo/voiage/")
        for section in hosted.values()
        if isinstance(section, dict) and "url" in section
    )


def test_performance_findings_are_complete_and_actionable() -> None:
    findings = _json(BASELINE)["findings"]

    assert {finding["id"] for finding in findings} == {
        f"PERF-{number:03d}" for number in range(1, 8)
    }
    assert {finding["state"] for finding in findings} == {"open"}
    assert all(finding["required_disposition"] for finding in findings)


def test_scalene_profiles_record_attribution_and_limitations() -> None:
    profile = _json(PROFILE)
    profiles = {entry["id"]: entry for entry in profile["profiles"]}

    assert profiles.keys() == {
        "general-evpi-evppi",
        "repository-harness",
        "paper-health-example",
    }
    assert all(entry["outcome"] == "passed" for entry in profiles.values())
    assert all(entry["attribution"] for entry in profiles.values())
    assert len(profile["tool_limitations"]) >= 4
    assert {entry["id"] for entry in profile["required_experiments"]} == {
        f"PROF-{number:03d}" for number in range(1, 6)
    }


def test_profile_paths_and_test_nodeids_remain_repository_owned() -> None:
    baseline = _json(BASELINE)
    profile = _json(PROFILE)

    for entry in baseline["serial_slowest_tests_seconds"]:
        relative_path = entry["nodeid"].split("::", maxsplit=1)[0]
        assert (ROOT / relative_path).is_file()
    for entry in profile["profiles"]:
        for attribution in entry["attribution"]:
            assert (ROOT / attribution["path"]).is_file()
