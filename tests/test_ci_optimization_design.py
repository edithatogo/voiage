"""Contracts for the fail-closed CI and test optimization design."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
DESIGN_PATH = (
    ROOT / "specs" / "submission-readiness" / "ci-optimization-design-20260829.json"
)
BASELINE_PATH = (
    ROOT
    / "specs"
    / "submission-readiness"
    / "test-ci-performance-baseline-20260829.json"
)
PROFILE_PATH = (
    ROOT / "specs" / "submission-readiness" / "scalene-test-profile-20260829.json"
)


def _design() -> dict[str, object]:
    return json.loads(DESIGN_PATH.read_text(encoding="utf-8"))


def test_workflow_classes_keep_full_main_and_release_authority() -> None:
    workflows = _design()["workflow_classes"]

    assert workflows.keys() == {"pull_request", "main", "release", "preview"}
    assert workflows["pull_request"]["target_feedback_minutes"] == 10
    assert workflows["main"]["required"] is True
    assert workflows["release"]["required"] is True
    assert "non-sharded full suite" in workflows["release"]["selection"]
    assert workflows["preview"]["required"] is False
    assert "Never cancel" in workflows["release"]["cancellation"]


def test_python_parallelism_is_bounded_deterministic_and_fail_closed() -> None:
    execution = _design()["python_execution"]
    shards = execution["deterministic_sharding"]

    assert execution["worker_policy"]["candidates"] == [4, 6, 8]
    assert "Unbounded -n auto" in execution["worker_policy"]["prohibited"]
    assert "sorted pytest node IDs" in shards["algorithm"]
    assert "Python hash()" in shards["forbidden_inputs"]
    assert "fails the fan-in gate" in shards["drift_rule"]
    assert execution["serial_groups"]["initial_membership"]


def test_coverage_and_change_selection_cannot_omit_release_validation() -> None:
    design = _design()
    coverage = design["python_execution"]["coverage"]
    selection = design["change_selection"]

    assert "reject missing shard inputs" in coverage["merge"]
    assert "fresh non-sharded full coverage gate" in coverage["release"]
    assert selection["role"] == "additive early feedback only"
    assert "selects all stable lanes" in selection["fail_closed"]
    assert "ignore change-based omission" in selection["fail_closed"]


def test_caches_and_reusable_artifacts_preserve_identity_and_evidence() -> None:
    design = _design()
    caches = design["cache_policy"]
    artifact_ids = {artifact["id"] for artifact in design["artifact_dag"]}

    assert artifact_ids == {
        "source-and-lock-identity",
        "python-wheel",
        "native-ffi",
        "documentation-inputs",
        "coverage-fan-in",
    }
    assert {
        "test outcomes",
        "release attestations",
        "Conductor evidence entries",
    } <= set(caches["never_cache"])
    assert "never treat a cache hit as validation" in caches["poisoning_controls"]
    assert all(caches["allowed"].values())


def test_design_covers_all_measured_performance_and_profile_findings() -> None:
    design = _design()
    closure = design["finding_closure"]
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    source_ids = {
        *(finding["id"] for finding in baseline["findings"]),
        *(experiment["id"] for experiment in profile["required_experiments"]),
    }

    assert (
        closure.keys()
        == source_ids
        == {
            "PERF-001",
            "PERF-002",
            "PERF-003",
            "PERF-004",
            "PERF-005",
            "PERF-006",
            "PERF-007",
            "PROF-001",
            "PROF-002",
            "PROF-003",
            "PROF-004",
            "PROF-005",
        }
    )
    assert len(design["required_fan_in"]) >= 7
    assert len(design["implementation_order"]) == 10
    assert "same outcomes" in design["profiling_and_measurement"]["acceptance"]


def test_promoted_python_lanes_use_bounded_workstealing_and_single_ci_coverage() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    operational = (ROOT / ".github/workflows/operational-assurance.yml").read_text(
        encoding="utf-8"
    )
    tox = (ROOT / "tox.ini").read_text(encoding="utf-8")

    assert "--numprocesses=auto" not in ci
    assert "--numprocesses=6" in ci
    assert "--dist=worksteal" in ci
    assert "Run bounded parallel unit compatibility tests" in ci
    assert ci.count("Upload authoritative coverage to Codecov") == 1
    assert "-n 6 --dist=worksteal" in operational
    assert "-n 6 --dist=worksteal" in tox
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in ci
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in operational
