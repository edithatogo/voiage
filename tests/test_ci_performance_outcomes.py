"""Executable checks for the measured CI and test-performance outcomes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTCOMES = (
    ROOT
    / "specs"
    / "submission-readiness"
    / "test-ci-performance-outcomes-20260829.json"
)


def _outcomes() -> dict[str, object]:
    return json.loads(OUTCOMES.read_text(encoding="utf-8"))


def test_parallel_outcomes_preserve_correctness_and_bound_stable_workers() -> None:
    outcome = _outcomes()
    suite = outcome["python_full_suite"]

    assert suite["local_winner"] in {4, 6, 8}
    assert suite["stable_ci_workers"] in {4, 6, 8}
    assert all(candidate["outcome"] == "passed" for candidate in suite["candidates"])
    assert suite["local_winner_reduction_percent"] >= 80
    assert "passed" in suite["correctness"]


def test_profiled_repairs_meet_declared_budgets_without_closing_hosted_gates() -> None:
    outcome = _outcomes()
    profiles = {item["id"]: item for item in outcome["profiled_repairs"]}

    assert profiles["repository-harness"]["reduction_percent"] >= 80
    assert profiles["paper-health-example"]["reduction_percent"] >= 50
    assert max(profiles["public-import"]["observed_seconds"]) < 5
    assert profiles["julia-fast-matrix"]["outcome"] == "passed"
    assert all(outcome["external_or_hosted_gates"])
    assert "No venue submission" in outcome["submission_boundary"]


def test_every_baseline_performance_finding_has_a_current_disposition() -> None:
    outcome = _outcomes()
    baseline = json.loads((ROOT / outcome["baseline"]).read_text(encoding="utf-8"))

    assert set(outcome["finding_dispositions"]) == {
        finding["id"] for finding in baseline["findings"]
    }
    assert all(
        state in {"resolved", "resolved_pending_green_hosted_measurement"}
        for state in outcome["finding_dispositions"].values()
    )
