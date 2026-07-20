"""Architecture and scoring tests for the strict C13 mutation lane."""

from __future__ import annotations

import json
from pathlib import Path

import tomllib

from scripts.run_critical_mutation_lane import CRITICAL_MUTMUT_CONFIG, mutation_report

ROOT = Path(__file__).resolve().parents[1]


def _stats(**changes: int) -> dict[str, object]:
    values: dict[str, object] = {
        "killed": 90,
        "survived": 5,
        "no_tests": 2,
        "suspicious": 1,
        "timeout": 1,
        "segfault": 1,
        "skipped": 3,
        "check_was_interrupted_by_user": 0,
        "total": 103,
    }
    values.update(changes)
    return values


def test_critical_lane_targets_c13_and_c14_production_used_invariants() -> None:
    assert '"voiage/assurance_policy.py"' in CRITICAL_MUTMUT_CONFIG
    assert '"voiage/contracts/critical_invariants.py"' in CRITICAL_MUTMUT_CONFIG
    assert '"tests/test_assurance_policy.py"' in CRITICAL_MUTMUT_CONFIG
    assert '"tests/test_critical_invariants.py"' in CRITICAL_MUTMUT_CONFIG
    assert "tests/test_critical_invariants.py" in CRITICAL_MUTMUT_CONFIG
    assert "mutation_policy.py" not in CRITICAL_MUTMUT_CONFIG


def test_critical_lane_enforces_honest_threshold() -> None:
    assert mutation_report(_stats(), threshold=90.0)["passed"] is True
    assert (
        mutation_report(_stats(killed=89, survived=6), threshold=90.0)["passed"]
        is False
    )


def test_repository_declares_separate_broad_and_critical_evidence() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "mutation-score-broad.json" in workflow
    assert "mutation-critical.json" in workflow
    assert "--baseline-stats .github/mutation-baselines/voiage-broad.json" in workflow
    assert (
        "github.event_name == 'schedule'"
        not in workflow.split("test-mutation:", maxsplit=1)[1].split(
            "# Performance profiling", maxsplit=1
        )[0]
    )
    assert "scripts/run_critical_mutation_lane.py . --threshold 90" in workflow
    assert "voiage/contracts/capabilities.py" in project
    assert "voiage/contracts/digests.py" in project
    assert 'only_mutate = ["voiage/mutation_policy.py"]' not in project
    baseline = json.loads(
        (ROOT / ".github/mutation-baselines/voiage-broad.json").read_text(
            encoding="utf-8"
        )
    )
    parsed_project = tomllib.loads(project)
    assert baseline["scope"] == parsed_project["tool"]["mutmut"]["only_mutate"]
    assert (baseline["killed"], baseline["total"]) == (51, 65)
