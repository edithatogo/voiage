"""Architecture and scoring tests for the strict C13 mutation lane."""

from __future__ import annotations

from pathlib import Path

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


def test_critical_lane_targets_only_production_used_invariants() -> None:
    assert (
        'only_mutate = ["voiage/contracts/critical_invariants.py"]'
        in CRITICAL_MUTMUT_CONFIG
    )
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
    assert "mutation-broad-stats.json" in workflow
    assert "mutation-critical.json" in workflow
    assert "scripts/run_critical_mutation_lane.py . --threshold 90" in workflow
    assert "voiage/contracts/capabilities.py" in project
    assert "voiage/contracts/digests.py" in project
    assert 'only_mutate = ["voiage/mutation_policy.py"]' not in project
