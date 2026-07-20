from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scripts.check_coverage_policy import evaluate_coverage

ROOT = Path(__file__).resolve().parents[1]


def fixture() -> tuple[dict[str, object], dict[str, object], dict[str, set[int]]]:
    coverage: dict[str, object] = {
        "totals": {"percent_covered": 92.0},
        "files": {
            "voiage/critical.py": {
                "summary": {"percent_covered": 100.0},
                "executed_lines": [4, 5],
                "missing_lines": [],
                "executed_branches": [[4, 5], [4, 8]],
                "missing_branches": [],
            }
        },
    }
    policy: dict[str, object] = {
        "aggregate_percent": 90.0,
        "critical_modules": {"voiage/critical.py": 100.0},
        "changed_line_percent": 95.0,
        "changed_branch_percent": 100.0,
    }
    return coverage, policy, {"voiage/critical.py": {4, 5}}


def test_coverage_policy_enforces_all_three_surfaces() -> None:
    coverage, policy, changed = fixture()
    report = evaluate_coverage(coverage, policy, changed)
    assert report["passed"] is True
    assert report["changed"]["branches"] == 2


def test_coverage_policy_rejects_uncovered_changed_branch() -> None:
    coverage, policy, changed = fixture()
    details = coverage["files"]["voiage/critical.py"]
    details["executed_branches"] = [[4, 5]]
    details["missing_branches"] = [[4, 8]]
    report = evaluate_coverage(coverage, policy, changed)
    assert report["passed"] is False
    assert report["changed"]["missing_branches"] == ["voiage/critical.py:4->8"]


def test_coverage_policy_rejects_aggregate_or_missing_critical_module() -> None:
    coverage, policy, changed = fixture()
    low = deepcopy(coverage)
    low["totals"]["percent_covered"] = 89.99
    assert evaluate_coverage(low, policy, changed)["passed"] is False
    policy["critical_modules"] = {"voiage/absent.py": 90.0}
    assert evaluate_coverage(coverage, policy, changed)["passed"] is False


def test_coverage_policy_rejects_changed_production_file_without_measurement() -> None:
    coverage, policy, changed = fixture()
    changed["voiage/new_module.py"] = {1}
    report = evaluate_coverage(coverage, policy, changed)
    assert report["passed"] is False
    assert report["changed"]["unmeasured_files"] == ["voiage/new_module.py"]


def test_coverage_policy_binds_tested_revision_to_source_head() -> None:
    coverage, policy, changed = fixture()
    exact = evaluate_coverage(
        coverage,
        policy,
        changed,
        source_head="a" * 40,
        tested_head="a" * 40,
    )
    assert exact["passed"] is True
    assert exact["exact_source_head"] is True
    mismatch = evaluate_coverage(
        coverage,
        policy,
        changed,
        source_head="a" * 40,
        tested_head="b" * 40,
    )
    assert mismatch["passed"] is False
    assert mismatch["exact_source_head"] is False


def test_workflow_uses_full_suite_provenance_floor_and_hidden_evidence() -> None:
    workflow = (ROOT / ".github/workflows/operational-assurance.yml").read_text(
        encoding="utf-8"
    )
    assert "pytest tests/ --cov=voiage" in workflow
    assert '-m "not integration' not in workflow
    assert "4017aac3b5803f2d68b09d74e57ebd6c55e933d0" in workflow
    assert 'merge-base --is-ancestor "${C15_PROVENANCE_FLOOR}" HEAD' in workflow
    assert "include-hidden-files: true" in workflow
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert '--source-head "${SOURCE_HEAD}"' in workflow
