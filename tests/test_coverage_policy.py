from __future__ import annotations

from copy import deepcopy

from scripts.check_coverage_policy import evaluate_coverage


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
