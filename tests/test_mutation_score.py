"""Tests for the bounded Mutmut score gate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from voiage.mutation_policy import mutation_score_from_mapping, validate_threshold

ROOT = Path(__file__).resolve().parents[1]


def _stats(**changes: int) -> dict[str, int]:
    values = {
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


def test_score_counts_every_non_skipped_unresolved_mutant() -> None:
    report = mutation_score_from_mapping(_stats()).report(90.0)
    assert report["eligible"] == 100
    assert report["score_percent"] == 90.0
    assert report["passed"] is True
    assert report["survived"] == 5
    assert report["no_tests"] == 2


def test_unreported_mutmut_statuses_cannot_inflate_score() -> None:
    report = mutation_score_from_mapping(_stats(total=104)).report(90.0)
    assert report["eligible"] == 101
    assert report["passed"] is False


def test_broad_baseline_ratchets_score_and_unresolved_debt() -> None:
    broad = {
        "no_tests": 0,
        "suspicious": 0,
        "timeout": 0,
        "segfault": 0,
        "skipped": 0,
    }
    baseline = mutation_score_from_mapping(
        _stats(killed=51, survived=14, total=65, **broad)
    )
    same = mutation_score_from_mapping(
        _stats(killed=51, survived=14, total=65, **broad)
    )
    report = same.report(75.0, baseline=baseline)
    assert report["non_decreasing"] is True
    assert report["debt_non_increasing"] is True
    assert report["baseline_killed"] == 51
    assert report["baseline_eligible"] == 65
    assert report["passed"] is True

    worse_score = mutation_score_from_mapping(
        _stats(killed=50, survived=15, total=65, **broad)
    ).report(75.0, baseline=baseline)
    assert worse_score["non_decreasing"] is False
    assert worse_score["passed"] is False

    new_debt = mutation_score_from_mapping(
        _stats(killed=52, survived=15, total=67, **broad)
    ).report(75.0, baseline=baseline)
    assert new_debt["debt_non_increasing"] is False
    assert new_debt["passed"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"killed": 89, "survived": 6},
        {"check_was_interrupted_by_user": 1, "total": 104},
        {
            "killed": 0,
            "survived": 0,
            "no_tests": 0,
            "suspicious": 0,
            "timeout": 0,
            "segfault": 0,
            "total": 3,
        },
    ],
)
def test_score_fails_below_threshold_interrupted_or_empty(
    changes: dict[str, int],
) -> None:
    assert (
        mutation_score_from_mapping(_stats(**changes)).report(90.0)["passed"] is False
    )


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", None])
def test_invalid_counts_fail_closed(value: object) -> None:
    raw: dict[str, object] = dict(_stats())
    raw["killed"] = value
    with pytest.raises(ValueError, match="non-negative integer"):
        mutation_score_from_mapping(raw)


def test_inconsistent_total_and_invalid_threshold_fail_closed() -> None:
    with pytest.raises(ValueError) as total_error:
        mutation_score_from_mapping(_stats(total=99))
    assert str(total_error.value) == (
        "mutation total is smaller than its reported status counts"
    )

    # Exercise one-count buckets and interruption independently. Subtracting any
    # reported status from the accounted total must not make an inconsistency valid.
    with pytest.raises(ValueError, match="smaller"):
        _ = mutation_score_from_mapping(_stats(total=102))
    with pytest.raises(ValueError, match="smaller"):
        _ = mutation_score_from_mapping(
            _stats(check_was_interrupted_by_user=1, total=103)
        )

    for threshold in (0.0, -1.0, 100.1):
        with pytest.raises(ValueError) as threshold_error:
            _ = validate_threshold(threshold)
        assert str(threshold_error.value) == (
            "mutation threshold must be greater than 0 and at most 100"
        )


def test_threshold_accepts_positive_fraction_and_closed_upper_boundary() -> None:
    assert validate_threshold(0.5) == 0.5
    assert validate_threshold(100.0) == 100.0


def test_cli_reports_counts_and_returns_pass_or_fail(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    report = tmp_path / "report.json"
    stats.write_text(json.dumps(_stats()), encoding="utf-8")
    passed = subprocess.run(
        [
            sys.executable,
            "scripts/check_mutation_score.py",
            "--stats",
            str(stats),
            "--output",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is True
    stats.write_text(json.dumps(_stats(killed=89, survived=6)), encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, "scripts/check_mutation_score.py", "--stats", str(stats)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 2
    assert json.loads(failed.stdout)["passed"] is False


def test_cli_enforces_hosted_broad_baseline(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    baseline = tmp_path / "baseline.json"
    stats.write_text(
        json.dumps(
            _stats(
                killed=50,
                survived=15,
                no_tests=0,
                suspicious=0,
                timeout=0,
                segfault=0,
                skipped=0,
                total=65,
            )
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            _stats(
                killed=51,
                survived=14,
                no_tests=0,
                suspicious=0,
                timeout=0,
                segfault=0,
                skipped=0,
                total=65,
            )
        ),
        encoding="utf-8",
    )
    failed = subprocess.run(
        [
            sys.executable,
            "scripts/check_mutation_score.py",
            "--stats",
            str(stats),
            "--baseline-stats",
            str(baseline),
            "--threshold",
            "75",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 2
    assert json.loads(failed.stdout)["non_decreasing"] is False
