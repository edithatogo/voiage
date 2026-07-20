from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.check_mutation_cohort import (
    cohort_identity,
    evaluate_cohort,
    mutation_universe,
    validate_runtime_version,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads(
    (ROOT / ".github/mutation-baselines/voiage-cohort.json").read_text(encoding="utf-8")
)
STATS = json.loads(
    (ROOT / ".github/mutation-baselines/voiage-broad.json").read_text(encoding="utf-8")
)
ANCHOR = "a" * 64


def _universe(*, replacement: bool = False, killed: int = 51) -> dict[str, object]:
    ids = [f"voiage.contracts.example__mutmut_{number}" for number in range(65)]
    if replacement:
        ids[-1] = "voiage.contracts.example__mutmut_replacement"
    rows = [
        f"    {mutant}: {'killed' if index < killed else 'survived'}"
        for index, mutant in enumerate(ids)
    ]
    return mutation_universe("\n".join(rows))


def _reviewed_baseline(identity: dict[str, object]) -> dict[str, object]:
    baseline = deepcopy(BASELINE)
    baseline["cohort"] = identity
    ids = _universe()["ids"]
    baseline["universe"] = {
        "ids": ids,
        "sha256": sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "promotion_state": "captured",
    }
    return baseline


def _evaluate(
    identity: dict[str, object],
    *,
    universe: dict[str, object] | None = None,
    baseline: dict[str, object] | None = None,
    reviewed: str = ANCHOR,
    threshold: float = 75.0,
) -> dict[str, object]:
    return evaluate_cohort(
        STATS,
        baseline or _reviewed_baseline(identity),
        identity,
        universe or _universe(),
        threshold,
        baseline_sha256=ANCHOR,
        reviewed_baseline_sha256=reviewed,
    )


def test_reviewed_mutation_cohort_binds_tool_lock_source_and_universe() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    report = _evaluate(identity)
    assert report["passed"] is True
    assert identity["tool_version"] == "3.6.0"
    assert len(identity["lock_sha256"]) == 64
    assert report["universe"]["added_ids"] == []
    assert report["debt"]["absolute"] == 14
    assert report["debt"]["density"] == 0.215385


def test_runtime_version_is_checked_separately_from_pure_cohort_identity() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    validate_runtime_version(identity, "3.6.0")
    message = ""
    try:
        validate_runtime_version(identity, "3.5.0")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("mismatched Mutmut runtime was accepted")
    assert "locked cohort" in message


def test_external_review_anchor_is_required_and_not_self_asserted() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    report = _evaluate(identity, reviewed="")
    assert report["passed"] is False
    assert report["external_review_anchor_valid"] is False
    assert "human_approved" not in BASELINE["promotion_provenance"]


def test_source_configuration_or_universe_drift_invalidates_cohort() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    drifted = deepcopy(identity)
    drifted["configuration_sha256"] = "0" * 64
    assert _evaluate(drifted, baseline=_reviewed_baseline(identity))["passed"] is False
    report = _evaluate(identity, universe=_universe(replacement=True))
    assert report["passed"] is False
    assert len(report["universe"]["added_ids"]) == 1
    assert len(report["universe"]["removed_ids"]) == 1


def test_debt_provenance_and_promoted_threshold_are_independent_gates() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    baseline = _reviewed_baseline(identity)
    worse = deepcopy(STATS)
    worse["killed"] = 50
    worse["survived"] = 15
    report = evaluate_cohort(
        worse,
        baseline,
        identity,
        _universe(killed=50),
        75.0,
        baseline_sha256=ANCHOR,
        reviewed_baseline_sha256=ANCHOR,
    )
    assert report["passed"] is False
    invalid = deepcopy(baseline)
    invalid["promotion_provenance"]["review_state"] = "self_approved"
    assert _evaluate(identity, baseline=invalid)["passed"] is False
    assert _evaluate(identity, threshold=74.9)["passed"] is False


def test_universe_parser_rejects_duplicates_and_unknown_status() -> None:
    duplicate = "pkg.fn__mutmut_1: killed\npkg.fn__mutmut_1: survived"
    unknown = "pkg.fn__mutmut_1: mystery"
    for payload in (duplicate, unknown):
        try:
            mutation_universe(payload)
        except ValueError:  # noqa: PERF203 - malformed inputs are table-driven
            pass
        else:
            raise AssertionError("invalid mutation universe was accepted")


def test_cohort_rejects_any_mutant_that_was_not_checked() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    ids = [f"voiage.contracts.example__mutmut_{number}" for number in range(65)]
    rows = [
        f"{mutant}: {'killed' if index < 51 else ('survived' if index < 64 else 'not checked')}"
        for index, mutant in enumerate(ids)
    ]
    with pytest.raises(ValueError, match="were not checked"):
        _evaluate(identity, universe=mutation_universe("\n".join(rows)))


def test_caught_by_type_check_is_reconciled_as_unresolved_debt() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    ids = [f"voiage.contracts.example__mutmut_{number}" for number in range(65)]
    rows = [
        f"{mutant}: {'killed' if index < 51 else ('survived' if index < 64 else 'caught by type check')}"
        for index, mutant in enumerate(ids)
    ]
    universe = mutation_universe("\n".join(rows))
    stats = deepcopy(STATS)
    stats["survived"] = 13
    report = evaluate_cohort(
        stats,
        _reviewed_baseline(identity),
        identity,
        universe,
        75.0,
        baseline_sha256=ANCHOR,
        reviewed_baseline_sha256=ANCHOR,
    )
    assert report["passed"] is True
    assert report["status_reconciliation"] == {
        "caught_by_type_check": 1,
        "caught_by_type_check_policy": "counts_as_unresolved_debt",
        "not_checked": 0,
        "reconciled_total": 65,
    }
    assert report["debt"]["absolute"] == 14
