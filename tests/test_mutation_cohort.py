from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scripts.check_mutation_cohort import cohort_identity, evaluate_cohort

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads(
    (ROOT / ".github/mutation-baselines/voiage-cohort.json").read_text(encoding="utf-8")
)
STATS = json.loads(
    (ROOT / ".github/mutation-baselines/voiage-broad.json").read_text(encoding="utf-8")
)


def test_current_mutation_cohort_has_bound_debt_and_density() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    report = evaluate_cohort(STATS, BASELINE, identity, 75.0)
    assert report["passed"] is True
    assert report["identity_matches"] is True
    assert report["debt"]["absolute"] == 14
    assert report["debt"]["density"] == 0.215385


def test_source_or_configuration_drift_invalidates_cohort() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    identity["configuration_sha256"] = "0" * 64
    assert evaluate_cohort(STATS, BASELINE, identity, 75.0)["passed"] is False


def test_debt_and_promotion_provenance_are_independent_gates() -> None:
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    worse = deepcopy(STATS)
    worse["killed"] = 50
    worse["survived"] = 15
    assert evaluate_cohort(worse, BASELINE, identity, 75.0)["passed"] is False
    unapproved = deepcopy(BASELINE)
    unapproved["promotion_provenance"]["human_approved"] = False
    assert evaluate_cohort(STATS, unapproved, identity, 75.0)["passed"] is False
    assert evaluate_cohort(STATS, BASELINE, identity, 74.9)["passed"] is False
