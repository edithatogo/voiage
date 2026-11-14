"""A staged mutation review applies only to its exact pull request revision."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

from scripts.check_mutation_cohort import (
    cohort_identity,
    evaluate_cohort,
    mutation_universe,
    select_reviewed_anchor,
)

ROOT = Path(__file__).resolve().parents[1]
CURRENT = "a" * 64
CANDIDATE = "b" * 64
HEAD = "c" * 40
BASE = "d" * 40


def _review(**changes: object) -> str:
    record: dict[str, object] = {
        "baseline_sha256": CANDIDATE,
        "current_sha256": CURRENT,
        "pull_request": 1313,
        "head_sha": HEAD,
        "base_sha": BASE,
    }
    record.update(changes)
    return json.dumps(record)


def _select(**changes: str) -> tuple[str, str]:
    context = {
        "staged_review_json": _review(),
        "event_name": "pull_request",
        "pull_request": "1313",
        "head_sha": HEAD,
        "base_sha": BASE,
    }
    context.update(changes)
    return select_reviewed_anchor(CANDIDATE, CURRENT, **context)


def test_exact_staged_review_is_selected_only_for_its_pr_revision() -> None:
    assert _select() == (CANDIDATE, "staged")
    assert select_reviewed_anchor(
        CURRENT,
        CURRENT,
        staged_review_json=_review(),
        event_name="pull_request",
        pull_request="1313",
        head_sha=HEAD,
        base_sha=BASE,
    ) == (CURRENT, "current")


@pytest.mark.parametrize("current", ["", "a" * 63, "A" * 64])
def test_staged_review_requires_an_existing_valid_main_anchor(current: str) -> None:
    assert select_reviewed_anchor(
        CANDIDATE,
        current,
        staged_review_json=_review(),
        event_name="pull_request",
        pull_request="1313",
        head_sha=HEAD,
        base_sha=BASE,
    ) == (current, "current")


@pytest.mark.parametrize(
    "changes",
    [
        {"event_name": "push"},
        {"event_name": "pull_request_target"},
        {"pull_request": "1314"},
        {"head_sha": "e" * 40},
        {"base_sha": "f" * 40},
        {"head_sha": "C" * 40},
        {"staged_review_json": ""},
        {"staged_review_json": "{"},
        {"staged_review_json": "[]"},
        {"staged_review_json": _review(baseline_sha256="0" * 64)},
        {"staged_review_json": _review(current_sha256="0" * 64)},
        {"staged_review_json": _review(pull_request=True)},
        {"staged_review_json": _review(extra="not allowed")},
        {"staged_review_json": _review()[:-1] + ', "pull_request": 1313}'},
    ],
)
def test_staged_review_fails_closed_for_any_scope_mismatch(
    changes: dict[str, str],
) -> None:
    assert _select(**changes) == (CURRENT, "current")


def test_staged_anchor_does_not_bypass_cohort_or_debt_checks() -> None:
    baseline_file = ROOT / ".github/mutation-baselines/voiage-cohort.json"
    baseline_bytes = baseline_file.read_bytes()
    baseline = json.loads(baseline_bytes)
    stats = json.loads(
        (ROOT / ".github/mutation-baselines/voiage-broad.json").read_text()
    )
    baseline_hash = sha256(baseline_bytes).hexdigest()
    review = _review(baseline_sha256=baseline_hash)
    anchor, source = select_reviewed_anchor(
        baseline_hash,
        CURRENT,
        staged_review_json=review,
        event_name="pull_request",
        pull_request="1313",
        head_sha=HEAD,
        base_sha=BASE,
    )
    ids = baseline["universe"]["ids"]
    universe = mutation_universe(
        "\n".join(
            f"{item}: {'killed' if index < 51 else 'survived'}"
            for index, item in enumerate(ids)
        )
    )
    identity = cohort_identity(ROOT, ROOT / "pyproject.toml")
    report = evaluate_cohort(
        stats,
        baseline,
        identity,
        universe,
        75.0,
        baseline_sha256=baseline_hash,
        reviewed_baseline_sha256=anchor,
    )
    assert source == "staged"
    assert report["passed"] is True
    drifted = deepcopy(identity)
    drifted["configuration_sha256"] = "0" * 64
    assert (
        evaluate_cohort(
            stats,
            baseline,
            drifted,
            universe,
            75.0,
            baseline_sha256=baseline_hash,
            reviewed_baseline_sha256=anchor,
        )["passed"]
        is False
    )
    worse_stats = deepcopy(stats)
    worse_stats["killed"] = 50
    worse_stats["survived"] = 15
    worse_universe = mutation_universe(
        "\n".join(
            f"{item}: {'killed' if index < 50 else 'survived'}"
            for index, item in enumerate(ids)
        )
    )
    assert (
        evaluate_cohort(
            worse_stats,
            baseline,
            identity,
            worse_universe,
            75.0,
            baseline_sha256=baseline_hash,
            reviewed_baseline_sha256=anchor,
        )["passed"]
        is False
    )


def test_hosted_mutation_gate_uses_trusted_event_context() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["test-mutation"]["steps"]
    exact_head = next(
        step
        for step in steps
        if step.get("name") == "Assert and record exact source head"
    )
    assert 'test "${tested_head}" = "${EXPECTED_SOURCE_HEAD}"' in exact_head["run"]
    assert exact_head["env"]["EXPECTED_SOURCE_HEAD"] == (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )
    cohort = next(step for step in steps if step.get("id") == "cohort-mutation")
    assert (
        cohort["env"]["STAGED_REVIEW_JSON"]
        == "${{ vars.VOIAGE_MUTATION_STAGED_REVIEW }}"
    )
    assert (
        cohort["env"]["REVIEW_PR_NUMBER"] == "${{ github.event.pull_request.number }}"
    )
    assert (
        cohort["env"]["REVIEW_HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    )
    assert (
        cohort["env"]["REVIEW_BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    )
    for option, value in {
        "--staged-review-json": "STAGED_REVIEW_JSON",
        "--event-name": "GITHUB_EVENT_NAME",
        "--pull-request": "REVIEW_PR_NUMBER",
        "--head-sha": "REVIEW_HEAD_SHA",
        "--base-sha": "REVIEW_BASE_SHA",
    }.items():
        assert f'{option} "${{{value}}}"' in cohort["run"]
