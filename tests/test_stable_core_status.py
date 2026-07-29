"""Conformance tests for generated stable-core maturity and capability status."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

STATUS_PATH = Path("specs/v1/stable-core-status.json")
ASSURANCE_PATH = Path("specs/v1/stable-estimator-assurance.json")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stable_core_status_is_current_generated_output() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_stable_core_status.py", "--check"],
        check=True,
    )


def test_status_covers_every_stable_profile_once() -> None:
    status = _load(STATUS_PATH)
    assurance = _load(ASSURANCE_PATH)
    records = status["methods"]
    method_ids = [record["method_id"] for record in records]

    assert len(method_ids) == len(set(method_ids))
    assert set(method_ids) == {
        profile["method_id"] for profile in assurance["profiles"]
    }


def test_status_keeps_api_maturity_distinct_from_v11_assurance() -> None:
    status = _load(STATUS_PATH)
    by_id = {record["method_id"]: record for record in status["methods"]}

    assert all(record["api_maturity"] == "stable" for record in by_id.values())
    assert by_id["net-benefit"]["v1_1_assurance_state"] == "eligible"
    assert by_id["dominance"]["v1_1_assurance_state"] == "eligible"
    assert by_id["expected-loss"]["runtime_assurance_state"] == (
        "integrated-incomplete-assurance"
    )
    assert by_id["expected-loss"]["open_gates"] == [
        "replication-and-convergence-assurance"
    ]
    assert by_id["evsi-nested-mc"]["authority_boundary"] == (
        "python-compatibility-path"
    )
    assert by_id["evsi-nested-mc"]["v1_1_assurance_state"] == "open"
    assert by_id["evsi-regression"]["v1_1_assurance_state"] == "eligible"
    assert by_id["evsi-moment-matching"]["v1_1_assurance_state"] == "eligible"


def test_status_disallows_aggregate_release_readiness_claim() -> None:
    status = _load(STATUS_PATH)

    assert status["aggregate"]["v1_1_release_ready"] is False
    assert status["aggregate"]["open_method_gates"]
    assert status["aggregate"]["open_repository_gates"] == [
        "replication-and-convergence-assurance",
        "matched-cross-language-performance",
    ]
    assert status["aggregate"]["human_gates"] == [
        "phase-1-manual-verification",
        "phase-2-manual-verification",
        "phase-3-manual-verification",
        "phase-4-manual-verification",
    ]
