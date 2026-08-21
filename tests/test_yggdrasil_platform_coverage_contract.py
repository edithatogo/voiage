"""Fail-closed contracts for maximum-coverage Yggdrasil filtering."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from jsonschema import Draft202012Validator
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "specs" / "yggdrasil-platform-coverage" / "v1"
SCHEMA = CONTRACT_ROOT / "platform-coverage.schema.json"
MANIFEST = CONTRACT_ROOT / "voiage-ffi-platform-coverage.json"
VALIDATOR = ROOT / "scripts" / "validate_yggdrasil_platform_coverage.py"


def _payload() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _run_validator(tmp_path: Path, payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    candidate = tmp_path / "platform-coverage.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--manifest", str(candidate)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_committed_contract_is_schema_valid_and_semantically_reconciled(
    tmp_path: Path,
) -> None:
    """The canonical manifest must pass both structural and semantic checks."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    payload = _payload()

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    result = _run_validator(tmp_path, payload)

    assert result.returncode == 0, result.stderr
    assert "platform coverage contract valid" in result.stdout


def test_contract_pins_inclusive_policy_and_two_initial_filters() -> None:
    """The first expanded run must not pre-emptively filter extra targets."""
    payload = _payload()

    assert payload["catalogue"]["function"] == "supported_platforms()"
    assert payload["policy"]["mode"] == "inclusive_negative_filter"
    assert [item["predicate"] for item in payload["policy"]["negative_filters"]] == [
        'Sys.isfreebsd(p) && arch(p) == "aarch64"',
        'arch(p) == "riscv64"',
    ]


def _remove_classification(payload: dict[str, Any]) -> None:
    payload["platforms"].pop()
    payload["aggregates"]["classified"] -= 1


def _duplicate_platform(payload: dict[str, Any]) -> None:
    payload["platforms"].append(deepcopy(payload["platforms"][0]))
    payload["aggregates"]["classified"] += 1


def _remove_reconsideration_trigger(payload: dict[str, Any]) -> None:
    excluded = next(row for row in payload["platforms"] if row["disposition"] == "excluded")
    del excluded["exclusion"]["reconsideration_trigger"]


def _forge_totals(payload: dict[str, Any]) -> None:
    payload["aggregates"]["included"] += 1


def _forge_runtime_claim(payload: dict[str, Any]) -> None:
    included = next(row for row in payload["platforms"] if row["disposition"] == "included")
    included["evidence"]["runtime"] = "passed"


def _add_uncontracted_filter(payload: dict[str, Any]) -> None:
    payload["policy"]["negative_filters"].append(
        {
            "predicate": "Sys.iswindows(p)",
            "reason_category": "project_architecture_limitation",
            "reason": "Broad platform exclusion without a failed target receipt.",
            "primary_evidence": "https://example.invalid/no-evidence",
            "observed_at": "2026-08-21T00:00:00Z",
            "reconsideration_trigger": "Unspecified future review.",
            "matched_platform_ids": [],
        }
    )


@pytest.mark.parametrize(
    "mutate",
    [
        _remove_classification,
        _duplicate_platform,
        _remove_reconsideration_trigger,
        _forge_totals,
        _forge_runtime_claim,
        _add_uncontracted_filter,
    ],
    ids=[
        "unclassified-catalogue-platform",
        "duplicate-platform",
        "missing-reconsideration-trigger",
        "stale-aggregate-counts",
        "runtime-overclaim",
        "broad-uncontracted-filter",
    ],
)
def test_validator_rejects_pathological_contract_mutations(
    tmp_path: Path,
    mutate: Any,
) -> None:
    """Structural, reconciliation, and evidence overclaim mutations fail closed."""
    payload = _payload()
    mutate(payload)

    result = _run_validator(tmp_path, payload)

    assert result.returncode != 0
    assert result.stderr.strip()
