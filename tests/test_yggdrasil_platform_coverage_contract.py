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


def _run_validator(
    tmp_path: Path, payload: dict[str, Any]
) -> subprocess.CompletedProcess[str]:
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
    assert payload["catalogue"]["binarybuilderbase_revision"] == (
        "76c4aab80ad5019af59af0f42e5669109cd5194b"
    )
    assert payload["candidate"]["source_revision"] == (
        "964a0fc334ece9509387cd07d43776adf38be240"
    )
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
    excluded = next(
        row for row in payload["platforms"] if row["disposition"] == "excluded"
    )
    del excluded["exclusion"]["reconsideration_trigger"]


def _forge_totals(payload: dict[str, Any]) -> None:
    payload["aggregates"]["included"] += 1


def _forge_runtime_claim(payload: dict[str, Any]) -> None:
    included = next(
        row for row in payload["platforms"] if row["disposition"] == "included"
    )
    included["evidence"]["runtime"] = "passed"


def _add_uncontracted_filter(payload: dict[str, Any]) -> None:
    platform = next(
        row for row in payload["platforms"] if row["id"] == "i686-w64-mingw32"
    )
    filter_contract = {
        "predicate": "Sys.iswindows(p)",
        "specificity": "exact_arch_os",
        "evidence_kind": "hosted_failure",
        "reason_category": "project_architecture_limitation",
        "reason": "One Windows target failed in the expanded hosted build matrix.",
        "primary_evidence": "https://buildkite.com/julialang/yggdrasil/builds/1",
        "observed_at": "2026-08-21T00:00:00Z",
        "reconsideration_trigger": "Retest after the project architecture limitation is removed.",
        "matched_platform_ids": [platform["id"]],
    }
    payload["policy"]["stage"] = "evidence_filtered"
    payload["policy"]["negative_filters"].append(filter_contract)
    payload["candidate"]["hosted_run"] = {
        "state": "in_progress",
        "url": "https://buildkite.com/julialang/yggdrasil/builds/1",
        "platform_status_source": "buildkite",
    }
    platform["disposition"] = "excluded"
    platform["lifecycle"] = "excluded_evidenced"
    platform["evidence"] = {
        "build": "not_run",
        "product": "not_run",
        "abi_smoke": "not_run",
        "runtime": "not_run",
        "locators": [filter_contract["primary_evidence"]],
    }
    platform["exclusion"] = {
        key: value
        for key, value in filter_contract.items()
        if key != "matched_platform_ids"
    }
    payload["aggregates"]["included"] -= 1
    payload["aggregates"]["excluded"] += 1
    payload["aggregates"]["lifecycle_counts"]["pending"] -= 1
    payload["aggregates"]["lifecycle_counts"]["excluded_evidenced"] += 1


def _replace_with_placeholder_evidence(payload: dict[str, Any]) -> None:
    policy_filter = payload["policy"]["negative_filters"][0]
    platform_id = policy_filter["matched_platform_ids"][0]
    platform = next(row for row in payload["platforms"] if row["id"] == platform_id)
    placeholder = "https://example.invalid/not-authoritative"
    policy_filter["primary_evidence"] = placeholder
    platform["exclusion"]["primary_evidence"] = placeholder


@pytest.mark.parametrize(
    "mutate",
    [
        _remove_classification,
        _duplicate_platform,
        _remove_reconsideration_trigger,
        _forge_totals,
        _forge_runtime_claim,
        _add_uncontracted_filter,
        _replace_with_placeholder_evidence,
    ],
    ids=[
        "unclassified-catalogue-platform",
        "duplicate-platform",
        "missing-reconsideration-trigger",
        "stale-aggregate-counts",
        "runtime-overclaim",
        "broad-uncontracted-filter",
        "placeholder-exclusion-evidence",
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


def test_every_catalogue_platform_is_required_by_reconciliation(tmp_path: Path) -> None:
    """Removing any single classification fails regardless of adjusted aggregates."""
    original = _payload()
    for platform_id in original["catalogue"]["platforms"]:
        payload = deepcopy(original)
        removed = next(row for row in payload["platforms"] if row["id"] == platform_id)
        payload["platforms"].remove(removed)
        payload["aggregates"]["classified"] -= 1
        payload["aggregates"][removed["disposition"]] -= 1
        payload["aggregates"]["lifecycle_counts"][removed["lifecycle"]] -= 1

        result = _run_validator(tmp_path, payload)

        assert result.returncode != 0, platform_id
        assert "classification mismatch" in result.stderr


def test_cli_rejects_malformed_json(tmp_path: Path) -> None:
    """Malformed evidence never falls through to a partial semantic check."""
    candidate = tmp_path / "malformed.json"
    candidate.write_text('{"schema_version":', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--manifest", str(candidate)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cannot load JSON" in result.stderr
