"""Governance tests for stable-core Python/Rust performance evidence."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

CONTRACT_PATH = Path("specs/v1/stable-core-performance.json")
SCHEMA_PATH = Path("specs/v1/stable-core-performance.schema.json")
RUST_BASELINE_PATH = Path(
    "rust/crates/voiage-numerics/benches/foundational-baseline.json"
)
PYTHON_BASELINE_PATH = Path("benchmarks/c15_performance_baseline.json")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_performance_contract_conforms_to_its_schema() -> None:
    Draft202012Validator(_load(SCHEMA_PATH)).validate(_load(CONTRACT_PATH))


def test_performance_contract_records_non_equivalent_baseline_boundaries() -> None:
    contract = _load(CONTRACT_PATH)
    by_id = {record["baseline_id"]: record for record in contract["baselines"]}

    assert by_id["rust-foundational-v1"]["execution_boundary"] == "direct-rust-kernel"
    assert by_id["python-c15-evpi-v1"]["execution_boundary"] == "numpy-reference-oracle"
    assert contract["comparison_policy"]["paired_speedup_claim"] == "prohibited"
    assert contract["comparison_policy"]["reason"] == (
        "existing Python and Rust baselines do not measure the same execution boundary"
    )


def test_performance_budgets_are_bound_to_executable_baselines() -> None:
    contract = _load(CONTRACT_PATH)
    rust = _load(RUST_BASELINE_PATH)
    python = _load(PYTHON_BASELINE_PATH)
    by_id = {record["baseline_id"]: record for record in contract["baselines"]}

    assert by_id["rust-foundational-v1"]["source"] == str(RUST_BASELINE_PATH)
    assert by_id["rust-foundational-v1"]["budgets"] == rust["regression_budgets"]
    assert by_id["python-c15-evpi-v1"]["source"] == str(PYTHON_BASELINE_PATH)
    assert by_id["python-c15-evpi-v1"]["budgets"] == {
        cohort: record["maximum_upper_seconds"]
        for cohort, record in python["cohorts"].items()
    }


def test_paired_benchmark_is_ready_for_hosted_measurement() -> None:
    gate = _load(CONTRACT_PATH)["paired_baseline_gate"]

    assert gate["status"] == "open"
    assert "blocked_by" not in gate
    assert "direct Rust facade timing" in gate["required_evidence"]


def test_every_baseline_declares_runner_and_claim_limitations() -> None:
    for baseline in _load(CONTRACT_PATH)["baselines"]:
        assert Path(baseline["source"]).is_file()
        assert baseline["runner_scope"]
        assert baseline["measures"]
        assert baseline["does_not_measure"]
        assert baseline["claim_limit"]


def test_dependency_frontier_success_is_retained_as_separate_evidence() -> None:
    gate = _load(CONTRACT_PATH)["dependency_frontier_gate"]

    assert gate["status"] == "satisfied"
    assert gate["reason"] == (
        "upgraded-lock-matches-newest-stable-releases-admitted-by-declared-"
        "compatibility-ranges"
    )
    assert gate["does_not_invalidate_performance_contract"] is True
