"""Contract tests for the committed native benchmark baseline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "rust/crates/voiage-numerics/benches/foundational-baseline.json"


def test_native_benchmark_baseline_is_complete_and_versioned() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    required = set(baseline["required_operations"])
    assert required == {
        "evpi",
        "enbs",
        "evppi",
        "evsi",
        "evsi_efficient_linear",
        "evsi_moment_based",
        "dominance",
        "ceaf",
    }
    assert required <= set(baseline["workloads"])

    budgets = baseline["regression_budgets"]
    assert baseline["regression_budget_version"] == 1
    assert budgets["metric"] == "elapsed_ns"
    assert budgets["scope"] == "native_evsi_only"
    assert budgets["enforcement"] == "fail_if_observed_exceeds_max"

    for operation, budget in budgets["operations"].items():
        assert operation in {"evsi", "evsi_efficient_linear", "evsi_moment_based"}
        assert 0 < budget["reference_elapsed_ns"] <= budget["max_elapsed_ns"]
