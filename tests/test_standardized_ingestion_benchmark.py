from pathlib import Path

import pytest

from scripts.benchmark_standardized_ingestion import (
    CANONICAL_PEAK_MEMORY_BUDGET_BYTES,
    assert_memory_budget,
    measure,
    measure_canonical_suite,
)


def test_ingestion_benchmark_separates_stages_and_validates_repeats() -> None:
    root = Path(__file__).parents[1]
    descriptor = (
        root
        / "tests"
        / "fixtures"
        / "standardized_ingestion"
        / "canonical-decision.croissant.json"
    )
    result = measure(descriptor, repeats=1)
    assert set(result) == {
        "parse_to_arrow_ms",
        "normalize_ms",
        "evpi_ms",
        "peak_bytes",
    }
    assert all(value >= 0 for value in result.values())
    with pytest.raises(ValueError, match="positive"):
        measure(descriptor, repeats=0)


def test_canonical_benchmark_suite_covers_each_builtin_provider() -> None:
    suite = measure_canonical_suite(repeats=1)

    assert set(suite) == {"croissant", "frictionless"}
    assert all(result["peak_bytes"] > 0 for result in suite.values())
    assert_memory_budget(suite)


def test_canonical_memory_budget_rejects_a_regression() -> None:
    with pytest.raises(AssertionError, match="frictionless peak memory"):
        assert_memory_budget(
            {
                "frictionless": {
                    "peak_bytes": float(CANONICAL_PEAK_MEMORY_BUDGET_BYTES + 1)
                }
            }
        )
