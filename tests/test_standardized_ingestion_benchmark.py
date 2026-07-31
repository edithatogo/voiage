import json
from pathlib import Path

import pytest

from scripts.benchmark_standardized_ingestion import (
    BOUNDED_FIXTURE_MAX_RESOURCE_BYTES,
    BOUNDED_FIXTURE_MAX_ROWS,
    CANONICAL_PEAK_MEMORY_BUDGET_BYTES,
    assert_memory_budget,
    measure,
    measure_canonical_suite,
    measure_strict_local_suite,
    write_strict_local_fixture_suite,
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
        "input_bytes",
        "row_count",
    }
    assert all(value >= 0 for value in result.values())
    with pytest.raises(ValueError, match="positive"):
        measure(descriptor, repeats=0)


def test_canonical_benchmark_suite_covers_each_builtin_provider() -> None:
    suite = measure_canonical_suite(repeats=1)

    assert set(suite) == {"croissant", "frictionless"}
    assert all(result["peak_bytes"] > 0 for result in suite.values())
    assert_memory_budget(suite)


def test_strict_local_benchmark_suite_covers_supported_parse_formats() -> None:
    """Format coverage is structural; elapsed time is recorded, never budgeted."""
    suite = measure_strict_local_suite(repeats=1)

    assert set(suite) == {
        "croissant_csv",
        "frictionless_csv",
        "frictionless_json",
        "frictionless_parquet",
        "frictionless_arrow",
    }
    assert_memory_budget(suite)
    for result in suite.values():
        assert result["row_count"] == BOUNDED_FIXTURE_MAX_ROWS
        assert 0 < result["input_bytes"] <= BOUNDED_FIXTURE_MAX_RESOURCE_BYTES
        assert all(
            result[key] >= 0
            for key in ("parse_to_arrow_ms", "normalize_ms", "evpi_ms", "peak_bytes")
        )


def test_benchmark_fixture_writer_is_local_and_bounded(tmp_path: Path) -> None:
    descriptors = write_strict_local_fixture_suite(tmp_path)

    assert set(descriptors) == {
        "croissant_csv",
        "frictionless_csv",
        "frictionless_json",
        "frictionless_parquet",
        "frictionless_arrow",
    }
    assert all(
        path.parent == tmp_path and path.is_file() for path in descriptors.values()
    )
    croissant = json.loads(descriptors["croissant_csv"].read_text(encoding="utf-8"))
    assert croissant["distribution"][0]["contentUrl"] == "samples.csv"
    for name, descriptor in descriptors.items():
        if name == "croissant_csv":
            continue
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        assert payload["resources"][0]["path"].startswith("samples.")


def test_canonical_memory_budget_rejects_a_regression() -> None:
    with pytest.raises(AssertionError, match="frictionless peak memory"):
        assert_memory_budget(
            {
                "frictionless": {
                    "peak_bytes": float(CANONICAL_PEAK_MEMORY_BUDGET_BYTES + 1)
                }
            }
        )
