from pathlib import Path

import pytest

from scripts.benchmark_standardized_ingestion import measure


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
    assert set(result) == {"parse_ms", "prepare_ms", "evpi_ms"}
    assert all(value >= 0 for value in result.values())
    with pytest.raises(ValueError, match="positive"):
        measure(descriptor, repeats=0)
