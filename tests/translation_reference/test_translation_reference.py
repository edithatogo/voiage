"""Executable contracts for the source-derived migration reference."""

from pathlib import Path

import pytest

from scripts.check_translation_reference import load_cases, run_case, validate

ROOT = Path(__file__).parents[2]
IDS = [
    "index", "columns", "rows", "grouping", "concat", "alignment",
    "reduction", "rolling", "resample", "interpolation", "nulls",
    "float-format", "attrs", "nd-interpolation",
]


def test_derived_inventory_and_documentation_are_complete() -> None:
    assert validate(ROOT) == []
    assert [row["id"] for row in load_cases(ROOT)] == IDS


@pytest.mark.parametrize("row_id", IDS)
def test_polars_target(row_id: str) -> None:
    row = next(row for row in load_cases(ROOT) if row["id"] == row_id)
    run_case(row, "target")


@pytest.mark.parametrize("row_id", IDS)
def test_external_legacy_oracle(row_id: str) -> None:
    row = next(row for row in load_cases(ROOT) if row["id"] == row_id)
    pytest.importorskip(row["legacy_module"], reason="External oracle is unavailable; it is not a runtime dependency")
    run_case(row, "legacy")
