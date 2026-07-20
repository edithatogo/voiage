"""Tests for the v1 Python runtime inventory contract."""

from pathlib import Path

from scripts.validate_python_runtime_inventory import validate


def test_every_runtime_module_has_one_inventory_category() -> None:
    errors = validate(Path(__file__).resolve().parents[1])
    assert errors == []
