"""Tests for the v1 Python runtime inventory contract."""

from pathlib import Path
import json

from scripts.validate_python_runtime_inventory import validate


def test_every_runtime_module_has_one_inventory_category() -> None:
    errors = validate(Path(__file__).resolve().parents[1])
    assert errors == []


def test_transitional_numerical_core_is_not_in_v1_retained_allowlist() -> None:
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "specs/v1/python-runtime-inventory.json").read_text()
    )
    assert "transitional_numerical_core" not in manifest["v1_retained_categories"]
