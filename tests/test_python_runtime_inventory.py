"""Tests for the v1 Python runtime inventory contract."""

import json
from pathlib import Path

from scripts.validate_python_runtime_inventory import validate


def test_every_runtime_module_has_one_inventory_category() -> None:
    errors = validate(Path(__file__).resolve().parents[1])
    assert errors == []


def test_transitional_numerical_core_is_not_in_v1_retained_allowlist() -> None:
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "specs/v1/python-runtime-inventory.json"
        ).read_text()
    )
    assert "transitional_numerical_core" not in manifest["v1_retained_categories"]


def test_transitional_kernel_inventory_is_explicit() -> None:
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "specs/v1/python-runtime-inventory.json"
        ).read_text()
    )
    assert set(manifest["transitional_kernel_modules"]["modules"]) == {
        "voiage/methods/basic.py",
        "voiage/methods/ceaf.py",
        "voiage/methods/dominance.py",
        "voiage/methods/sample_information.py",
    }
