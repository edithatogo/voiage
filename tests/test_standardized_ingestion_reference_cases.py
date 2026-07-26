"""Executable evidence for the cross-domain standardized-ingestion examples."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_reference_cases_use_one_binding_and_one_evpi() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "standardized_ingestion"
        / "reference_cases.py"
    )
    spec = importlib.util.spec_from_file_location("reference_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_reference_cases()

    assert result["binding"]["role"] == "net_benefit"
    assert result["binding"]["field_ids"] == ["strategy_a", "strategy_b"]
    assert result["evpi"] == {
        "ml": pytest.approx(5.0),
        "engineering": pytest.approx(5.0),
        "business": pytest.approx(5.0),
    }
