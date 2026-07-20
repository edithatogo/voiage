"""Executable governance checks for the normative v1.0 public API contract."""

from __future__ import annotations

import json
from pathlib import Path

import voiage

CONTRACT_PATH = Path("specs/v1/stable-api.json")


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_v1_contract_defines_supported_runtime_and_behavior() -> None:
    """The stable contract must freeze runtime and numerical expectations."""
    contract = _contract()

    assert contract["contract_version"] == "1.0.0"
    assert contract["python"] == {
        "versions": ["3.10", "3.11", "3.12", "3.13", "3.14"],
        "platforms": ["linux", "macos", "windows"],
        "reference_facade": True,
        "authoritative_numerics": "rust",
    }
    assert contract["numerics"] == {
        "default_absolute_tolerance": 1e-10,
        "default_relative_tolerance": 1e-8,
        "seed_type": "unsigned-64-bit-integer",
        "same_seed_same_platform": "bitwise",
        "cross_platform": "within-declared-tolerance",
        "nan_policy": "reject-unless-method-explicitly-documents-support",
        "infinity_policy": "reject",
    }
    assert contract["errors"]["invalid_input"] == "InputError"
    assert contract["errors"]["shape_mismatch"] == "DimensionMismatchError"
    assert contract["errors"]["unsupported_capability"] == "BackendNotAvailableError"


def test_v1_contract_freezes_stable_python_exports() -> None:
    """Every stable Python symbol must exist and remain distinct from research APIs."""
    contract = _contract()
    symbols = contract["symbols"]
    stable = symbols["stable"]

    expected = {
        "DecisionAnalysis",
        "DecisionOption",
        "CEAFResult",
        "DominanceResult",
        "ParameterSet",
        "PortfolioSpec",
        "PortfolioStudy",
        "TrialDesign",
        "ValueArray",
        "ceaf",
        "dominance",
        "enbs",
        "evpi",
        "evppi",
        "evsi",
    }
    assert set(stable) == expected
    assert all(hasattr(voiage, name) for name in stable)
    assert set(stable).isdisjoint(symbols["provisional"])
    assert set(stable).isdisjoint(symbols["experimental"])
    assert set(stable).isdisjoint(symbols["deprecated"])
    assert symbols["removed"] == []


def test_v1_contract_specifies_core_method_shapes_and_failures() -> None:
    """Core methods must define input shape, output, missing values, and errors."""
    methods = _contract()["methods"]

    assert set(methods) == {"evpi", "evppi", "evsi", "enbs", "ceaf", "dominance"}
    for name, method in methods.items():
        assert method["status"] == "stable"
        assert method["implementation"] == "rust"
        assert method["missing_values"] == "reject"
        assert method["errors"]
        assert method["output"]
        if name != "enbs":
            assert method["input_shape"]


def test_v1_contract_defines_non_numerical_stable_surfaces() -> None:
    """Schema, diagnostics, reporting, provenance, plotting, and CLI are governed."""
    surfaces = _contract()["surfaces"]

    assert set(surfaces) == {
        "schemas",
        "diagnostics",
        "reporting",
        "provenance",
        "plotting",
        "cli",
    }
    assert all(surface["status"] == "stable" for surface in surfaces.values())
    assert surfaces["cli"]["serialization"] == ["json", "csv"]
    assert surfaces["provenance"]["required_fields"] == [
        "voiage_version",
        "core_version",
        "method",
        "settings",
    ]
