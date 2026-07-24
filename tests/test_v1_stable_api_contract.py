"""Executable governance checks for the normative v1.0 public API contract."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import voiage

CONTRACT_PATH = Path("specs/v1/stable-api.json")
V1_RELEASE_DIGESTS = {
    "README.md": "fb1304871c009eb386c8291bc55db0faf007bba204134bdf06d84d8e6e0c3ddc",
    "binding-matrix.json": "554fd6248a65bf15ea74c7564aff4b530291c5b54d65536bfa7f475dace3d150",
    "compatibility-policy.json": "3027c936cfced144113ada048b42c1ac9e461d9634849b7a45add618946bce8a",
    "compatibility-policy.schema.json": "3b02cbaf713f4cf3b02c61c45d331b9eea3fad1e63c94277495c34398c65a993",
    "extension-packaging.json": "5a91eff0d45d025f96958f50ac202ff2f4f4b54d1afab3125c944b0aac392fcc",
    "extension-policy.json": "a030224ceb0f65bc62123136814cfcc0e05c2aed7633308636cfad94c611b234",
    "extension-surface-policy.json": "6a9b14d55b40de2be11984b49a37e2a387c343db88f3a92320e7d9f9a5dbd343",
    "python-runtime-inventory.json": "5a852675c64d014ebdd62a9f077f19c10191223342aa7053cf15f053ebfe51e0",
    "stable-api.json": "ff77ce326e29d17354dbe274b30d42e0ed89ae80efb7700bba94eef8ae68249d",
}


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_v1_contract_files_match_the_v1_0_0_release_snapshot() -> None:
    """Published v1 contract files must remain byte-for-byte immutable."""
    directory = CONTRACT_PATH.parent

    assert {path.name for path in directory.iterdir() if path.is_file()} == set(
        V1_RELEASE_DIGESTS
    )
    for name, expected_digest in V1_RELEASE_DIGESTS.items():
        assert sha256((directory / name).read_bytes()).hexdigest() == expected_digest


def test_v1_contract_defines_supported_runtime_and_behavior() -> None:
    """The stable contract must freeze runtime and numerical expectations."""
    contract = _contract()

    assert contract["contract_version"] == "1.0.0"
    assert contract["python"] == {
        "versions": ["3.12", "3.13", "3.14"],
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
