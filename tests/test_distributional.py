"""Tests for distributional and equity-weighted VOI."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.distributional import (
    DistributionalEquityResult,
    value_of_distributional_equity,
)
from voiage.schema import ValueArray


@pytest.fixture
def distributional_value_array() -> ValueArray:
    values = np.array(
        [
            [10.0, 2.0],
            [8.0, 4.0],
            [1.0, 11.0],
            [2.0, 9.0],
        ]
    )
    return ValueArray.from_numpy(values, ["A", "B"])


def test_distributional_contract_files_are_valid_json() -> None:
    """The experimental distributional contract scaffold should parse cleanly."""
    contract_root = Path("specs/frontier/distributional/v1")
    for relative_path in [
        "schemas/distributional-equity-result.schema.json",
        "examples/value-of-distributional-equity.example.json",
    ]:
        payload = json.loads((contract_root / relative_path).read_text())
        assert isinstance(payload, dict)


def test_distributional_fixture_manifest_and_payload_are_deterministic() -> None:
    """The deterministic fixture set should anchor the distributional contract."""
    fixture_root = Path("specs/frontier/distributional/v1/fixtures")
    manifest = json.loads((fixture_root / "manifest.json").read_text())
    assert manifest["version"] == "v1"
    assert manifest["status"] == "fixture-backed"
    normative = cast("list[dict[str, object]]", manifest["normative"])
    assert len(normative) == 1

    entry = normative[0]
    assert entry["name"] == "screening program distributional equity comparison"
    assert entry["method_family"] == "value_of_distributional_equity"
    assert entry["input_artifact"] == "normative/distributional-equity-input.json"
    assert (
        entry["expected_output_artifact"]
        == "normative/value-of-distributional-equity.json"
    )
    assert entry["tolerance_policy"] == "exact"

    input_payload = json.loads(
        (fixture_root / "normative" / "distributional-equity-input.json").read_text()
    )
    expected = json.loads(
        (fixture_root / "normative" / "value-of-distributional-equity.json").read_text()
    )
    assert isinstance(input_payload, dict)
    assert input_payload["strategy_names"] == ["A", "B"]
    assert input_payload["subgroups"] == ["low", "low", "high", "high"]

    value_array = ValueArray.from_numpy(
        np.asarray(input_payload["values"], dtype=float),
        cast("list[str]", input_payload["strategy_names"]),
    )
    result = value_of_distributional_equity(
        value_array,
        subgroups=cast("list[object]", input_payload["subgroups"]),
        strategy_names=cast("list[str]", input_payload["strategy_names"]),
        equity_weights=cast("dict[str, float]", input_payload["equity_weights"]),
    )

    assert result.subgroup_labels == expected["subgroup_labels"]
    assert result.subgroup_weights.tolist() == expected["subgroup_weights"]
    assert result.equity_weights.tolist() == expected["equity_weights"]
    assert result.value == pytest.approx(float(expected["value"]))
    assert result.reporting == expected["reporting"]
    assert result.diagnostics == expected["diagnostics"]


def test_distributional_equity_value_uses_subgroup_weights(
    distributional_value_array: ValueArray,
) -> None:
    """Distributional VOI should weight subgroup-specific optima explicitly."""
    result = value_of_distributional_equity(
        distributional_value_array,
        subgroups=["low", "low", "high", "high"],
        equity_weights={"low": 0.25, "high": 0.75},
    )

    assert isinstance(result, DistributionalEquityResult)
    assert result.subgroup_labels == ["high", "low"]
    assert result.subgroup_optimal_strategy_names == ["B", "A"]
    assert result.overall_optimal_strategy_name == "B"
    np.testing.assert_allclose(
        result.equity_weighted_expected_net_benefits,
        np.array([3.375, 8.25]),
    )
    assert result.social_welfare_optimal_strategy_name == "B"
    assert result.social_welfare_value == pytest.approx(8.25)
    assert result.value == pytest.approx(3.0)
    assert result.method_maturity == "experimental"
    assert result.reporting["reporting_standard"] == "CHEERS-VOI"
    assert result.reporting["analysis_type"] == "value_of_distributional_equity"


def test_distributional_equity_supports_numeric_binning(
    distributional_value_array: ValueArray,
) -> None:
    """Numeric subgroup labels should be binned before the equity summary."""
    result = value_of_distributional_equity(
        distributional_value_array,
        subgroups=np.array([0.1, 0.2, 0.9, 1.0]),
        n_bins=2,
    )

    assert result.subgroup_labels == ["bin_1", "bin_2"]
    assert result.value == pytest.approx(3.0)


def test_distributional_equity_rejects_invalid_binning(
    distributional_value_array: ValueArray,
) -> None:
    """Binning should reject invalid bin counts and non-numeric labels."""
    with pytest.raises(InputError, match="at least 2"):
        value_of_distributional_equity(
            distributional_value_array,
            subgroups=np.array([0.1, 0.2, 0.9, 1.0]),
            n_bins=1,
        )

    with pytest.raises(InputError, match="numeric subgroups"):
        value_of_distributional_equity(
            distributional_value_array,
            subgroups=["low", "low", "high", "high"],
            n_bins=2,
        )


def test_distributional_equity_rejects_bad_weight_vectors(
    distributional_value_array: ValueArray,
) -> None:
    """Equity-weight normalization should reject malformed vectors."""
    with pytest.raises(InputError, match="one value per subgroup"):
        value_of_distributional_equity(
            distributional_value_array,
            subgroups=["low", "low", "high", "high"],
            equity_weights=[0.5, 0.25, 0.25],
        )

    with pytest.raises(InputError, match="non-negative"):
        value_of_distributional_equity(
            distributional_value_array,
            subgroups=["low", "low", "high", "high"],
            equity_weights={"low": 0.5, "high": -0.25},
        )

    with pytest.raises(InputError, match="positive value"):
        value_of_distributional_equity(
            distributional_value_array,
            subgroups=["low", "low", "high", "high"],
            equity_weights={"low": 0.0, "high": 0.0},
        )


def test_decision_analysis_wraps_distributional_equity(
    distributional_value_array: ValueArray,
) -> None:
    """DecisionAnalysis should expose the distributional equity method."""
    analysis = DecisionAnalysis(distributional_value_array)

    result = analysis.value_of_distributional_equity(
        subgroups=["low", "low", "high", "high"],
        equity_weights={"low": 0.25, "high": 0.75},
    )

    assert isinstance(result, DistributionalEquityResult)
    assert result.social_welfare_optimal_strategy_name == "B"


def test_distributional_equity_rejects_invalid_inputs(
    distributional_value_array: ValueArray,
) -> None:
    """Bad inputs should fail before any summary is produced."""
    three_dimensional_values = ValueArray(
        dataset=xr.Dataset(
            {
                "net_benefit": (
                    ("n_samples", "n_strategies", "n_perspectives"),
                    np.ones((4, 2, 2)),
                )
            },
            coords={
                "n_samples": np.arange(4),
                "n_strategies": np.arange(2),
                "strategy": ("n_strategies", ["A", "B"]),
            },
        )
    )

    with pytest.raises(InputError, match="ValueArray"):
        value_of_distributional_equity(cast("ValueArray", "not a value array"), ["a"])

    with pytest.raises(InputError, match="2D"):
        value_of_distributional_equity(
            three_dimensional_values,
            ["a", "a", "b", "b"],
        )

    with pytest.raises(InputError, match="length"):
        value_of_distributional_equity(distributional_value_array, ["a"])

    with pytest.raises(InputError, match="weights"):
        value_of_distributional_equity(
            distributional_value_array,
            ["a", "a", "b", "b"],
            equity_weights={"a": 1.0},
        )
