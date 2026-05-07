"""Tests for implementation-adjusted VOI."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.implementation import (
    ImplementationAdjustedResult,
    value_of_implementation,
)
from voiage.schema import ValueArray


@pytest.fixture
def implementation_value_array() -> ValueArray:
    values = np.array(
        [
            [10.0, 12.0],
            [8.0, 11.0],
            [9.0, 10.0],
            [7.0, 9.0],
        ]
    )
    return ValueArray.from_numpy(values, ["A", "B"])


def test_implementation_contract_files_are_valid_json() -> None:
    """The experimental implementation contract scaffold should parse cleanly."""
    contract_root = Path("specs/frontier/implementation/v1")
    for relative_path in [
        "schemas/implementation-adjusted-result.schema.json",
        "examples/value-of-implementation.example.json",
    ]:
        payload = json.loads((contract_root / relative_path).read_text())
        assert isinstance(payload, dict)


def test_implementation_fixture_manifest_and_payload_are_deterministic() -> None:
    """The deterministic fixture set should anchor the implementation contract."""
    fixture_root = Path("specs/frontier/implementation/v1/fixtures")
    manifest = json.loads((fixture_root / "manifest.json").read_text())
    assert manifest["version"] == "v1"
    assert manifest["status"] == "fixture-backed"
    normative = cast("list[dict[str, object]]", manifest["normative"])
    assert len(normative) == 1

    entry = normative[0]
    assert entry["name"] == "screening program implementation adjustment"
    assert entry["method_family"] == "value_of_implementation"
    assert entry["input_artifact"] == "normative/implementation-adjusted-input.json"
    assert entry["expected_output_artifact"] == "normative/value-of-implementation.json"
    assert entry["tolerance_policy"] == "exact"

    input_payload = json.loads(
        (fixture_root / "normative" / "implementation-adjusted-input.json").read_text()
    )
    expected = json.loads(
        (fixture_root / "normative" / "value-of-implementation.json").read_text()
    )
    assert isinstance(input_payload, dict)
    assert input_payload["strategy_names"] == ["A", "B"]

    value_array = ValueArray.from_numpy(
        np.asarray(input_payload["values"], dtype=float),
        cast("list[str]", input_payload["strategy_names"]),
    )
    result = value_of_implementation(
        value_array,
        uptake=float(input_payload["uptake"]),
        adherence=float(input_payload["adherence"]),
        coverage=float(input_payload["coverage"]),
        implementation_delay=float(input_payload["implementation_delay"]),
        implementation_uncertainty=float(input_payload["implementation_uncertainty"]),
        discount_rate=float(input_payload["discount_rate"]),
        time_horizon=float(input_payload["time_horizon"]),
    )

    assert (
        result.baseline_expected_net_benefits.tolist()
        == expected["baseline_expected_net_benefits"]
    )
    assert (
        result.adjusted_expected_net_benefits.tolist()
        == expected["adjusted_expected_net_benefits"]
    )
    assert result.value == pytest.approx(float(expected["value"]))
    assert result.reporting == expected["reporting"]
    assert result.diagnostics == expected["diagnostics"]


def test_value_of_implementation_adjusts_expected_value(
    implementation_value_array: ValueArray,
) -> None:
    """Implementation-adjusted VOI should shrink the realized value."""
    result = value_of_implementation(
        implementation_value_array,
        uptake=0.8,
        adherence=0.9,
        coverage=0.75,
        implementation_delay=2.0,
        implementation_uncertainty=0.1,
        discount_rate=0.05,
        time_horizon=5.0,
    )

    assert isinstance(result, ImplementationAdjustedResult)
    assert result.baseline_optimal_strategy_name == "B"
    assert result.adjusted_optimal_strategy_name == "B"
    assert result.implementation_multiplier == pytest.approx(
        0.8 * 0.9 * 0.75 * 0.9 * (1.0 / (1.0 + 0.05) ** 2)
    )
    assert result.adjusted_expected_net_benefits.tolist() == pytest.approx(
        [3.746938775510204, 4.628571428571429]
    )
    assert result.value == pytest.approx(5.871428571428571)
    assert result.method_maturity == "experimental"
    assert result.reporting["reporting_standard"] == "CHEERS-VOI"
    assert result.reporting["analysis_type"] == "value_of_implementation"


def test_decision_analysis_wraps_value_of_implementation(
    implementation_value_array: ValueArray,
) -> None:
    """DecisionAnalysis should expose the implementation-adjusted method."""
    analysis = DecisionAnalysis(implementation_value_array)

    result = analysis.value_of_implementation(
        uptake=0.8,
        adherence=0.9,
        coverage=0.75,
        implementation_delay=2.0,
        implementation_uncertainty=0.1,
        discount_rate=0.05,
        time_horizon=5.0,
    )

    assert isinstance(result, ImplementationAdjustedResult)
    assert result.baseline_optimal_strategy_name == "B"


def test_value_of_implementation_scales_population_value(
    implementation_value_array: ValueArray,
) -> None:
    """Population and horizon inputs should scale the reported value."""
    result = value_of_implementation(
        implementation_value_array,
        uptake=0.8,
        adherence=0.9,
        coverage=0.75,
        implementation_delay=2.0,
        implementation_uncertainty=0.1,
        discount_rate=0.05,
        time_horizon=5.0,
        population=1000.0,
    )

    assert result.value > 0.0
    assert result.population == pytest.approx(1000.0)
    assert result.time_horizon == pytest.approx(5.0)


def test_value_of_implementation_rejects_bad_probability_like_inputs(
    implementation_value_array: ValueArray,
) -> None:
    """Probability-like inputs should be validated consistently."""
    with pytest.raises(InputError, match="uptake"):
        value_of_implementation(implementation_value_array, uptake=np.nan)

    with pytest.raises(InputError, match="adherence"):
        value_of_implementation(implementation_value_array, adherence=1.1)

    with pytest.raises(InputError, match="coverage"):
        value_of_implementation(implementation_value_array, coverage=-0.1)

    with pytest.raises(InputError, match="implementation_uncertainty"):
        value_of_implementation(
            implementation_value_array, implementation_uncertainty=1.5
        )

    with pytest.raises(InputError, match="discount_rate"):
        value_of_implementation(implementation_value_array, discount_rate=-0.1)

    with pytest.raises(InputError, match="time_horizon"):
        value_of_implementation(implementation_value_array, time_horizon=0.0)

    with pytest.raises(InputError, match="population"):
        value_of_implementation(implementation_value_array, population=0.0)


def test_value_of_implementation_rejects_invalid_inputs(
    implementation_value_array: ValueArray,
) -> None:
    """Invalid implementation inputs should be rejected early."""
    with pytest.raises(InputError, match="ValueArray"):
        value_of_implementation(cast("ValueArray", "not a value array"))

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

    with pytest.raises(InputError, match="2D"):
        value_of_implementation(three_dimensional_values)

    with pytest.raises(InputError, match="uptake"):
        value_of_implementation(implementation_value_array, uptake=1.5)

    with pytest.raises(InputError, match="delay"):
        value_of_implementation(implementation_value_array, implementation_delay=-1.0)
