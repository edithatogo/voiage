"""Public contracts for Rust-authoritative expected opportunity loss."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from voiage import _runtime, expected_loss
from voiage.exceptions import InputError
from voiage.methods.basic import evpi
from voiage.schema import ValueArray

ROOT = Path(__file__).parents[1]
ASSURANCE_SCHEMA = ROOT / (
    "specs/v1/schemas/statistical-assurance-envelope.schema.json"
)


def test_expected_loss_matches_analytical_fixture_and_evpi() -> None:
    values = np.array([[10.0, 12.0], [11.0, 9.0], [13.0, 14.0]])

    result = expected_loss(values)

    # Another comprehensive test deliberately reloads ``methods.basic`` to
    # exercise optional-import behavior.  Assert the public result contract
    # without retaining a pre-reload class identity.
    assert result.__class__.__name__ == "ExpectedLossResult"
    assert result.optimal_strategy_index == 1
    assert result.sample_count == 3
    assert result.strategy_count == 2
    assert result.strategy_names == ["Strategy 1", "Strategy 2"]
    np.testing.assert_allclose(
        result.expected_net_benefit_by_strategy,
        [34.0 / 3.0, 35.0 / 3.0],
    )
    np.testing.assert_allclose(
        result.expected_opportunity_loss_by_strategy,
        [1.0, 2.0 / 3.0],
    )
    assert result.minimum_expected_opportunity_loss == pytest.approx(evpi(values))
    assert result.assurance is not None
    assert result.assurance.reporting_class == "sample-average"
    assert result.assurance.variance_estimate == pytest.approx(4.0 / 3.0)
    assert result.assurance.monte_carlo_standard_error == pytest.approx(2.0 / 3.0)
    assert result.assurance.replications == 1
    assert result.assurance.budget.draws == 3
    assert result.assurance.budget.evaluations == 6
    assert result.assurance.stopping_reason == "fixed-budget"


def test_expected_loss_assurance_conforms_to_the_portable_schema() -> None:
    result = expected_loss(np.array([[10.0, 12.0], [11.0, 9.0], [13.0, 14.0]]))

    assurance = result.assurance

    assert assurance is not None
    payload = assurance.to_dict()
    Draft202012Validator(
        json.loads(ASSURANCE_SCHEMA.read_text(encoding="utf-8"))
    ).validate(payload)
    assert payload["confidence_interval"]["lower"] == 0.0
    assert payload["confidence_interval"]["upper"] == pytest.approx(
        2.0 / 3.0 + 1.959963984540054 * 2.0 / 3.0
    )
    json.dumps(payload, allow_nan=False)


def test_one_sample_expected_loss_discloses_unavailable_uncertainty() -> None:
    result = expected_loss(np.array([[1.0, 3.0]]))

    assurance = result.assurance

    assert assurance is not None
    assert assurance.variance_estimate is None
    assert assurance.monte_carlo_standard_error is None
    assert assurance.confidence_interval is None
    assert "single supplied draw" in assurance.bias_assessment


def test_expected_loss_accepts_value_array_and_uses_first_tie() -> None:
    values = ValueArray.from_numpy(np.array([[0.0, 2.0], [2.0, 0.0]]))

    result = expected_loss(values)

    assert result.optimal_strategy_index == 0
    assert result.strategy_names == values.strategy_names
    np.testing.assert_array_equal(
        result.expected_opportunity_loss_by_strategy,
        [1.0, 1.0],
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0, np.nan]]),
        np.empty((0, 2)),
        np.empty((2, 0)),
    ],
)
def test_expected_loss_rejects_invalid_inputs(values: np.ndarray) -> None:
    with pytest.raises(InputError):
        expected_loss(values)


def test_expected_loss_serializes_to_the_closed_result_schema() -> None:
    result = expected_loss(np.array([[10.0, 12.0], [11.0, 9.0], [13.0, 14.0]]))

    payload = result.to_dict(
        analysis_id="expected-loss-001",
        decision_problem_id="decision-001",
    )

    schema = json.loads(
        (
            ROOT / "specs/core-api/schemas/v1/results/expected-loss.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(payload)
    assert payload["analysis_type"] == "expected_loss"
    assert payload["method"] == "expected-opportunity-loss"
    assert payload["reporting"]["statistical_assurance"] == (result.assurance.to_dict())
    json.dumps(payload, allow_nan=False)


def test_expected_loss_serialization_requires_identifiers() -> None:
    result = expected_loss(np.array([[1.0, 2.0]]))

    with pytest.raises(InputError, match="analysis_id"):
        result.to_dict(analysis_id="", decision_problem_id="decision-001")


def test_expected_loss_has_no_python_numerical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = ModuleNotFoundError("No module named 'voiage._core'")

    def fail_import(_: str) -> None:
        raise missing

    monkeypatch.setattr(_runtime, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError, match="voiage._core"):
        expected_loss(np.array([[1.0, 2.0]]))
