import numpy as np
import pytest

from voiage.experimental_design import (
    ExperimentalDesign,
    amortized_evsi,
    expected_information_gain,
    select_active_learning_batch,
    select_bayesian_design,
)


def test_expected_information_gain_reports_monte_carlo_uncertainty() -> None:
    result = expected_information_gain([1.0, 2.0, 3.0])
    assert result.estimate == pytest.approx(2.0)
    assert result.samples == 3
    assert result.standard_error == pytest.approx(1 / np.sqrt(3))


def test_bayesian_design_selection_is_cost_aware() -> None:
    selected = select_bayesian_design(
        [ExperimentalDesign("large", 10.0, 8.0), ExperimentalDesign("small", 2.0, 2.0)]
    )
    assert selected.name == "small"


def test_active_learning_and_amortized_evsi_workflows() -> None:
    assert select_active_learning_batch([0.1, 0.9, 0.4], 2).tolist() == [1, 2]
    assert amortized_evsi([11.0, 13.0, 12.0], 10.0) == pytest.approx(2.0)


@pytest.mark.parametrize("values", [[], [1.0, float("nan")]])
def test_experimental_design_inputs_are_validated(values: list[float]) -> None:
    with pytest.raises(ValueError):
        expected_information_gain(values)
