"""Tests for schema helpers and validators."""

from __future__ import annotations

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.schema import (
    DecisionOption,
    DynamicSpec,
    ParameterSet,
    PortfolioSpec,
    PortfolioStudy,
    TrialDesign,
    ValueArray,
)


def test_value_array_from_numpy_and_copy() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    value_array = ValueArray.from_numpy(values, strategy_names=["A", "B"])
    copied = value_array.copy()

    assert value_array.n_samples == 2
    assert value_array.n_strategies == 2
    assert value_array.strategy_names == ["A", "B"]
    assert np.array_equal(value_array.numpy_values, values)
    assert copied is not value_array
    assert np.array_equal(copied.numpy_values, values)


def test_value_array_from_numpy_validates_shape() -> None:
    with pytest.raises(InputError, match="2D"):
        ValueArray.from_numpy(np.array([1.0, 2.0]))


def test_value_array_from_numpy_validates_strategy_names_length() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(InputError, match="strategy_names"):
        ValueArray.from_numpy(values, strategy_names=["only-one"])


def test_value_array_get_strategy_index_and_slice() -> None:
    value_array = ValueArray.from_numpy(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        strategy_names=["A", "B", "C"],
    )

    sliced = value_array.slice_by_strategies(["C", "A"])

    assert value_array.get_strategy_index("B") == 1
    assert sliced.strategy_names == ["C", "A"]
    assert np.array_equal(sliced.numpy_values, np.array([[3.0, 1.0], [6.0, 4.0]]))


def test_value_array_get_strategy_index_missing_name() -> None:
    value_array = ValueArray.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        strategy_names=["A", "B"],
    )

    with pytest.raises(ValueError, match="Strategy 'missing' not found"):
        value_array.get_strategy_index("missing")


def test_parameter_set_from_numpy_or_dict_with_ndarray() -> None:
    parameter_set = ParameterSet.from_numpy_or_dict(np.array([[0.1, 1.0], [0.2, 2.0]]))

    assert parameter_set.n_samples == 2
    assert parameter_set.parameter_names == ["param_0", "param_1"]
    assert np.array_equal(parameter_set.parameters["param_0"], np.array([0.1, 0.2]))
    assert np.array_equal(parameter_set.parameters["param_1"], np.array([1.0, 2.0]))


def test_parameter_set_from_numpy_or_dict_validates_ndarray_inputs() -> None:
    with pytest.raises(InputError, match="2D"):
        ParameterSet.from_numpy_or_dict(np.array([0.1, 0.2]))


def test_parameter_set_from_numpy_or_dict_with_dict_and_copy() -> None:
    parameter_set = ParameterSet.from_numpy_or_dict(
        {
            "alpha": np.array([0.1, 0.2]),
            "beta": np.array([1.0, 2.0]),
        }
    )
    copied = parameter_set.copy()

    assert set(parameter_set.parameter_names) == {"alpha", "beta"}
    assert parameter_set.n_samples == 2
    assert copied is not parameter_set
    assert np.array_equal(copied.parameters["alpha"], np.array([0.1, 0.2]))


def test_parameter_set_from_numpy_or_dict_validates_dict_inputs() -> None:
    with pytest.raises(InputError, match="cannot be empty"):
        ParameterSet.from_numpy_or_dict({})

    with pytest.raises(InputError, match="same length"):
        ParameterSet.from_numpy_or_dict(
            {
                "alpha": np.array([0.1, 0.2]),
                "beta": np.array([1.0]),
            }
        )

    with pytest.raises(InputError, match="numpy array or dictionary"):
        ParameterSet.from_numpy_or_dict("bad-input")


def test_parameter_set_subset_by_parameters() -> None:
    parameter_set = ParameterSet.from_numpy_or_dict(
        {
            "alpha": np.array([0.1, 0.2]),
            "beta": np.array([1.0, 2.0]),
            "gamma": np.array([3.0, 4.0]),
        }
    )

    subset = parameter_set.subset_by_parameters(["gamma", "alpha"])

    assert subset.parameter_names == ["gamma", "alpha"]
    assert np.array_equal(subset.parameters["gamma"], np.array([3.0, 4.0]))
    assert np.array_equal(subset.parameters["alpha"], np.array([0.1, 0.2]))


def test_parameter_set_subset_by_parameters_missing_name() -> None:
    parameter_set = ParameterSet.from_numpy_or_dict({"alpha": np.array([0.1, 0.2])})

    with pytest.raises(ValueError, match="Parameters not found: beta"):
        parameter_set.subset_by_parameters(["beta"])


def test_decision_option_validates_inputs() -> None:
    option = DecisionOption(name="Option A", sample_size=10)

    assert option.name == "Option A"
    assert option.sample_size == 10

    with pytest.raises(InputError, match="non-empty string"):
        DecisionOption(name="", sample_size=10)

    with pytest.raises(InputError, match="positive integer"):
        DecisionOption(name="Option A", sample_size=0)


def test_trial_design_total_sample_size() -> None:
    design = TrialDesign(
        arms=[
            DecisionOption(name="arm_a", sample_size=10),
            DecisionOption(name="arm_b", sample_size=15),
        ]
    )

    assert design.total_sample_size == 25


def test_trial_design_validates_arms() -> None:
    with pytest.raises(InputError, match="non-empty list of DecisionOption"):
        TrialDesign(arms=[])

    with pytest.raises(InputError, match="DecisionOption"):
        TrialDesign(arms=["not-an-option"])  # type: ignore[list-item]

    with pytest.raises(InputError, match="must be unique"):
        TrialDesign(
            arms=[
                DecisionOption(name="shared", sample_size=10),
                DecisionOption(name="shared", sample_size=15),
            ]
        )


def test_portfolio_study_validates_inputs() -> None:
    design = TrialDesign(arms=[DecisionOption(name="arm_a", sample_size=10)])
    study = PortfolioStudy(name="study", design=design, cost=5.0)

    assert study.name == "study"
    assert study.design is design
    assert study.cost == pytest.approx(5.0)

    with pytest.raises(InputError, match="non-empty string"):
        PortfolioStudy(name="", design=design, cost=5.0)

    with pytest.raises(InputError, match="TrialDesign"):
        PortfolioStudy(name="study", design="bad", cost=5.0)  # type: ignore[arg-type]

    with pytest.raises(InputError, match="non-negative number"):
        PortfolioStudy(name="study", design=design, cost=-5.0)


def test_portfolio_spec_validates_inputs() -> None:
    design = TrialDesign(arms=[DecisionOption(name="arm_a", sample_size=10)])
    study_a = PortfolioStudy(name="study_a", design=design, cost=5.0)
    study_b = PortfolioStudy(name="study_b", design=design, cost=7.5)
    spec = PortfolioSpec(
        studies=[study_a, study_b],
        budget_constraint=20.0,
    )

    assert spec.studies == [study_a, study_b]
    assert spec.budget_constraint == pytest.approx(20.0)

    with pytest.raises(InputError, match="non-empty list of PortfolioStudy"):
        PortfolioSpec(studies=[])

    with pytest.raises(InputError, match="PortfolioStudy"):
        PortfolioSpec(studies=["bad"])  # type: ignore[list-item]

    with pytest.raises(InputError, match="must be unique"):
        PortfolioSpec(studies=[study_a, study_a])

    with pytest.raises(InputError, match="non-negative number"):
        PortfolioSpec(studies=[study_a], budget_constraint=-1.0)


def test_dynamic_spec_validates_time_steps() -> None:
    spec = DynamicSpec(time_steps=[0.0, 1.0, 2.0])

    assert spec.time_steps == [0.0, 1.0, 2.0]

    with pytest.raises(InputError, match="non-empty sequence"):
        DynamicSpec(time_steps=[])

    with pytest.raises(InputError, match="must be numbers"):
        DynamicSpec(time_steps=[0.0, "bad"])  # type: ignore[list-item]


def test_decision_option_to_dict_from_dict_round_trip() -> None:
    option = DecisionOption(name="Option A", sample_size=10)

    assert DecisionOption.from_dict(option.to_dict()) == option


def test_trial_design_to_dict_from_dict_round_trip() -> None:
    design = TrialDesign(
        arms=[
            DecisionOption(name="arm_a", sample_size=10),
            DecisionOption(name="arm_b", sample_size=15),
        ]
    )

    assert TrialDesign.from_dict(design.to_dict()) == design
