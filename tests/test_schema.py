# tests/test_data_structures.py

"""Unit tests for the core data structures in voiage."""

import numpy as np
import pytest

from voiage.schema import (
    DynamicSpec,
    ValueArray,
    PortfolioSpec,
    PortfolioStudy,
    ParameterSet,
    DecisionOption,
    TrialDesign,
)
from voiage.exceptions import DimensionMismatchError, InputError


class TestValueArray:
    @pytest.mark.parametrize(
        "values, names",
        [
            (np.array([[1, 2], [3, 4]]), ["Strategy A", "Strategy B"]),
            (np.array([[1, 2, 3], [4, 5, 6]]), ["A", "B", "C"]),
        ],
    )
    def test_init_and_properties(self, values, names):
        nba = ValueArray(values=values, strategy_names=names)
        np.testing.assert_array_equal(nba.values, values)
        assert nba.strategy_names == names
        assert nba.n_samples == values.shape[0]
        assert nba.n_strategies == values.shape[1]

    def test_post_init_validations(self):
        with pytest.raises(InputError, match="'values' must be a NumPy array"):
            ValueArray(values=[[1, 2], [3, 4]])

        with pytest.raises(DimensionMismatchError, match="'values' must be a 2D array"):
            ValueArray(values=np.array([1, 2, 3]))

        with pytest.raises(
            InputError, match="'strategy_names' must be a list of strings"
        ):
            ValueArray(values=np.array([[1, 2]]), strategy_names=["A", 1])

        with pytest.raises(DimensionMismatchError, match="Length of 'strategy_names'"):
            ValueArray(values=np.array([[1, 2]]), strategy_names=["A"])

    def test_optional_strategy_names(self):
        values = np.array([[1, 2], [3, 4]])
        nba = ValueArray(values=values)
        assert nba.strategy_names is None


class TestParameterSet:
    def test_init_and_properties(self):
        params = {"p1": np.array([1, 2, 3]), "p2": np.array([4, 5, 6])}
        psa = ParameterSet(parameters=params)
        assert psa.parameters == params
        assert psa.n_samples == 3
        assert psa.parameter_names == ["p1", "p2"]

    def test_post_init_validations(self):
        with pytest.raises(InputError, match="'parameters' must be a dictionary"):
            ParameterSet(parameters=[1, 2, 3])

        with pytest.raises(InputError, match="'parameters' dictionary cannot be empty"):
            ParameterSet(parameters={})

        with pytest.raises(
            InputError, match="Parameter names in ParameterSet dictionary must be strings"
        ):
            ParameterSet(parameters={1: np.array([1, 2])})

        with pytest.raises(InputError, match="values must be a NumPy array"):
            ParameterSet(parameters={"p1": [1, 2]})

        with pytest.raises(DimensionMismatchError, match="array must be 1D"):
            ParameterSet(parameters={"p1": np.array([[1, 2]])})

        with pytest.raises(DimensionMismatchError, match="must have the same length"):
            ParameterSet(parameters={"p1": np.array([1, 2]), "p2": np.array([3, 4, 5])})

    def test_post_init_no_samples(self):
        with pytest.raises(
            InputError,
            match="Could not determine n_samples from parameters dictionary, or dictionary contains empty arrays.",
        ):
            ParameterSet(parameters={"p1": np.array([])})

    def test_empty_psa_sample(self):
        with pytest.raises(InputError, match="cannot be empty"):
            ParameterSet(parameters={})

    def test_n_samples_property_empty_dict(self):
        with pytest.raises(InputError, match="cannot be empty"):
            ParameterSet(parameters={})

    def test_n_samples_property_empty_dict_no_post_init(self):
        psa_no_post_init = object.__new__(ParameterSet)
        object.__setattr__(psa_no_post_init, "parameters", {})
        assert psa_no_post_init.n_samples == 0

    def test_parameter_names_property_empty_dict(self):
        with pytest.raises(InputError, match="cannot be empty"):
            ParameterSet(parameters={})

    def test_parameter_names_property_empty_dict_no_post_init(self):
        psa_no_post_init = object.__new__(ParameterSet)
        object.__setattr__(psa_no_post_init, "parameters", {})
        assert psa_no_post_init.parameter_names == []

    def test_n_samples_property_no_n_samples_attribute(self):
        params = {"p1": np.array([1, 2, 3]), "p2": np.array([4, 5, 6])}
        psa = ParameterSet(parameters=params)
        # The _n_samples attribute is set in __post_init__
        assert psa.n_samples == 3
        # To test the case where _n_samples is not present, we need to bypass __post_init__
        psa_no_post_init = object.__new__(ParameterSet)
        object.__setattr__(psa_no_post_init, "parameters", params)
        assert psa_no_post_init.n_samples == 3

    def test_parameter_names_not_dict(self):
        with pytest.raises(InputError, match="'parameters' must be a dictionary"):
            ParameterSet(parameters="not a dict")

    def test_parameter_names_not_dict_no_post_init(self):
        psa_no_post_init = object.__new__(ParameterSet)
        object.__setattr__(psa_no_post_init, "parameters", "not a dict")
        assert psa_no_post_init.parameter_names == []

    def test_n_samples_not_dict(self):
        with pytest.raises(InputError, match="'parameters' must be a dictionary"):
            ParameterSet(parameters="not a dict")

    def test_n_samples_not_dict_no_post_init(self):
        psa_no_post_init = object.__new__(ParameterSet)
        object.__setattr__(psa_no_post_init, "parameters", "not a dict")
        assert psa_no_post_init.n_samples == 0


class TestDecisionOption:
    @pytest.mark.parametrize(
        "name, sample_size", [("Treatment A", 100), ("Control", 50)]
    )
    def test_init(self, name, sample_size):
        arm = DecisionOption(name=name, sample_size=sample_size)
        assert arm.name == name
        assert arm.sample_size == sample_size

    def test_post_init_validations(self):
        with pytest.raises(InputError, match="'name' must be a non-empty string"):
            DecisionOption(name="", sample_size=100)

        with pytest.raises(
            InputError, match="'sample_size' must be a positive integer"
        ):
            DecisionOption(name="Treatment A", sample_size=0)

        with pytest.raises(
            InputError, match="'sample_size' must be a positive integer"
        ):
            DecisionOption(name="Treatment A", sample_size=-1)


class TestTrialDesign:
    def test_init_and_properties(self):
        arm1 = DecisionOption(name="Treatment A", sample_size=100)
        arm2 = DecisionOption(name="Control", sample_size=100)
        td = TrialDesign(arms=[arm1, arm2])
        assert td.arms == [arm1, arm2]
        assert td.total_sample_size == 200

    def test_post_init_validations(self):
        with pytest.raises(InputError, match="'arms' must be a non-empty list"):
            TrialDesign(arms=[])

        with pytest.raises(
            InputError, match="All elements in 'arms' must be DecisionOption objects"
        ):
            TrialDesign(arms=["not an arm"])

        with pytest.raises(
            InputError, match="DecisionOption names within a TrialDesign must be unique"
        ):
            arm1 = DecisionOption(name="Treatment A", sample_size=100)
            TrialDesign(arms=[arm1, arm1])


class TestPortfolioStudy:
    def test_init(self):
        design = TrialDesign(arms=[DecisionOption(name="T1", sample_size=50)])
        study = PortfolioStudy(name="Study A", design=design, cost=1000)
        assert study.name == "Study A"
        assert study.design == design
        assert study.cost == 1000

    def test_post_init_validations(self):
        design = TrialDesign(arms=[DecisionOption(name="T1", sample_size=50)])
        with pytest.raises(InputError, match="'name' must be a non-empty string"):
            PortfolioStudy(name="", design=design, cost=1000)

        with pytest.raises(InputError, match="'design' must be a TrialDesign object"):
            PortfolioStudy(name="Study A", design="not a design", cost=1000)

        with pytest.raises(InputError, match="'cost' must be a non-negative number"):
            PortfolioStudy(name="Study A", design=design, cost=-100)


class TestPortfolioSpec:
    def test_init_and_properties(self):
        study1 = PortfolioStudy(
            name="S1", design=TrialDesign(arms=[DecisionOption("T1", 50)]), cost=100
        )
        study2 = PortfolioStudy(
            name="S2", design=TrialDesign(arms=[DecisionOption("T2", 50)]), cost=200
        )
        spec = PortfolioSpec(studies=[study1, study2], budget_constraint=300)
        assert spec.studies == [study1, study2]
        assert spec.budget_constraint == 300

    def test_post_init_validations(self):
        with pytest.raises(InputError, match="'studies' must be a non-empty list"):
            PortfolioSpec(studies=[])

        with pytest.raises(
            InputError, match="All elements in 'studies' must be PortfolioStudy objects"
        ):
            PortfolioSpec(studies=["not a study"])

        with pytest.raises(
            InputError,
            match="PortfolioStudy names within a PortfolioSpec must be unique",
        ):
            study1 = PortfolioStudy(
                name="S1", design=TrialDesign(arms=[DecisionOption("T1", 50)]), cost=100
            )
            PortfolioSpec(studies=[study1, study1])

        with pytest.raises(
            InputError, match="'budget_constraint' must be a non-negative number"
        ):
            study1 = PortfolioStudy(
                name="S1", design=TrialDesign(arms=[DecisionOption("T1", 50)]), cost=100
            )
            PortfolioSpec(studies=[study1], budget_constraint=-100)

    def test_optional_budget_constraint(self):
        study1 = PortfolioStudy(
            name="S1", design=TrialDesign(arms=[DecisionOption("T1", 50)]), cost=100
        )
        spec = PortfolioSpec(studies=[study1])
        assert spec.budget_constraint is None


class TestDynamicSpec:
    def test_init(self):
        spec = DynamicSpec(time_steps=[0, 1, 5])
        assert spec.time_steps == [0, 1, 5]

    def test_post_init_validations(self):
        with pytest.raises(
            InputError, match="'time_steps' must be a non-empty sequence"
        ):
            DynamicSpec(time_steps=[])

        with pytest.raises(
            InputError, match="All elements in 'time_steps' must be numbers"
        ):
            DynamicSpec(time_steps=[0, 1, "2"])
