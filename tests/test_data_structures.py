# tests/test_data_structures.py

"""Unit tests for the core data structures in voiage."""

import numpy as np
import pytest

import xarray as xr
from voiage.core.data_structures import (
    DynamicSpec,
    ValueArray,
    PortfolioSpec,
    PortfolioStudy,
    ParameterSet,
    DecisionOption,
    TrialDesign,
    TrialArm,
)
from voiage.exceptions import DimensionMismatchError, InputError


class TestValueArray:
    def test_init_and_properties(self):
        dataset = xr.Dataset(
            {
                "net_benefit": (("n_samples", "n_strategies"), np.array([[1, 2], [3, 4]])),
            },
            coords={
                "n_samples": [0, 1],
                "n_strategies": [0, 1],
                "strategy": ("n_strategies", ["Strategy A", "Strategy B"]),
            },
        )
        va = ValueArray(dataset=dataset)
        np.testing.assert_array_equal(va.values, np.array([[1, 2], [3, 4]]))
        assert va.strategy_names == ["Strategy A", "Strategy B"]
        assert va.n_samples == 2
        assert va.n_strategies == 2

    def test_post_init_validations(self):
        with pytest.raises(InputError, match="must be a xarray.Dataset"):
            ValueArray(dataset="not a dataset")

        with pytest.raises(InputError, match="must have a 'n_samples' dimension"):
            ValueArray(dataset=xr.Dataset())

        with pytest.raises(InputError, match="must have a 'n_strategies' dimension"):
            ValueArray(dataset=xr.Dataset(coords={"n_samples": [0, 1]}))

        with pytest.raises(InputError, match="must have a 'net_benefit' data variable"):
            ValueArray(dataset=xr.Dataset(coords={"n_samples": [0, 1], "n_strategies": [0, 1]}))


class TestParameterSet:
    def test_init_and_properties(self):
        dataset = xr.Dataset(
            {
                "p1": ("n_samples", np.array([1, 2, 3])),
                "p2": ("n_samples", np.array([4, 5, 6])),
            },
            coords={"n_samples": [0, 1, 2]},
        )
        ps = ParameterSet(dataset=dataset)
        assert ps.n_samples == 3
        assert ps.parameter_names == ["p1", "p2"]
        np.testing.assert_array_equal(ps.parameters["p1"], np.array([1, 2, 3]))

    def test_post_init_validations(self):
        with pytest.raises(InputError, match="must be a xarray.Dataset"):
            ParameterSet(dataset="not a dataset")

        with pytest.raises(InputError, match="must have a 'n_samples' dimension"):
            ParameterSet(dataset=xr.Dataset())


class TestTrialArm:
    @pytest.mark.parametrize(
        "name, sample_size", [("Treatment A", 100), ("Control", 50)]
    )
    def test_init(self, name, sample_size):
        arm = TrialArm(name=name, sample_size=sample_size)
        assert arm.name == name
        assert arm.sample_size == sample_size

    def test_post_init_validations(self):
        with pytest.raises(InputError, match="'name' must be a non-empty string"):
            TrialArm(name="", sample_size=100)

        with pytest.raises(
            InputError, match="'sample_size' must be a positive integer"
        ):
            TrialArm(name="Treatment A", sample_size=0)

        with pytest.raises(
            InputError, match="'sample_size' must be a positive integer"
        ):
            TrialArm(name="Treatment A", sample_size=-1)


class TestTrialDesign:
    def test_init_and_properties(self):
        arm1 = TrialArm(name="Treatment A", sample_size=100)
        arm2 = TrialArm(name="Control", sample_size=100)
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
            InputError,
            match="DecisionOption names within a TrialDesign must be unique",
        ):
            arm1 = TrialArm(name="Treatment A", sample_size=100)
            TrialDesign(arms=[arm1, arm1])


class TestPortfolioStudy:
    def test_init(self):
        design = TrialDesign(arms=[TrialArm(name="T1", sample_size=50)])
        study = PortfolioStudy(name="Study A", design=design, cost=1000)
        assert study.name == "Study A"
        assert study.design == design
        assert study.cost == 1000

    def test_post_init_validations(self):
        design = TrialDesign(arms=[TrialArm(name="T1", sample_size=50)])
        with pytest.raises(InputError, match="'name' must be a non-empty string"):
            PortfolioStudy(name="", design=design, cost=1000)

        with pytest.raises(InputError, match="'design' must be a TrialDesign object"):
            PortfolioStudy(name="Study A", design="not a design", cost=1000)

        with pytest.raises(InputError, match="'cost' must be a non-negative number"):
            PortfolioStudy(name="Study A", design=design, cost=-100)


class TestPortfolioSpec:
    def test_init_and_properties(self):
        study1 = PortfolioStudy(
            name="S1", design=TrialDesign(arms=[TrialArm("T1", 50)]), cost=100
        )
        study2 = PortfolioStudy(
            name="S2", design=TrialDesign(arms=[TrialArm("T2", 50)]), cost=200
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
                name="S1", design=TrialDesign(arms=[TrialArm("T1", 50)]), cost=100
            )
            PortfolioSpec(studies=[study1, study1])

        with pytest.raises(
            InputError, match="'budget_constraint' must be a non-negative number"
        ):
            study1 = PortfolioStudy(
                name="S1", design=TrialDesign(arms=[TrialArm("T1", 50)]), cost=100
            )
            PortfolioSpec(studies=[study1], budget_constraint=-100)

    def test_optional_budget_constraint(self):
        study1 = PortfolioStudy(
            name="S1", design=TrialDesign(arms=[TrialArm("T1", 50)]), cost=100
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
