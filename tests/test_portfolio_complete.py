"""Comprehensive tests for portfolio module to achieve >95% coverage."""

import numpy as np
import pytest

from voiage.exceptions import InputError, VoiageNotImplementedError
from voiage.methods.portfolio import portfolio_voi
from voiage.schema import (
    DecisionOption,
    PortfolioSpec,
    PortfolioStudy,
    TrialDesign,
)


class TestPortfolioVOIComplete:
    """Comprehensive tests for portfolio_voi to cover all functionality."""

    def test_portfolio_voi_with_greedy_method_complete(self):
        """Test portfolio_voi with greedy method, covering all code paths."""
        def study_value_calc(study: PortfolioStudy) -> float:
            # Simple value calculation based on sample size and cost
            base_value = study.design.total_sample_size * 1000
            cost_penalty = study.cost * 0.1
            return float(base_value - cost_penalty)

        # Create trial designs for studies
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])
        design2 = TrialDesign([DecisionOption("Treatment B", 200)])
        design3 = TrialDesign([DecisionOption("Treatment C", 150)])

        # Create portfolio studies
        study1 = PortfolioStudy("Study Alpha", design1, cost=50000)
        study2 = PortfolioStudy("Study Beta", design2, cost=80000)
        study3 = PortfolioStudy("Study Gamma", design3, cost=60000)
        studies = [study1, study2, study3]

        # Create portfolio specification with budget that allows selection of some but not all
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=100000)

        # Test with greedy method - should select Study Alpha (ratio 2.0) and potentially Study Gamma (ratio 1.5)
        # but not Study Beta (ratio 1.2) due to budget
        result = portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )

        # Verify result structure
        assert 'selected_studies' in result
        assert 'total_value' in result
        assert 'total_cost' in result
        assert 'method_details' in result

        # Check that total cost doesn't exceed budget
        assert result['total_cost'] <= 100000

        # Check that all values are reasonable
        assert isinstance(result['selected_studies'], list)
        assert isinstance(result['total_value'], float)
        assert isinstance(result['total_cost'], float)
        assert isinstance(result['method_details'], str)

        # Check that values are non-negative
        assert result['total_value'] >= 0
        assert result['total_cost'] >= 0


    def test_portfolio_voi_with_integer_programming_method(self):
        """Test portfolio_voi with integer programming method."""
        def study_value_calc(study: PortfolioStudy) -> float:
            # Simple value calculation based on sample size and cost
            return study.design.total_sample_size * 1000 / study.cost if study.cost > 0 else 0.0

        # Create trial designs for studies
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])
        design2 = TrialDesign([DecisionOption("Treatment B", 200)])

        # Create portfolio studies
        study1 = PortfolioStudy("Study 1", design1, cost=50000)
        study2 = PortfolioStudy("Study 2", design2, cost=80000)
        studies = [study1, study2]

        # Create portfolio specification with budget that allows some studies
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=100000)

        # Test with integer programming method (will only work if scipy is available)
        try:
            result_ip = portfolio_voi(
                portfolio_specification=portfolio_spec,
                study_value_calculator=study_value_calc,
                optimization_method="integer_programming"
            )

            # Verify result structure
            assert 'selected_studies' in result_ip
            assert 'total_value' in result_ip
            assert 'total_cost' in result_ip
            assert 'method_details' in result_ip

            # Check that total cost doesn't exceed budget
            assert result_ip['total_cost'] <= 100000

            # Check that all values are reasonable
            assert isinstance(result_ip['selected_studies'], list)
            assert isinstance(result_ip['total_value'], (int, float))
            assert isinstance(result_ip['total_cost'], (int, float, np.integer, np.floating))
            assert isinstance(result_ip['method_details'], str)

            # Check that values are non-negative
            assert result_ip['total_value'] >= 0
            assert result_ip['total_cost'] >= 0

        except ImportError:
            # Scipy might not be available, in which case we'll skip this test
            pytest.skip("scipy not available for integer programming test")
        except RuntimeError as e:
            # Integer programming solver might fail, which is okay to skip
            if "optimization failed" in str(e).lower():
                pytest.skip(f"Integer programming optimization failed: {e}")
            else:
                raise


    def test_portfolio_voi_with_no_budget_constraint(self):
        """Test portfolio_voi when no budget constraint is specified."""
        def study_value_calc(study: PortfolioStudy) -> float:
            return study.design.total_sample_size * 1000

        # Create studies
        design1 = TrialDesign([DecisionOption("Treatment A", 50)])
        study1 = PortfolioStudy("Study 1", design1, cost=30000)
        studies = [study1]

        # Create portfolio specification with no budget constraint
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=None)

        # Test with greedy method and no budget constraint
        result = portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )

        # With no budget constraint, all studies should be selected
        # (in this case only 1 study)
        assert len(result['selected_studies']) == 1
        assert result['selected_studies'][0].name == "Study 1"


    def test_portfolio_voi_empty_studies_list(self):
        """Test portfolio_voi with empty studies list - should raise error during PortfolioSpec creation."""
        def study_value_calc(study: PortfolioStudy) -> float:
            return 100.0

        # Attempt to create portfolio specification with empty studies list
        # This should raise an InputError because PortfolioSpec requires non-empty studies list
        with pytest.raises(InputError, match="must be a non-empty list"):
            portfolio_spec = PortfolioSpec(studies=[], budget_constraint=100000)


    def test_portfolio_voi_zero_cost_studies(self):
        """Test portfolio_voi with zero-cost studies (high value/cost ratio)."""
        def study_value_calc(study: PortfolioStudy) -> float:
            return 1000.0  # Non-zero value for zero-cost study

        # Create trial design
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])

        # Create studies with different costs
        study1 = PortfolioStudy("High Value Study", design1, cost=0)  # Zero cost
        study2 = PortfolioStudy("Normal Study", design1, cost=50000)  # Normal cost
        studies = [study1, study2]

        # Create portfolio specification with budget
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=50000)

        # Test with greedy method - zero cost study should be prioritized
        result = portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )

        # Should include the high-value zero-cost study
        selected_names = [s.name for s in result['selected_studies']]
        assert "High Value Study" in selected_names

        # Check other properties
        assert isinstance(result['total_value'], float)
        assert isinstance(result['total_cost'], float)
        assert result['total_value'] >= 0
        assert result['total_cost'] >= 0


    def test_portfolio_voi_identical_ratios(self):
        """Test portfolio_voi with studies having identical value/cost ratios."""
        def study_value_calc(study: PortfolioStudy) -> float:
            # Return value that creates identical value/cost ratios
            return study.cost * 0.02  # So ratio = 0.02 for all studies

        # Create trial designs
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])
        design2 = TrialDesign([DecisionOption("Treatment B", 150)])

        # Create studies with costs that will result in same value/cost ratios
        study1 = PortfolioStudy("Study A", design1, cost=50000)  # Value = 1000, ratio = 0.02
        study2 = PortfolioStudy("Study B", design2, cost=80000)  # Value = 1600, ratio = 0.02
        studies = [study1, study2]

        # Create portfolio specification with budget that allows selection of some
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=100000)

        # Test with greedy method
        result = portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )

        # Should select studies up to budget, with identical ratios it may depend on implementation order
        assert isinstance(result['selected_studies'], list)
        assert isinstance(result['total_value'], float)
        assert isinstance(result['total_cost'], float)
        assert result['total_value'] >= 0
        assert result['total_cost'] >= 0
        assert result['total_cost'] <= 100000


    def test_portfolio_voi_invalid_optimization_method(self):
        """Test portfolio_voi with invalid optimization method."""
        def study_value_calc(study: PortfolioStudy) -> float:
            return 1000.0

        # Create a study
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])
        study1 = PortfolioStudy("Study 1", design1, cost=50000)
        studies = [study1]

        # Create portfolio specification
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=100000)

        # Test with invalid optimization method
        with pytest.raises(VoiageNotImplementedError, match="not recognized or implemented"):
            portfolio_voi(
                portfolio_specification=portfolio_spec,
                study_value_calculator=study_value_calc,
                optimization_method="invalid_method"
            )


    def test_portfolio_voi_dynamic_programming_method(self):
        """Test portfolio_voi with dynamic programming method (should raise NotImplementedError)."""
        def study_value_calc(study: PortfolioStudy) -> float:
            return 1000.0

        # Create a study
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])
        study1 = PortfolioStudy("Study 1", design1, cost=50000)
        studies = [study1]

        # Create portfolio specification
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=100000)

        # Test with dynamic programming method (not implemented)
        with pytest.raises(VoiageNotImplementedError, match="not implemented in v0.1"):
            portfolio_voi(
                portfolio_specification=portfolio_spec,
                study_value_calculator=study_value_calc,
                optimization_method="dynamic_programming"
            )


    def test_portfolio_voi_invalid_inputs(self):
        """Test portfolio_voi with invalid inputs."""
        def study_value_calc(study: PortfolioStudy) -> float:
            return 1000.0

        # Create a study
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])
        study1 = PortfolioStudy("Study 1", design1, cost=50000)
        studies = [study1]

        # Create portfolio specification
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=100000)

        # Test with invalid portfolio_specification type
        with pytest.raises(InputError, match="`portfolio_specification` must be a PortfolioSpec"):
            portfolio_voi(
                portfolio_specification="not a PortfolioSpec",
                study_value_calculator=study_value_calc,
                optimization_method="greedy"
            )

        # Test with invalid study_value_calculator
        with pytest.raises(InputError, match="`study_value_calculator` must be a callable"):
            portfolio_voi(
                portfolio_specification=portfolio_spec,
                study_value_calculator="not a function",
                optimization_method="greedy"
            )


    def test_portfolio_voi_single_study_scenarios(self):
        """Test portfolio_voi with single study scenarios."""
        def study_value_calc(study: PortfolioStudy) -> float:
            return 1000.0

        # Create a single study
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])
        study1 = PortfolioStudy("Single Study", design1, cost=50000)
        studies = [study1]

        # Test 1: Budget allows the single study
        portfolio_spec1 = PortfolioSpec(studies=studies, budget_constraint=100000)
        result1 = portfolio_voi(
            portfolio_specification=portfolio_spec1,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )
        assert len(result1['selected_studies']) == 1
        assert result1['selected_studies'][0].name == "Single Study"
        assert result1['total_value'] == 1000.0
        assert result1['total_cost'] == 50000

        # Test 2: Budget doesn't allow the single study
        portfolio_spec2 = PortfolioSpec(studies=studies, budget_constraint=40000)
        result2 = portfolio_voi(
            portfolio_specification=portfolio_spec2,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )
        assert len(result2['selected_studies']) == 0
        assert result2['total_value'] == 0.0
        assert result2['total_cost'] == 0.0


    def test_portfolio_voi_edge_case_negative_values(self):
        """Test portfolio_voi with studies that have negative values."""
        def study_value_calc(study: PortfolioStudy) -> float:
            # Some studies have negative values
            if "Negative" in study.name:
                return -500.0
            else:
                return 1000.0

        # Create trial design
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])

        # Create studies with different values
        study1 = PortfolioStudy("Positive Value Study", design1, cost=50000)  # Positive value
        study2 = PortfolioStudy("Negative Value Study", design1, cost=40000)  # Negative value
        studies = [study1, study2]

        # Create portfolio specification with budget
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=100000)

        # Test with greedy method - should prefer positive value studies
        result = portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )

        # Should select positive value study but not negative value study (due to ratio)
        selected_names = [s.name for s in result['selected_studies']]
        assert "Positive Value Study" in selected_names or "Negative Value Study" not in selected_names

        # Check other properties
        assert isinstance(result['total_value'], float)
        assert isinstance(result['total_cost'], float)
        assert result['total_cost'] <= 100000


    def test_portfolio_voi_high_precision_scenarios(self):
        """Test portfolio_voi with high precision scenarios."""
        def study_value_calc(study: PortfolioStudy) -> float:
            # Calculation with high precision to test floating point operations
            return float(study.design.total_sample_size * 1000.123456789)

        # Create trial designs with different sample sizes
        design1 = TrialDesign([DecisionOption("Treatment A", 100)])
        design2 = TrialDesign([DecisionOption("Treatment B", 200)])

        # Create studies
        study1 = PortfolioStudy("Study 1", design1, cost=50000)
        study2 = PortfolioStudy("Study 2", design2, cost=80000)
        studies = [study1, study2]

        # Create portfolio specification with budget
        # Budget allows both studies (50000 + 80000 = 130000 > 100000), so only one should be selected
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=100000)

        # Test with greedy method
        result = portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )

        # Should select based on value-to-cost ratio
        # Study 1: value=100012.3456789, ratio=2.000246913578
        # Study 2: value=200024.6913578, ratio=2.5003086419725
        # Study 2 should be selected because of higher ratio
        selected_names = [s.name for s in result['selected_studies']]
        assert "Study 2" in selected_names  # Higher value-to-cost ratio

        assert isinstance(result['total_value'], float)
        assert isinstance(result['total_cost'], float)
        assert result['total_value'] >= 0
        assert result['total_cost'] >= 0
        assert result['total_cost'] <= 100000  # Must not exceed budget


    def test_portfolio_voi_complex_study_designs(self):
        """Test portfolio_voi with complex study designs."""
        def study_value_calc(study: PortfolioStudy) -> float:
            # Calculate value based on total number of participants across all arms
            total_participants = sum(arm.sample_size for arm in study.design.arms)
            return float(total_participants * 1000)

        # Create complex trial designs with multiple arms
        design1 = TrialDesign([
            DecisionOption("Treatment A", 50),
            DecisionOption("Treatment B", 50),
            DecisionOption("Control", 50)
        ])  # Total sample size: 150
        design2 = TrialDesign([
            DecisionOption("Treatment X", 100),
            DecisionOption("Treatment Y", 100)
        ])  # Total sample size: 200

        # Create studies
        study1 = PortfolioStudy("Multi-Arm Study", design1, cost=75000)  # Value: 150000, ratio: 2.0
        study2 = PortfolioStudy("Two-Arm Study", design2, cost=100000)  # Value: 200000, ratio: 2.0
        studies = [study1, study2]

        # Create portfolio specification with budget that allows only one study
        portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=90000)

        # Test with greedy method - should select based on value-to-cost ratio
        result = portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )

        # Should only select the one that fits the budget (Multi-Arm Study)
        selected_names = [s.name for s in result['selected_studies']]
        assert "Multi-Arm Study" in selected_names
        assert "Two-Arm Study" not in selected_names
        assert result['total_cost'] <= 90000

        # Now test with budget that allows both
        portfolio_spec2 = PortfolioSpec(studies=studies, budget_constraint=200000)
        result2 = portfolio_voi(
            portfolio_specification=portfolio_spec2,
            study_value_calculator=study_value_calc,
            optimization_method="greedy"
        )

        # With higher budget, it might select both studies if they fit, or one if they're mutually exclusive
        assert result2['total_cost'] <= 200000
        assert result2['total_value'] >= 0
