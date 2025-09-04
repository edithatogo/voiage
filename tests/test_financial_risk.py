"""Tests for financial risk analysis components."""

import numpy as np
import pytest

from voiage.financial.risk_analysis import (
    calculate_value_at_risk,
    calculate_conditional_value_at_risk,
    calculate_sharpe_ratio,
    monte_carlo_var,
    stress_testing
)


def test_calculate_value_at_risk():
    """Test Value at Risk calculation."""
    # Generate normal returns with known properties
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 10000)  # Mean 0.1%, std 2%
    
    # Test 95% VaR
    var_95 = calculate_value_at_risk(returns, 0.95)
    # For normal distribution, 95% VaR should be approximately 1.645 * std
    expected_var_95 = 1.645 * 0.02
    assert abs(var_95 - expected_var_95) < 0.005  # Allow some tolerance


def test_calculate_value_at_risk_invalid_inputs():
    """Test Value at Risk calculation with invalid inputs."""
    # Empty returns array
    with pytest.raises(ValueError):
        calculate_value_at_risk([])
    
    # Invalid confidence level
    returns = [0.01, -0.02, 0.03]
    with pytest.raises(ValueError):
        calculate_value_at_risk(returns, 1.1)
    
    with pytest.raises(ValueError):
        calculate_value_at_risk(returns, -0.1)


def test_calculate_conditional_value_at_risk():
    """Test Conditional Value at Risk calculation."""
    # Generate normal returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 10000)
    
    # Test 95% CVaR
    cvar_95 = calculate_conditional_value_at_risk(returns, 0.95)
    # CVaR should be greater than VaR for normal distribution
    var_95 = calculate_value_at_risk(returns, 0.95)
    assert cvar_95 >= var_95


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Generate returns with positive mean
    np.random.seed(42)
    returns = np.random.normal(0.005, 0.02, 1000)  # Mean 0.5%, std 2%
    risk_free_rate = 0.001  # 0.1% risk-free rate
    
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    assert isinstance(sharpe, float)
    
    # Test with zero volatility (constant returns) - should return inf
    constant_returns = [0.01] * 100
    sharpe_const = calculate_sharpe_ratio(constant_returns, 0.005)
    # When std is zero and mean excess return > 0, should return inf
    # But due to floating point precision, it might be a very large number
    assert sharpe_const > 1000000


def test_monte_carlo_var():
    """Test Monte Carlo VaR calculation."""
    # Test with realistic parameters
    initial_value = 100000  # $100,000 portfolio
    expected_return = 0.08   # 8% expected annual return
    volatility = 0.15        # 15% annual volatility
    time_horizon = 1.0       # 1 year
    
    var_mc = monte_carlo_var(
        initial_value, expected_return, volatility, time_horizon, 0.95, 10000
    )
    
    # VaR should be positive
    assert var_mc > 0
    
    # VaR should be reasonable (not too large compared to portfolio value)
    assert var_mc < initial_value * 0.5


def test_monte_carlo_var_invalid_inputs():
    """Test Monte Carlo VaR with invalid inputs."""
    # Invalid initial value
    with pytest.raises(ValueError):
        monte_carlo_var(-1000, 0.08, 0.15, 1.0)
    
    # Invalid confidence level
    with pytest.raises(ValueError):
        monte_carlo_var(100000, 0.08, 0.15, 1.0, 1.1)
    
    # Invalid number of simulations
    with pytest.raises(ValueError):
        monte_carlo_var(100000, 0.08, 0.15, 1.0, 0.95, -1000)


def test_stress_testing():
    """Test stress testing functionality."""
    # Generate base returns
    np.random.seed(42)
    base_returns = np.random.normal(0.001, 0.02, 1000)
    
    # Define stress scenarios
    stress_scenarios = {
        "moderate_stress": 1.5,
        "severe_stress": 2.5
    }
    
    results = stress_testing(base_returns, stress_scenarios)
    
    # Should have base case plus all scenarios
    assert "base_case" in results
    assert "moderate_stress" in results
    assert "severe_stress" in results
    
    # Stress scenarios should have higher VaR
    assert results["moderate_stress"] >= results["base_case"]
    assert results["severe_stress"] >= results["moderate_stress"]


if __name__ == "__main__":
    test_calculate_value_at_risk()
    test_calculate_value_at_risk_invalid_inputs()
    test_calculate_conditional_value_at_risk()
    test_calculate_sharpe_ratio()
    test_monte_carlo_var()
    test_monte_carlo_var_invalid_inputs()
    test_stress_testing()
    print("All financial risk analysis tests passed!")