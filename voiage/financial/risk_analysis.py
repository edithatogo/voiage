"""Financial risk analysis components for Value of Information analysis."""

from typing import Dict, List, Union

import numpy as np


def calculate_value_at_risk(
    returns: Union[np.ndarray, List[float]],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) for a portfolio or investment.

    Args:
        returns: Array of historical returns or simulated returns
        confidence_level: Confidence level for VaR calculation (default 0.95 = 95%)

    Returns
    -------
        float: Value at Risk at the specified confidence level

    Example:
        >>> returns = np.random.normal(0.001, 0.02, 1000)  # 1000 daily returns
        >>> var_95 = calculate_value_at_risk(returns, 0.95)
        >>> print(f"95% VaR: {var_95:.4f}")
    """
    # Convert to numpy array if needed
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)

    # Validate inputs
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1")

    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Calculate VaR as the percentile of returns
    var = -np.percentile(returns, (1 - confidence_level) * 100)

    return float(var)


def calculate_conditional_value_at_risk(
    returns: Union[np.ndarray, List[float]],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.

    Args:
        returns: Array of historical returns or simulated returns
        confidence_level: Confidence level for CVaR calculation (default 0.95 = 95%)

    Returns
    -------
        float: Conditional Value at Risk at the specified confidence level
    """
    # Convert to numpy array if needed
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)

    # Validate inputs
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1")

    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Calculate VaR first
    var = calculate_value_at_risk(returns, confidence_level)

    # Calculate CVaR as the mean of returns below the VaR threshold
    var_threshold = -var
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        # If no returns are below the threshold, return the VaR
        return var

    cvar = -np.mean(tail_returns)
    return float(cvar)


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    annualize: bool = False
) -> float:
    """
    Calculate Sharpe ratio for portfolio or investment performance.

    Args:
        returns: Array of historical returns or simulated returns
        risk_free_rate: Risk-free rate (default 0.0)
        annualize: Whether to annualize the Sharpe ratio (default False)

    Returns
    -------
        float: Sharpe ratio
    """
    # Convert to numpy array if needed
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)

    # Validate inputs
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Calculate excess returns
    excess_returns = returns - risk_free_rate

    # Calculate Sharpe ratio
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)

    if std_excess_return == 0:
        # Avoid division by zero
        return float('inf') if mean_excess_return > 0 else float('-inf') if mean_excess_return < 0 else 0.0

    sharpe_ratio = mean_excess_return / std_excess_return

    # Annualize if requested (assuming daily returns)
    if annualize:
        sharpe_ratio *= np.sqrt(252)  # 252 trading days in a year

    return float(sharpe_ratio)


def monte_carlo_var(
    initial_value: float,
    expected_return: float,
    volatility: float,
    time_horizon: float,
    confidence_level: float = 0.95,
    n_simulations: int = 10000
) -> float:
    """
    Calculate Value at Risk using Monte Carlo simulation.

    Args:
        initial_value: Initial portfolio value
        expected_return: Expected return (per time period)
        volatility: Volatility (standard deviation of returns per time period)
        time_horizon: Time horizon for VaR calculation
        confidence_level: Confidence level for VaR calculation (default 0.95 = 95%)
        n_simulations: Number of Monte Carlo simulations (default 10000)

    Returns
    -------
        float: Value at Risk from Monte Carlo simulation
    """
    # Validate inputs
    if initial_value <= 0:
        raise ValueError("Initial value must be positive")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1")

    if n_simulations <= 0:
        raise ValueError("Number of simulations must be positive")

    if time_horizon <= 0:
        raise ValueError("Time horizon must be positive")

    # Generate random returns using geometric Brownian motion
    # Generate standard normal random variables
    random_shocks = np.random.standard_normal(n_simulations)

    # Calculate terminal values using geometric Brownian motion
    # S_T = S_0 * exp((μ - σ²/2)T + σ√T * Z)
    drift = (expected_return - 0.5 * volatility**2) * time_horizon
    diffusion = volatility * np.sqrt(time_horizon) * random_shocks
    terminal_values = initial_value * np.exp(drift + diffusion)

    # Calculate portfolio value changes
    value_changes = terminal_values - initial_value

    # Calculate VaR as percentile of value changes
    var = -np.percentile(value_changes, (1 - confidence_level) * 100)

    return float(var)


def stress_testing(
    base_returns: Union[np.ndarray, List[float]],
    stress_scenarios: Dict[str, float]
) -> Dict[str, float]:
    """
    Perform stress testing on portfolio returns under different scenarios.

    Args:
        base_returns: Array of base case returns
        stress_scenarios: Dictionary mapping scenario names to stress multipliers
                         (e.g., {"recession": 1.5, "market_crash": 3.0})

    Returns
    -------
        Dict[str, float]: Dictionary mapping scenario names to stressed VaR values
    """
    # Convert to numpy array if needed
    if not isinstance(base_returns, np.ndarray):
        base_returns = np.array(base_returns)

    # Validate inputs
    if len(base_returns) == 0:
        raise ValueError("Base returns array cannot be empty")

    results = {}

    # Calculate base case VaR
    base_var = calculate_value_at_risk(base_returns)
    results["base_case"] = base_var

    # Apply stress scenarios
    for scenario_name, stress_multiplier in stress_scenarios.items():
        # Apply stress by multiplying returns by the stress multiplier
        # (This assumes that stress increases the magnitude of losses)
        stressed_returns = base_returns * stress_multiplier
        stressed_var = calculate_value_at_risk(stressed_returns)
        results[scenario_name] = stressed_var

    return results
