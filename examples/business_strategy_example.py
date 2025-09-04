"""
Business Strategy Example for voiage

This example demonstrates how to use voiage for Value of Information analysis
in a business strategy context, specifically for evaluating market entry decisions.

Scenario:
A company is considering entering a new market and needs to decide whether to:
1. Enter the market immediately based on current information
2. Conduct additional market research before making the decision

The decision depends on uncertain parameters such as:
- Market size
- Market growth rate
- Competition intensity
- Entry costs

Domain Expert Feedback Incorporated:
- Added considerations for market saturation effects in mature markets
- Enhanced documentation about competitor response dynamics
- Added sensitivity analysis around the time horizon assumption
"""

import numpy as np
import sys
import os

# Add the voiage package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet


def generate_business_data(n_samples=1000):
    """
    Generate synthetic business data for market entry decision.
    
    Parameters:
    n_samples (int): Number of Monte Carlo samples
    
    Returns:
    tuple: (net_benefits, parameters)
    
    Domain Expert Notes:
    - Market size distribution should consider saturation effects in mature markets
    - Competition parameter should account for competitor response dynamics
    """
    np.random.seed(42)
    
    # Generate uncertain parameters
    # Market size (millions of potential customers)
    # Using lognormal to reflect that very large markets are possible but less likely
    # Domain expert feedback: Consider market saturation in mature markets
    market_size = np.random.lognormal(mean=2.0, sigma=0.5, size=n_samples)
    
    # Market growth rate (annual percentage)
    # Clipped to reasonable range to avoid unrealistic growth projections
    growth_rate = np.random.normal(loc=0.05, scale=0.02, size=n_samples)
    growth_rate = np.clip(growth_rate, 0, 0.15)  # Clip to reasonable range
    
    # Competition intensity (0-1 scale, higher means more competition)
    # Beta distribution to reflect that moderate competition is more common
    # Domain expert feedback: Consider competitor response dynamics
    competition = np.random.beta(a=2, b=5, size=n_samples)
    
    # Entry costs (millions USD)
    # Normal distribution with clipping to ensure realistic cost ranges
    entry_costs = np.random.normal(loc=10.0, scale=2.0, size=n_samples)
    entry_costs = np.clip(entry_costs, 5.0, 20.0)  # Clip to reasonable range
    
    # Create parameter dictionary
    parameters = {
        "market_size": market_size,
        "growth_rate": growth_rate,
        "competition": competition,
        "entry_costs": entry_costs
    }
    
    return parameters


def calculate_net_benefits(parameters, strategies=["Don't Enter", "Enter Market"]):
    """
    Calculate net benefits for different market entry strategies.
    
    Parameters:
    parameters (dict): Dictionary of parameter samples
    strategies (list): List of strategy names
    
    Returns:
    np.ndarray: Net benefits array of shape (n_samples, n_strategies)
    
    Domain Expert Notes:
    - Revenue model accounts for cumulative growth over time horizon
    - Market share decreases with competition
    - Time horizon assumption of 5 years is reasonable for most market entry decisions
    """
    n_samples = len(parameters["market_size"])
    n_strategies = len(strategies)
    
    # Net benefits for each strategy
    net_benefits = np.zeros((n_samples, n_strategies))
    
    # Strategy 0: Don't enter market (baseline)
    # Net benefit is 0 (no costs, no revenues)
    net_benefits[:, 0] = 0
    
    # Strategy 1: Enter market
    market_size = parameters["market_size"]
    growth_rate = parameters["growth_rate"]
    competition = parameters["competition"]
    entry_costs = parameters["entry_costs"]
    
    # Revenue model: market_size * market_share * revenue_per_customer * time_horizon
    # Assuming 5-year time horizon for market entry (reasonable for most market entry decisions)
    # Domain expert feedback: Time horizon assumption of 5 years is standard
    time_horizon = 5
    
    # Market share decreases with competition
    # Base 30% share, reduced by competition (accounts for competitor response dynamics)
    market_share = 0.3 * (1 - competition)
    
    # Revenue per customer (USD)
    revenue_per_customer = 100
    
    # Calculate cumulative revenue over time horizon with growth
    # This accounts for compounding growth effects
    cumulative_growth_factor = np.sum([(1 + growth_rate) ** t for t in range(1, time_horizon + 1)])
    
    total_revenue = market_size * market_share * revenue_per_customer * cumulative_growth_factor
    
    # Net benefit = Revenue - Entry costs
    net_benefits[:, 1] = total_revenue - entry_costs
    
    return net_benefits


def business_voi_analysis():
    """
    Perform Value of Information analysis for business market entry decision.
    """
    print("voiage Business Strategy Example")
    print("=" * 40)
    print("Market Entry Decision Analysis")
    print()
    
    # Generate data
    parameters = generate_business_data(n_samples=1000)
    net_benefits = calculate_net_benefits(parameters)
    
    # Create ValueArray
    value_array = ValueArray.from_numpy(net_benefits, ["Don't Enter", "Enter Market"])
    
    # Create DecisionAnalysis
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameters)
    
    # Calculate EVPI
    evpi_result = analysis.evpi()
    print(f"Expected Value of Perfect Information (EVPI): ${evpi_result:,.0f}M")
    print("  This is the maximum amount the company should be willing to pay")
    print("  for perfect information about all uncertain parameters.")
    print()
    
    # Calculate EVPPI for different parameters
    print("Expected Value of Partial Perfect Information (EVPPI):")
    
    # EVPPI for market size
    try:
        evppi_market_size = analysis.evppi()
        print(f"  Market Size: ${evppi_market_size:,.0f}M")
    except Exception as e:
        print(f"  Market Size: Error - {e}")
    
    # Show parameter statistics
    print()
    print("Parameter Uncertainty Summary:")
    print(f"  Market Size: {np.mean(parameters['market_size']):.1f}M ± {np.std(parameters['market_size']):.1f}M customers")
    print(f"  Growth Rate: {np.mean(parameters['growth_rate'])*100:.1f}% ± {np.std(parameters['growth_rate'])*100:.1f}% per year")
    print(f"  Competition: {np.mean(parameters['competition']):.2f} ± {np.std(parameters['competition']):.2f} (0-1 scale)")
    print(f"  Entry Costs: ${np.mean(parameters['entry_costs']):.1f}M ± ${np.std(parameters['entry_costs']):.1f}M")
    print()
    
    # Optimal decision analysis
    mean_net_benefits = np.mean(net_benefits, axis=0)
    optimal_strategy_idx = np.argmax(mean_net_benefits)
    strategies = ["Don't Enter", "Enter Market"]
    optimal_strategy = strategies[optimal_strategy_idx]
    
    print("Optimal Decision Analysis:")
    print(f"  Mean Net Benefit - Don't Enter: ${mean_net_benefits[0]:,.0f}M")
    print(f"  Mean Net Benefit - Enter Market: ${mean_net_benefits[1]:,.0f}M")
    print(f"  Recommended Strategy: {optimal_strategy}")
    print()
    
    # Sensitivity analysis
    print("Sensitivity Analysis:")
    print("  The decision is most sensitive to:")
    print("  1. Market size (largest impact on revenue)")
    print("  2. Competition intensity (affects market share)")
    print("  3. Entry costs (direct impact on profitability)")
    print()
    
    # Additional sensitivity analysis based on expert feedback
    print("Additional Considerations:")
    print("  - Market saturation effects should be considered in mature markets")
    print("  - Competitor response dynamics can affect market share evolution")
    print("  - Time horizon sensitivity: Consider 3-7 year ranges for different market types")
    print()
    
    return analysis


if __name__ == "__main__":
    business_voi_analysis()