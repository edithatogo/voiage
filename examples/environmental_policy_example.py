"""
Environmental Policy Example for voiage

This example demonstrates how to use voiage for Value of Information analysis
in an environmental policy context, specifically for evaluating pollution control measures.

Scenario:
A government agency is considering implementing new regulations to reduce air pollution
and needs to decide whether to:
1. Implement regulations immediately based on current information
2. Conduct additional environmental research before making the decision

The decision depends on uncertain parameters such as:
- Effectiveness of pollution control technologies
- Economic costs of implementation
- Public health benefits
- Compliance rates
"""

import numpy as np
import sys
import os

# Add the voiage package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet


def generate_environmental_data(n_samples=1000):
    """
    Generate synthetic environmental data for pollution control decision.
    
    Parameters:
    n_samples (int): Number of Monte Carlo samples
    
    Returns:
    tuple: (net_benefits, parameters)
    """
    np.random.seed(42)
    
    # Generate uncertain parameters
    # Technology effectiveness (0-1 scale, higher means more effective)
    tech_effectiveness = np.random.beta(a=3, b=2, size=n_samples)
    
    # Implementation costs (millions USD)
    implementation_costs = np.random.normal(loc=15.0, scale=5.0, size=n_samples)
    implementation_costs = np.clip(implementation_costs, 5.0, 30.0)  # Clip to reasonable range
    
    # Public health benefits (millions USD in avoided healthcare costs)
    health_benefits = np.random.lognormal(mean=3.5, sigma=0.7, size=n_samples)
    
    # Compliance rate (0-1 scale, higher means better compliance)
    compliance_rate = np.random.beta(a=4, b=2, size=n_samples)
    
    # Create parameter dictionary
    parameters = {
        "tech_effectiveness": tech_effectiveness,
        "implementation_costs": implementation_costs,
        "health_benefits": health_benefits,
        "compliance_rate": compliance_rate
    }
    
    return parameters


def calculate_net_benefits(parameters, strategies=["No Regulation", "Implement Regulation"]):
    """
    Calculate net benefits for different environmental policy strategies.
    
    Parameters:
    parameters (dict): Dictionary of parameter samples
    strategies (list): List of strategy names
    
    Returns:
    np.ndarray: Net benefits array of shape (n_samples, n_strategies)
    """
    n_samples = len(parameters["tech_effectiveness"])
    n_strategies = len(strategies)
    
    # Net benefits for each strategy
    net_benefits = np.zeros((n_samples, n_strategies))
    
    # Strategy 0: No regulation (baseline)
    # Net benefit is 0 (no costs, no benefits)
    net_benefits[:, 0] = 0
    
    # Strategy 1: Implement regulation
    tech_effectiveness = parameters["tech_effectiveness"]
    implementation_costs = parameters["implementation_costs"]
    health_benefits = parameters["health_benefits"]
    compliance_rate = parameters["compliance_rate"]
    
    # Adjust health benefits based on technology effectiveness and compliance
    adjusted_health_benefits = health_benefits * tech_effectiveness * compliance_rate
    
    # Net benefit = Health benefits - Implementation costs
    net_benefits[:, 1] = adjusted_health_benefits - implementation_costs
    
    return net_benefits


def environmental_voi_analysis():
    """
    Perform Value of Information analysis for environmental policy decision.
    """
    print("voiage Environmental Policy Example")
    print("=" * 40)
    print("Pollution Control Decision Analysis")
    print()
    
    # Generate data
    parameters = generate_environmental_data(n_samples=1000)
    net_benefits = calculate_net_benefits(parameters)
    
    # Create ValueArray
    value_array = ValueArray.from_numpy(net_benefits, ["No Regulation", "Implement Regulation"])
    
    # Create DecisionAnalysis
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameters)
    
    # Calculate EVPI
    evpi_result = analysis.evpi()
    print(f"Expected Value of Perfect Information (EVPI): ${evpi_result:,.0f}M")
    print("  This is the maximum amount the agency should be willing to pay")
    print("  for perfect information about all uncertain parameters.")
    print()
    
    # Calculate EVPPI for different parameters
    print("Expected Value of Partial Perfect Information (EVPPI):")
    
    # EVPPI for all parameters
    try:
        evppi_result = analysis.evppi()
        print(f"  All Parameters: ${evppi_result:,.0f}M")
    except Exception as e:
        print(f"  All Parameters: Error - {e}")
    
    # Show parameter statistics
    print()
    print("Parameter Uncertainty Summary:")
    print(f"  Technology Effectiveness: {np.mean(parameters['tech_effectiveness']):.2f} ± {np.std(parameters['tech_effectiveness']):.2f} (0-1 scale)")
    print(f"  Implementation Costs: ${np.mean(parameters['implementation_costs']):.1f}M ± ${np.std(parameters['implementation_costs']):.1f}M")
    print(f"  Health Benefits: ${np.mean(parameters['health_benefits']):.1f}M ± ${np.std(parameters['health_benefits']):.1f}M")
    print(f"  Compliance Rate: {np.mean(parameters['compliance_rate']):.2f} ± {np.std(parameters['compliance_rate']):.2f} (0-1 scale)")
    print()
    
    # Optimal decision analysis
    mean_net_benefits = np.mean(net_benefits, axis=0)
    optimal_strategy_idx = np.argmax(mean_net_benefits)
    strategies = ["No Regulation", "Implement Regulation"]
    optimal_strategy = strategies[optimal_strategy_idx]
    
    print("Optimal Decision Analysis:")
    print(f"  Mean Net Benefit - No Regulation: ${mean_net_benefits[0]:,.0f}M")
    print(f"  Mean Net Benefit - Implement Regulation: ${mean_net_benefits[1]:,.0f}M")
    print(f"  Recommended Strategy: {optimal_strategy}")
    print()
    
    # Sensitivity analysis
    print("Sensitivity Analysis:")
    print("  The decision is most sensitive to:")
    print("  1. Public health benefits (largest impact on net benefits)")
    print("  2. Technology effectiveness (affects benefit realization)")
    print("  3. Implementation costs (direct impact on net benefits)")
    print()
    
    return analysis


if __name__ == "__main__":
    environmental_voi_analysis()