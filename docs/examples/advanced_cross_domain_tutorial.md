# Advanced Cross-Domain VOI Analysis Tutorial

This tutorial demonstrates how to apply voiage to complex cross-domain problems, including engineering design optimization and financial portfolio management.

## Engineering Design Optimization

In engineering contexts, VOI can help optimize design decisions under uncertainty by quantifying the value of additional testing or simulation.

### Complex System Design Example

This example shows how to use voiage for evaluating design alternatives for a complex engineering system with multiple uncertain parameters.

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet

def generate_engineering_data(n_samples=1000):
    """
    Generate synthetic engineering data for system design decision.
    
    Parameters:
    n_samples (int): Number of Monte Carlo samples
    
    Returns:
    dict: Parameter samples
    """
    np.random.seed(42)
    
    # Generate uncertain parameters for engineering system
    # Material strength (MPa)
    material_strength = np.random.normal(loc=400, scale=50, size=n_samples)
    
    # Manufacturing tolerance (mm)
    manufacturing_tolerance = np.random.lognormal(mean=-2, sigma=0.5, size=n_samples)
    
    # Operating temperature (°C)
    operating_temperature = np.random.normal(loc=70, scale=15, size=n_samples)
    
    # Load conditions (N)
    load_conditions = np.random.normal(loc=10000, scale=2000, size=n_samples)
    
    # Maintenance costs ($)
    maintenance_costs = np.random.gamma(shape=2, scale=5000, size=n_samples)
    
    # Create parameter dictionary
    parameters = {
        "material_strength": material_strength,
        "manufacturing_tolerance": manufacturing_tolerance,
        "operating_temperature": operating_temperature,
        "load_conditions": load_conditions,
        "maintenance_costs": maintenance_costs
    }
    
    return parameters

def calculate_engineering_net_benefits(parameters, strategies=["Design A", "Design B", "Design C"]):
    """
    Calculate net benefits for different engineering design strategies.
    
    Parameters:
    parameters (dict): Dictionary of parameter samples
    strategies (list): List of strategy names
    
    Returns:
    np.ndarray: Net benefits array of shape (n_samples, n_strategies)
    """
    n_samples = len(parameters["material_strength"])
    n_strategies = len(strategies)
    
    # Net benefits for each strategy
    net_benefits = np.zeros((n_samples, n_strategies))
    
    # Extract parameters
    material_strength = parameters["material_strength"]
    manufacturing_tolerance = parameters["manufacturing_tolerance"]
    operating_temperature = parameters["operating_temperature"]
    load_conditions = parameters["load_conditions"]
    maintenance_costs = parameters["maintenance_costs"]
    
    # Strategy 0: Design A (conservative design)
    # Lower performance but higher reliability
    performance_A = 0.8 * material_strength / 400  # Normalized performance
    reliability_A = np.exp(-manufacturing_tolerance * 100)  # Reliability decreases with tolerance
    operating_cost_A = 1.2 * maintenance_costs  # Higher maintenance costs
    failure_cost_A = np.where(
        (load_conditions > 0.9 * material_strength) | (operating_temperature > 85),
        100000, 0  # High failure cost if overloaded or overheated
    )
    
    net_benefits[:, 0] = performance_A * 1000000 * reliability_A - operating_cost_A - failure_cost_A
    
    # Strategy 1: Design B (balanced design)
    # Moderate performance and reliability
    performance_B = material_strength / 400
    reliability_B = np.exp(-manufacturing_tolerance * 50)
    operating_cost_B = maintenance_costs
    failure_cost_B = np.where(
        (load_conditions > 1.0 * material_strength) | (operating_temperature > 90),
        150000, 0
    )
    
    net_benefits[:, 1] = performance_B * 1000000 * reliability_B - operating_cost_B - failure_cost_B
    
    # Strategy 2: Design C (aggressive design)
    # Higher performance but lower reliability
    performance_C = 1.2 * material_strength / 400
    reliability_C = np.exp(-manufacturing_tolerance * 25)
    operating_cost_C = 0.8 * maintenance_costs
    failure_cost_C = np.where(
        (load_conditions > 1.1 * material_strength) | (operating_temperature > 95),
        200000, 0
    )
    
    net_benefits[:, 2] = performance_C * 1000000 * reliability_C - operating_cost_C - failure_cost_C
    
    return net_benefits

def engineering_voi_analysis():
    """
    Perform Value of Information analysis for engineering design decision.
    """
    print("voiage Engineering Design Example")
    print("=" * 40)
    print("Complex System Design Analysis")
    print()
    
    # Generate data
    parameters = generate_engineering_data(n_samples=1000)
    net_benefits = calculate_engineering_net_benefits(parameters)
    
    # Create ValueArray
    value_array = ValueArray.from_numpy(net_benefits, ["Design A", "Design B", "Design C"])
    
    # Create DecisionAnalysis
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameters)
    
    # Calculate EVPI
    evpi_result = analysis.evpi()
    print(f"Expected Value of Perfect Information (EVPI): ${evpi_result:,.0f}")
    print("  This is the maximum amount that should be willing to pay")
    print("  for perfect information about all uncertain parameters.")
    print()
    
    # Calculate EVPPI for key parameters
    print("Expected Value of Partial Perfect Information (EVPPI):")
    
    # EVPPI for material strength
    try:
        evppi_strength = analysis.evppi("material_strength")
        print(f"  Material Strength: ${evppi_strength:,.0f}")
    except Exception as e:
        print(f"  Material Strength: Error - {e}")
    
    # EVPPI for manufacturing tolerance
    try:
        evppi_tolerance = analysis.evppi("manufacturing_tolerance")
        print(f"  Manufacturing Tolerance: ${evppi_tolerance:,.0f}")
    except Exception as e:
        print(f"  Manufacturing Tolerance: Error - {e}")
    
    # Show parameter statistics
    print()
    print("Parameter Uncertainty Summary:")
    print(f"  Material Strength: {np.mean(parameters['material_strength']):.0f} ± {np.std(parameters['material_strength']):.0f} MPa")
    print(f"  Manufacturing Tolerance: {np.mean(parameters['manufacturing_tolerance'])*1000:.1f} ± {np.std(parameters['manufacturing_tolerance'])*1000:.1f} mm")
    print(f"  Operating Temperature: {np.mean(parameters['operating_temperature']):.1f} ± {np.std(parameters['operating_temperature']):.1f} °C")
    print(f"  Load Conditions: {np.mean(parameters['load_conditions']):,.0f} ± {np.std(parameters['load_conditions']):,.0f} N")
    print(f"  Maintenance Costs: ${np.mean(parameters['maintenance_costs']):,.0f} ± ${np.std(parameters['maintenance_costs']):,.0f}")
    print()
    
    # Optimal decision analysis
    mean_net_benefits = np.mean(net_benefits, axis=0)
    optimal_strategy_idx = np.argmax(mean_net_benefits)
    strategies = ["Design A", "Design B", "Design C"]
    optimal_strategy = strategies[optimal_strategy_idx]
    
    print("Optimal Decision Analysis:")
    for i, strategy in enumerate(strategies):
        print(f"  Mean Net Benefit - {strategy}: ${mean_net_benefits[i]:,.0f}")
    print(f"  Recommended Strategy: {optimal_strategy}")
    print()
    
    return analysis

# Run the analysis
if __name__ == "__main__":
    engineering_voi_analysis()
```

## Financial Portfolio Management

In finance, VOI can help optimize portfolio allocation by quantifying the value of additional market research or economic forecasting.

### Portfolio Optimization Example

This example demonstrates how to use voiage for evaluating portfolio allocation strategies under market uncertainty.

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet

def generate_financial_data(n_samples=1000):
    """
    Generate synthetic financial data for portfolio allocation decision.
    
    Parameters:
    n_samples (int): Number of Monte Carlo samples
    
    Returns:
    dict: Parameter samples
    """
    np.random.seed(42)
    
    # Generate uncertain parameters for financial markets
    # Stock market returns (annual)
    stock_returns = np.random.normal(loc=0.08, scale=0.15, size=n_samples)
    
    # Bond market returns (annual)
    bond_returns = np.random.normal(loc=0.03, scale=0.05, size=n_samples)
    
    # Real estate returns (annual)
    real_estate_returns = np.random.normal(loc=0.06, scale=0.12, size=n_samples)
    
    # Commodity returns (annual)
    commodity_returns = np.random.normal(loc=0.05, scale=0.18, size=n_samples)
    
    # Inflation rate (annual)
    inflation_rate = np.random.normal(loc=0.02, scale=0.02, size=n_samples)
    
    # Economic policy uncertainty index
    policy_uncertainty = np.random.gamma(shape=2, scale=25, size=n_samples)
    
    # Create parameter dictionary
    parameters = {
        "stock_returns": stock_returns,
        "bond_returns": bond_returns,
        "real_estate_returns": real_estate_returns,
        "commodity_returns": commodity_returns,
        "inflation_rate": inflation_rate,
        "policy_uncertainty": policy_uncertainty
    }
    
    return parameters

def calculate_financial_net_benefits(parameters, strategies=["Conservative", "Balanced", "Aggressive"]):
    """
    Calculate net benefits for different portfolio allocation strategies.
    
    Parameters:
    parameters (dict): Dictionary of parameter samples
    strategies (list): List of strategy names
    
    Returns:
    np.ndarray: Net benefits array of shape (n_samples, n_strategies)
    """
    n_samples = len(parameters["stock_returns"])
    n_strategies = len(strategies)
    
    # Net benefits for each strategy (risk-adjusted returns)
    net_benefits = np.zeros((n_samples, n_strategies))
    
    # Extract parameters
    stock_returns = parameters["stock_returns"]
    bond_returns = parameters["bond_returns"]
    real_estate_returns = parameters["real_estate_returns"]
    commodity_returns = parameters["commodity_returns"]
    inflation_rate = parameters["inflation_rate"]
    policy_uncertainty = parameters["policy_uncertainty"]
    
    # Adjust returns for inflation
    real_stock_returns = stock_returns - inflation_rate
    real_bond_returns = bond_returns - inflation_rate
    real_real_estate_returns = real_estate_returns - inflation_rate
    real_commodity_returns = commodity_returns - inflation_rate
    
    # Strategy 0: Conservative portfolio (60% bonds, 20% stocks, 20% real estate)
    conservative_returns = (
        0.6 * real_bond_returns +
        0.2 * real_stock_returns +
        0.2 * real_real_estate_returns
    )
    
    # Risk penalty for conservative portfolio (lower risk penalty)
    risk_penalty_conservative = 0.5 * (
        0.6**2 * (bond_returns - real_bond_returns)**2 +
        0.2**2 * (stock_returns - real_stock_returns)**2 +
        0.2**2 * (real_estate_returns - real_real_estate_returns)**2
    )
    
    # Policy uncertainty impact (reduces all portfolios but less for conservative)
    uncertainty_impact_conservative = -0.1 * policy_uncertainty
    
    net_benefits[:, 0] = conservative_returns - risk_penalty_conservative + uncertainty_impact_conservative
    
    # Strategy 1: Balanced portfolio (40% stocks, 30% bonds, 20% real estate, 10% commodities)
    balanced_returns = (
        0.4 * real_stock_returns +
        0.3 * real_bond_returns +
        0.2 * real_real_estate_returns +
        0.1 * real_commodity_returns
    )
    
    # Risk penalty for balanced portfolio
    risk_penalty_balanced = (
        0.4**2 * (stock_returns - real_stock_returns)**2 +
        0.3**2 * (bond_returns - real_bond_returns)**2 +
        0.2**2 * (real_estate_returns - real_real_estate_returns)**2 +
        0.1**2 * (commodity_returns - real_commodity_returns)**2
    )
    
    # Policy uncertainty impact
    uncertainty_impact_balanced = -0.2 * policy_uncertainty
    
    net_benefits[:, 1] = balanced_returns - risk_penalty_balanced + uncertainty_impact_balanced
    
    # Strategy 2: Aggressive portfolio (70% stocks, 15% real estate, 15% commodities)
    aggressive_returns = (
        0.7 * real_stock_returns +
        0.15 * real_real_estate_returns +
        0.15 * real_commodity_returns
    )
    
    # Risk penalty for aggressive portfolio (higher risk penalty)
    risk_penalty_aggressive = 1.5 * (
        0.7**2 * (stock_returns - real_stock_returns)**2 +
        0.15**2 * (real_estate_returns - real_real_estate_returns)**2 +
        0.15**2 * (commodity_returns - real_commodity_returns)**2
    )
    
    # Policy uncertainty impact (higher impact for aggressive)
    uncertainty_impact_aggressive = -0.3 * policy_uncertainty
    
    net_benefits[:, 2] = aggressive_returns - risk_penalty_aggressive + uncertainty_impact_aggressive
    
    return net_benefits

def financial_voi_analysis():
    """
    Perform Value of Information analysis for portfolio allocation decision.
    """
    print("voiage Financial Portfolio Example")
    print("=" * 40)
    print("Portfolio Allocation Analysis")
    print()
    
    # Generate data
    parameters = generate_financial_data(n_samples=1000)
    net_benefits = calculate_financial_net_benefits(parameters)
    
    # Create ValueArray
    value_array = ValueArray.from_numpy(net_benefits, ["Conservative", "Balanced", "Aggressive"])
    
    # Create DecisionAnalysis
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameters)
    
    # Calculate EVPI
    evpi_result = analysis.evpi()
    print(f"Expected Value of Perfect Information (EVPI): ${evpi_result:.4f}")
    print("  This is the maximum amount that should be willing to pay")
    print("  for perfect information about all uncertain parameters.")
    print()
    
    # Calculate EVPPI for key parameters
    print("Expected Value of Partial Perfect Information (EVPPI):")
    
    # EVPPI for stock returns
    try:
        evppi_stocks = analysis.evppi("stock_returns")
        print(f"  Stock Returns: ${evppi_stocks:.4f}")
    except Exception as e:
        print(f"  Stock Returns: Error - {e}")
    
    # EVPPI for policy uncertainty
    try:
        evppi_policy = analysis.evppi("policy_uncertainty")
        print(f"  Policy Uncertainty: ${evppi_policy:.4f}")
    except Exception as e:
        print(f"  Policy Uncertainty: Error - {e}")
    
    # Show parameter statistics
    print()
    print("Parameter Uncertainty Summary:")
    print(f"  Stock Returns: {np.mean(parameters['stock_returns']*100):.1f}% ± {np.std(parameters['stock_returns']*100):.1f}%")
    print(f"  Bond Returns: {np.mean(parameters['bond_returns']*100):.1f}% ± {np.std(parameters['bond_returns']*100):.1f}%")
    print(f"  Real Estate Returns: {np.mean(parameters['real_estate_returns']*100):.1f}% ± {np.std(parameters['real_estate_returns']*100):.1f}%")
    print(f"  Commodity Returns: {np.mean(parameters['commodity_returns']*100):.1f}% ± {np.std(parameters['commodity_returns']*100):.1f}%")
    print(f"  Inflation Rate: {np.mean(parameters['inflation_rate']*100):.1f}% ± {np.std(parameters['inflation_rate']*100):.1f}%")
    print(f"  Policy Uncertainty: {np.mean(parameters['policy_uncertainty']):.1f} ± {np.std(parameters['policy_uncertainty']):.1f}")
    print()
    
    # Optimal decision analysis
    mean_net_benefits = np.mean(net_benefits, axis=0)
    optimal_strategy_idx = np.argmax(mean_net_benefits)
    strategies = ["Conservative", "Balanced", "Aggressive"]
    optimal_strategy = strategies[optimal_strategy_idx]
    
    print("Optimal Decision Analysis:")
    for i, strategy in enumerate(strategies):
        print(f"  Mean Risk-Adjusted Return - {strategy}: {mean_net_benefits[i]*100:.2f}%")
    print(f"  Recommended Strategy: {optimal_strategy}")
    print()
    
    return analysis

# Run the analysis
if __name__ == "__main__":
    financial_voi_analysis()
```

## Best Practices for Cross-Domain Applications

### 1. Parameter Selection and Modeling

When applying voiage to new domains, carefully consider which parameters are both uncertain and influential to decision outcomes:

- **Identify Key Uncertainties**: Focus on parameters that have the greatest impact on outcomes
- **Use Appropriate Distributions**: Choose probability distributions that reflect the nature of uncertainty in your domain
- **Validate with Domain Experts**: Engage domain experts to validate parameter ranges and relationships

### 2. Model Validation and Calibration

Ensure your outcome models accurately represent the domain:

- **Cross-Validation**: Validate your models against historical data when available
- **Sensitivity Analysis**: Perform sensitivity analysis to understand which parameters drive value of information
- **Scenario Testing**: Test your models under different scenarios to ensure robustness

### 3. Interpretation and Communication

Effectively communicate VOI results to domain stakeholders:

- **Contextualize Results**: Present VOI values in the context of domain-specific decision-making
- **Visualize Uncertainty**: Use appropriate visualizations to show parameter uncertainty and its impact
- **Provide Actionable Insights**: Translate VOI results into concrete recommendations for data collection or research

### 4. Domain-Specific Considerations

Different domains may require specific methodological adaptations:

- **Engineering**: Account for physical constraints and safety margins
- **Finance**: Consider risk measures and regulatory constraints
- **Public Policy**: Incorporate multiple stakeholder perspectives and social costs/benefits
- **Supply Chain**: Model logistics complexities and supplier relationships

## Conclusion

voiage's flexible architecture allows it to be applied across various domains. By following the patterns demonstrated in these examples and adhering to domain-specific best practices, you can adapt voiage to your specific needs while maintaining the rigorous analytical framework of Value of Information analysis.