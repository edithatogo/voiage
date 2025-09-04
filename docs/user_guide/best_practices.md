# Best Practices for Value of Information Analysis

This guide provides comprehensive best practices for conducting Value of Information (VOI) analysis using voiage, covering methodological considerations, implementation tips, and common pitfalls to avoid.

## Table of Contents

1. [Methodological Best Practices](#methodological-best-practices)
2. [Implementation Best Practices](#implementation-best-practices)
3. [Common Pitfalls and How to Avoid Them](#common-pitfalls-and-how-to-avoid-them)
4. [Performance Optimization](#performance-optimization)
5. [Validation and Quality Assurance](#validation-and-quality-assurance)
6. [Domain-Specific Considerations](#domain-specific-considerations)

## Methodological Best Practices

### 1. Parameter Selection and Uncertainty Modeling

#### Identify Key Uncertainties
- Focus on parameters that significantly influence decision outcomes
- Use sensitivity analysis to identify the most influential parameters
- Consider both first-order and interaction effects

#### Choose Appropriate Probability Distributions
- Use distributions that reflect the nature of uncertainty in your domain
- Validate distributional assumptions with domain experts
- Consider using empirical distributions when theoretical distributions are inappropriate

#### Correlation Structure
- Model correlations between parameters when they exist
- Use copulas for complex correlation structures
- Validate correlation assumptions with historical data or expert judgment

### 2. Model Development and Validation

#### Outcome Modeling
- Ensure your outcome model accurately represents the decision problem
- Validate model assumptions with domain experts
- Test model behavior under extreme parameter values

#### Convergence Diagnostics
- Check for Monte Carlo convergence in all VOI calculations
- Use multiple convergence diagnostics (e.g., Gelman-Rubin, Geweke)
- Increase sample sizes until results stabilize

#### Cross-Validation
- Validate your models against historical data when available
- Use out-of-sample testing to assess predictive performance
- Consider using holdout samples for validation

### 3. Decision Modeling

#### Strategy Definition
- Clearly define decision alternatives
- Ensure strategies are mutually exclusive and collectively exhaustive
- Consider the practical feasibility of each strategy

#### Net Benefit Calculation
- Use appropriate willingness-to-pay thresholds
- Account for time value of money when relevant
- Consider risk preferences in net benefit calculations

#### Population Scaling
- Apply appropriate population scaling when calculating population-level VOI
- Use realistic discount rates and time horizons
- Consider heterogeneity in population characteristics

## Implementation Best Practices

### 1. Data Preparation and Management

#### ParameterSet Creation
```python
# Good practice: Use descriptive parameter names
parameters = {
    "treatment_effect": np.random.normal(0.1, 0.05, 1000),
    "control_event_rate": np.random.beta(20, 80, 1000),  # Beta distribution for rates
    "cost_per_patient": np.random.gamma(2, 2500, 1000)   # Gamma distribution for costs
}

# Create ParameterSet
psa_samples = ParameterSet.from_numpy_or_dict(parameters)
```

#### ValueArray Creation
```python
# Good practice: Use clear strategy names
strategy_names = ["Standard Care", "New Treatment", "Combination Therapy"]
net_benefits = calculate_net_benefits(parameters, strategy_names)

# Create ValueArray
value_array = ValueArray.from_numpy(net_benefits, strategy_names)
```

### 2. VOI Calculation

#### Sample Size Selection
- Use sufficient sample sizes for stable VOI estimates
- Start with smaller samples for exploratory analysis
- Increase sample sizes for final results

#### Convergence Checking
```python
# Good practice: Check for convergence
def check_evpi_convergence(value_array, sample_sizes=[100, 200, 500, 1000]):
    results = []
    for n in sample_sizes:
        subset_array = ValueArray.from_numpy(
            value_array.values[:n], 
            value_array.strategy_names
        )
        analysis = DecisionAnalysis(nb_array=subset_array, parameter_samples=None)
        evpi_result = analysis.evpi()
        results.append(evpi_result)
    return results

# Check convergence before final analysis
convergence_results = check_evpi_convergence(value_array)
```

### 3. Error Handling and Validation

#### Input Validation
```python
# Good practice: Validate inputs
def validate_parameters(parameters):
    """Validate parameter inputs for common issues."""
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be a dictionary")
    
    for name, values in parameters.items():
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Parameter {name} must be a numpy array")
        
        if len(values) < 100:
            warnings.warn(f"Parameter {name} has fewer than 100 samples, results may be unstable")
        
        if np.any(np.isnan(values)):
            raise ValueError(f"Parameter {name} contains NaN values")
    
    return True
```

## Common Pitfalls and How to Avoid Them

### 1. Insufficient Sample Sizes

**Problem**: Using too few samples leads to unstable VOI estimates.

**Solution**: 
- Start with at least 1,000 samples for basic analysis
- Use 10,000+ samples for final results
- Check for convergence before reporting results

### 2. Inappropriate Parameter Distributions

**Problem**: Using unrealistic or inappropriate probability distributions.

**Solution**:
- Validate distributions with domain experts
- Use empirical data to inform distribution selection
- Consider using non-parametric methods when appropriate

### 3. Ignoring Correlations

**Problem**: Failing to model correlations between parameters.

**Solution**:
- Identify potential correlations during model development
- Use appropriate methods to model correlations
- Validate correlation assumptions with data or expert judgment

### 4. Overlooking Model Uncertainty

**Problem**: Treating the model as certain when it may be misspecified.

**Solution**:
- Consider structural uncertainty using model averaging
- Perform sensitivity analysis to model assumptions
- Use cross-validation to assess model performance

### 5. Misinterpreting Results

**Problem**: Misinterpreting VOI as the value of collecting any data.

**Solution**:
- Remember that VOI represents the value of perfect or partial perfect information
- Consider the cost and feasibility of data collection
- Interpret results in the context of decision-making

## Performance Optimization

### 1. Efficient Sampling

#### Use Vectorized Operations
```python
# Efficient: Vectorized operations
net_benefits = np.zeros((n_samples, n_strategies))
net_benefits[:, 1] = treatment_effect * effectiveness_slope - cost_per_patient

# Inefficient: Loop-based operations
net_benefits = np.zeros((n_samples, n_strategies))
for i in range(n_samples):
    net_benefits[i, 1] = treatment_effect[i] * effectiveness_slope[i] - cost_per_patient[i]
```

#### Memory Management
```python
# Good practice: Use appropriate data types
# Use float32 instead of float64 when precision allows
parameters = {
    "param1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    "param2": np.array([0.1, 0.2, 0.3], dtype=np.float32)
}
```

### 2. Parallel Processing

#### JAX Backend for Acceleration
```python
# When available, use JAX backend for acceleration
from voiage.backends import jax_backend

# Enable JAX backend
jax_backend.enable()

# Your VOI calculations will now use JAX for acceleration
analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=psa_samples)
evpi_result = analysis.evpi()  # Automatically uses JAX backend
```

### 3. Caching and Memoization

#### Cache Expensive Calculations
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_model_calculation(param_tuple):
    """Cache expensive model calculations."""
    # Convert tuple back to parameters
    # Perform expensive calculation
    return result

# Use in VOI calculations
def calculate_net_benefits_cached(parameters):
    # Convert parameters to hashable tuple for caching
    param_tuple = tuple(parameters["treatment_effect"])
    return expensive_model_calculation(param_tuple)
```

## Validation and Quality Assurance

### 1. Unit Testing

#### Test Individual Components
```python
import pytest
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet

def test_evpi_calculation():
    """Test EVPI calculation with known values."""
    # Create simple test case with known solution
    net_benefits = np.array([
        [0, 100],  # Sample 1: Strategy B better
        [0, 50],   # Sample 2: Strategy B better
        [100, 0],  # Sample 3: Strategy A better
        [50, 0]    # Sample 4: Strategy A better
    ])
    
    value_array = ValueArray.from_numpy(net_benefits, ["A", "B"])
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=None)
    
    evpi_result = analysis.evpi()
    expected_evpi = 25.0  # Known result for this simple case
    
    assert abs(evpi_result - expected_evpi) < 1e-10
```

### 2. Integration Testing

#### Test Complete Workflows
```python
def test_complete_voi_workflow():
    """Test a complete VOI analysis workflow."""
    # Generate test data
    parameters = generate_test_parameters(n_samples=1000)
    
    # Calculate net benefits
    net_benefits = calculate_net_benefits(parameters)
    
    # Create schema objects
    value_array = ValueArray.from_numpy(net_benefits, ["Strategy A", "Strategy B"])
    psa_samples = ParameterSet.from_numpy_or_dict(parameters)
    
    # Perform VOI analysis
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=psa_samples)
    
    # Calculate VOI metrics
    evpi_result = analysis.evpi()
    evppi_result = analysis.evppi()
    
    # Validate results
    assert evpi_result >= 0
    assert evppi_result >= 0
    assert evpi_result >= evppi_result  # EVPI should be >= EVPPI
```

### 3. Reproducibility

#### Set Random Seeds
```python
# Good practice: Set random seeds for reproducibility
def voi_analysis_with_reproducibility():
    """Perform VOI analysis with reproducible results."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    parameters = generate_parameters(n_samples=10000)
    
    # Perform analysis
    # Results will be identical across runs
```

## Domain-Specific Considerations

### 1. Healthcare Economics

#### Time Horizon and Discounting
- Use appropriate time horizons for health economic models
- Apply standard discount rates (typically 3% for costs and effects)
- Consider lifetime horizons for chronic conditions

#### WTP Thresholds
- Use country-specific willingness-to-pay thresholds
- Consider value of statistical life adjustments
- Account for equity considerations in threshold selection

### 2. Business Strategy

#### Market Dynamics
- Model competitive responses to strategic decisions
- Consider network effects and market feedback loops
- Account for regulatory and policy changes

#### Risk Preferences
- Incorporate risk aversion in business decisions
- Consider stakeholder risk preferences
- Model downside risk and tail outcomes

### 3. Environmental Policy

#### Long-Term Impacts
- Model long-term environmental and health impacts
- Consider irreversible environmental changes
- Account for intergenerational equity

#### Uncertainty Characterization
- Use heavy-tailed distributions for extreme events
- Model deep uncertainty using scenario approaches
- Consider precautionary principles in decision-making

### 4. Engineering Design

#### Safety Margins
- Incorporate safety factors in design decisions
- Model failure modes and their consequences
- Consider regulatory safety requirements

#### Physical Constraints
- Ensure parameter combinations are physically realistic
- Model constraint interactions and trade-offs
- Validate designs against physical laws

## Conclusion

Following these best practices will help ensure that your VOI analyses are methodologically sound, computationally efficient, and practically useful for decision-making. Remember that VOI analysis is an iterative process that benefits from continuous refinement and validation. Regular consultation with domain experts and stakeholders is essential for producing meaningful results that inform real-world decisions.

The voiage library is designed to support these best practices through its flexible architecture and comprehensive set of tools. By combining rigorous methodology with efficient implementation, you can produce high-quality VOI analyses that provide valuable insights for decision-making across a wide range of domains.