# Basic VOI Methods

This guide covers the basic Value of Information (VOI) methods implemented in voiage: EVPI (Expected Value of Perfect Information) and EVPPI (Expected Value of Partial Perfect Information).

## Expected Value of Perfect Information (EVPI)

EVPI quantifies the maximum amount a decision-maker should be willing to pay for perfect information about all uncertain parameters in a decision model.

### Mathematical Definition

EVPI = E[max(NB)] - max(E[NB])

Where:
- E is the expectation over the PSA samples
- NB represents net benefits for each strategy
- max(NB) is the maximum net benefit for each sample
- E[max(NB)] is the expected value of the maximum net benefits
- max(E[NB]) is the maximum of the expected net benefits across strategies

### Usage Example

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

# Create sample net benefit data (n_samples x n_strategies)
# For example, 1000 PSA samples and 2 strategies
np.random.seed(42)
nb_data = np.random.normal([100, 110], [10, 15], (1000, 2))

# Create ValueArray
value_array = ValueArray.from_numpy(nb_data, ["Standard Care", "New Treatment"])

# Create DecisionAnalysis
analysis = DecisionAnalysis(nb_array=value_array)

# Calculate EVPI
evpi_result = analysis.evpi()
print(f"EVPI: {evpi_result:.2f}")

# Calculate population-adjusted EVPI
evpi_pop = analysis.evpi(population=100000, time_horizon=10, discount_rate=0.03)
print(f"Population EVPI: {evpi_pop:.2f}")
```

## Expected Value of Partial Perfect Information (EVPPI)

EVPPI quantifies the value of learning the true value of a specific subset of model parameters.

### Mathematical Definition

EVPPI = E_p[max_d E[NB_d|p]] - max_d E[NB_d]

Where:
- E_p is the expectation over the parameter(s) of interest
- E[NB_d|p] is the expected net benefit of strategy d conditional on the parameter(s) p
- This is typically estimated via regression

### Usage Example

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet

# Create sample net benefit data
np.random.seed(42)
nb_data = np.random.normal([100, 110], [10, 15], (1000, 2))

# Create ValueArray
value_array = ValueArray.from_numpy(nb_data, ["Standard Care", "New Treatment"])

# Create parameter samples
parameters = {
    "effectiveness": np.random.beta(2, 1, 1000),
    "cost": np.random.normal(50, 5, 1000)
}
parameter_set = ParameterSet.from_numpy_or_dict(parameters)

# Create DecisionAnalysis with parameters
analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameter_set)

# Calculate EVPPI
evppi_result = analysis.evppi()
print(f"EVPPI: {evppi_result:.2f}")

# Calculate population-adjusted EVPPI
evppi_pop = analysis.evppi(population=100000, time_horizon=10, discount_rate=0.03)
print(f"Population EVPPI: {evppi_pop:.2f}")
```

## Advanced Options

### Regression Model Selection

For EVPPI calculations, you can specify a custom regression model:

```python
from sklearn.ensemble import RandomForestRegressor

# Use a Random Forest regressor instead of the default LinearRegression
evppi_result = analysis.evppi(regression_model=RandomForestRegressor)
```

### Sample Size Control

For large datasets, you can control the number of samples used for regression fitting:

```python
# Use only 500 samples for regression fitting to speed up computation
evppi_result = analysis.evppi(n_regression_samples=500)
```

## Best Practices

1. **Data Quality**: Ensure your PSA samples are representative and sufficiently numerous
2. **Parameter Selection**: Carefully select which parameters to include in EVPPI analysis
3. **Model Choice**: Consider using non-linear regression models for complex relationships
4. **Validation**: Always validate results with domain experts