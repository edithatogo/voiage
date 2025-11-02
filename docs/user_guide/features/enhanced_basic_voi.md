# Basic VOI Methods

This guide covers the basic Value of Information (VOI) methods implemented in voiage: EVPI (Expected Value of Perfect Information) and EVPPI (Expected Value of Partial Perfect Information).

## Expected Value of Perfect Information (EVPI)

EVPI quantifies the maximum amount a decision-maker should be willing to pay for perfect information about all uncertain parameters in a decision model.

### Mathematical Definition

More formally, let $\theta$ denote the vector of uncertain parameters, $d$ denote decision strategies, and $\text{NB}(d, \theta)$ denote the net benefit of strategy $d$ given parameters $\theta$. Then:

$$\text{EVPI} = \mathbb{E}_\theta\left[\max_d \text{NB}(d, \theta)\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $\mathbb{E}_\theta$ denotes expectation over the parameter uncertainty distribution
- $\text{NB}(d, \theta)$ is the net benefit of strategy $d$ given parameters $\theta$
- $\max_d \text{NB}(d, \theta)$ is the maximum net benefit over strategies for a given parameter set
- $\mathbb{E}_\theta\left[\max_d \text{NB}(d, \theta)\right]$ is the expected value of the maximum net benefits
- $\max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$ is the maximum of the expected net benefits across strategies

### Statistical Properties

The EVPI estimator has the following statistical properties:
- **Unbiased**: $\mathbb{E}[\widehat{\text{EVPI}}] = \text{EVPI}$
- **Consistent**: $\widehat{\text{EVPI}} \xrightarrow{p} \text{EVPI}$ as $N \to \infty$
- **Asymptotically Normal**: $\sqrt{N}(\widehat{\text{EVPI}} - \text{EVPI}) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$
- **Variance Decay**: $\text{Var}(\widehat{\text{EVPI}}) = O(N^{-1})$ where $N$ is the number of PSA samples

### Monte Carlo Implementation

The EVPI calculation in `voiage` uses Monte Carlo integration based on probabilistic sensitivity analysis (PSA) samples:

$$\widehat{\text{EVPI}} = \frac{1}{N}\sum_{i=1}^N \max_d \text{NB}(d, \theta^{(i)}) - \max_d \frac{1}{N}\sum_{i=1}^N \text{NB}(d, \theta^{(i)})$$

### Usage Example

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

# Create sample net benefit data (n_samples x n_strategies)
# For example, 1000 PSA samples and 2 strategies
np.random.seed(42)
nb_data = np.random.normal(loc=[100, 110], scale=[10, 15], size=(1000, 2))

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

EVPPI quantifies the value of learning the true value of a specific subset $\phi$ of parameters.

### Mathematical Definition

More formally, let $\phi$ denote the subset of parameters of interest and $\theta_{-\phi}$ denote the remaining parameters. Then:

$$\text{EVPPI} = \mathbb{E}_{\phi}\left[\max_d \mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi \right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $\mathbb{E}_{\phi}$ denotes expectation over the parameters of interest
- $\mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi \right]$ is the expected net benefit of strategy $d$ conditional on parameters $\phi$

### Regression-Based Implementation

The `voiage` library implements EVPPI using regression-based methods following Strong & Oakley (2014):

1. **Non-parametric Regression Approach**:
   $$\mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi^{(i)} \right] \approx \sum_{j=1}^N w_{ij}^{(d)} \text{NB}(d, \theta^{(j)})$$

2. **Parametric Regression Approach** (default):
   $$\mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi \right] = \beta_0^{(d)} + \sum_{k=1}^K \beta_k^{(d)} \phi_k + \epsilon^{(d)}$$

### Statistical Properties

The EVPPI estimator has the following statistical properties:
- **Consistent** under appropriate regularity conditions on the regression model
- **Asymptotically Normal** with variance that decreases with sample size
- **Dependent on Regression Method** choice and smoothness of conditional expectations

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

# Use Random Forest for EVPPI calculation
evppi_rf = analysis.evppi(regression_model=RandomForestRegressor(n_estimators=100))
print(f"EVPPI with Random Forest: {evppi_rf:.2f}")
```

### Computational Efficiency

For large datasets, you can use subsampling for computational efficiency:

```python
# Use only 500 samples for regression fitting
evppi_subsample = analysis.evppi(n_regression_samples=500)
print(f"EVPPI with subsampling: {evppi_subsample:.2f}")
```

## Best Practices

1. **Data Quality**: Ensure your PSA samples are representative and sufficiently numerous
2. **Parameter Selection**: Carefully select which parameters to include in EVPPI analysis
3. **Model Choice**: Consider using non-linear regression models for complex relationships
4. **Validation**: Always validate results with domain experts
5. **Convergence**: Check for convergence with increasing sample sizes
6. **Uncertainty Quantification**: Use bootstrap methods to quantify uncertainty in VOI estimates