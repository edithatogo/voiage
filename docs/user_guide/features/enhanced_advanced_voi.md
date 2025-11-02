# Advanced VOI Methods

This guide covers the advanced Value of Information (VOI) methods implemented in voiage, including structural uncertainty VOI, network meta-analysis VOI, adaptive design VOI, portfolio optimization, and value of heterogeneity.

## Structural Uncertainty VOI

Structural uncertainty VOI quantifies the value of learning about model structure uncertainty rather than parameter uncertainty.

### Mathematical Framework

Let $M_m$ denote different model structures, with prior probabilities $\pi(M_m)$. The structural uncertainty VOI is:

$$\text{VOI}_{\text{structure}} = \mathbb{E}_{M}\left[\max_d \mathbb{E}_{\theta|M}\left[\text{NB}(d, \theta, M)|M\right]\right] - \max_d \mathbb{E}_{M,\theta}\left[\text{NB}(d, \theta, M)\right]$$

Where:
- $\mathbb{E}_{M}$ denotes expectation over model structures
- $\mathbb{E}_{\theta|M}\left[\text{NB}(d, \theta, M)|M\right]$ is the expected net benefit of strategy $d$ conditional on model structure $M$
- $\text{NB}(d, \theta, M)$ is the net benefit of strategy $d$ given parameters $\theta$ and model structure $M$

### Implementation Approach

The `voiage` library implements structural uncertainty VOI using mixture models and Bayesian model averaging techniques. The implementation accounts for:
1. Model prior probabilities
2. Conditional parameter distributions for each model structure
3. Strategy-specific net benefit calculations

### Usage Example

```python
import numpy as np
from voiage.methods.structural import structural_evpi, structural_evppi

# Simulated model structures with different parameter distributions
model_structures = {
    "Model_A": {
        "parameters": np.random.normal(0.8, 0.1, 1000),  # Effectiveness
        "weights": np.ones(1000) * 0.6  # Prior probability 0.6
    },
    "Model_B": {
        "parameters": np.random.normal(0.7, 0.15, 1000),  # Effectiveness
        "weights": np.ones(1000) * 0.4  # Prior probability 0.4
    }
}

# Calculate structural EVPI
struct_evpi = structural_evpi(model_structures)
print(f"Structural EVPI: {struct_evpi:.2f}")
```

## Network Meta-Analysis VOI

Network meta-analysis VOI methods specifically address evidence synthesis from multiple studies comparing interventions.

### Mathematical Framework

For a network of $K$ interventions with $S$ studies, the network meta-analysis VOI accounts for indirect evidence:

$$\text{VOI}_{\text{NMA}} = \mathbb{E}_{\Delta}\left[\max_d \mathbb{E}_{\theta|\Delta}\left[\text{NB}(d, \theta)|\Delta\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $\Delta$ represents the contrast parameters in the network meta-analysis
- $\theta$ represents the absolute treatment effects derived from $\Delta$
- $\mathbb{E}_{\theta|\Delta}\left[\text{NB}(d, \theta)|\Delta\right]$ is the expected net benefit conditional on the network contrasts

### Implementation Details

The implementation in `voiage` uses:
1. Consistency equations for network meta-analysis
2. Bayesian hierarchical modeling for parameter uncertainty
3. Efficient Monte Carlo sampling for VOI calculation

### Usage Example

```python
import numpy as np
from voiage.methods.network_nma import evsi_nma

# Simulated network structure data
network_data = {
    "treatments": ["A", "B", "C", "D"],
    "contrasts": np.random.normal(loc=[0.1, 0.2, 0.15], scale=[0.05, 0.07, 0.06], size=(1000, 3)),
    "study_variances": np.random.gamma(2, 0.1, (1000, 3))
}

# Calculate NMA EVSI
nma_evsi = evsi_nma(network_data, sample_size=500)
print(f"NMA EVSI: {nma_evsi:.2f}")
```

## Adaptive Design VOI

Adaptive design VOI evaluates the value of adaptive trial designs with pre-planned modifications.

### Mathematical Framework

For a trial with interim analyses at times $t_1, t_2, ..., t_K$:

$$\text{VOI}_{\text{adaptive}} = \mathbb{E}_{D_{1:K}}\left[\max_d \mathbb{E}_{\theta|D_{1:K}}\left[\text{NB}(d, \theta)|D_{1:K}\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $D_{1:K}$ represents the data collected at interim analyses
- $\mathbb{E}_{\theta|D_{1:K}}\left[\text{NB}(d, \theta)|D_{1:K}\right]$ is the expected net benefit conditional on interim data

### Implementation Details

The `voiage` library accounts for:
1. Adaptation rules and potential modifications
2. Conditional power calculations
3. Optimal decision rules at interim analyses

### Usage Example

```python
import numpy as np
from voiage.methods.adaptive import adaptive_evsi

# Define trial design parameters
design_parameters = {
    "sample_size": 1000,
    "interim_analysis_times": [0.5],  # Interim analysis at 50% recruitment
    "effectiveness_threshold": 0.1,
    "alpha_spending": [0.025]  # Alpha spending function
}

# Calculate adaptive EVSI
adaptive_evsi_result = adaptive_evsi(design_parameters)
print(f"Adaptive EVSI: {adaptive_evsi_result:.2f}")
```

## Portfolio Optimization

Portfolio optimization prioritizes multiple research opportunities simultaneously to optimize research investments.

### Mathematical Framework

For a portfolio of $J$ research opportunities with budget constraint $B$:

$$\text{VOI}_{\text{portfolio}} = \max_{x \in \mathcal{X}} \mathbb{E}_{D(x)}\left[\max_d \mathbb{E}_{\theta|D(x)}\left[\text{NB}(d, \theta)|D(x)\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Subject to:
$$\sum_{j=1}^J c_j x_j \leq B$$

Where:
- $x_j \in \{0,1\}$ indicates whether to invest in research opportunity $j$
- $c_j$ is the cost of research opportunity $j$
- $\mathcal{X}$ is the feasible set of portfolios
- $D(x)$ represents the data that would be collected from portfolio $x$

### Implementation Details

The portfolio optimization implementation uses:
1. Branch-and-bound algorithms for discrete optimization
2. Efficient bounds for pruning the search space
3. Parallel computation for large portfolios

### Usage Example

```python
import numpy as np
from voiage.methods.portfolio import portfolio_voi

# Define research opportunities
opportunities = [
    {"name": "Opportunity_A", "cost": 100000, "evsi": 50000},
    {"name": "Opportunity_B", "cost": 150000, "evsi": 75000},
    {"name": "Opportunity_C", "cost": 80000, "evsi": 40000}
]

# Define budget constraint
budget = 200000

# Calculate optimal portfolio
optimal_portfolio = portfolio_voi(opportunities, budget)
print(f"Optimal portfolio: {optimal_portfolio}")
```

## Value of Heterogeneity

Value of heterogeneity quantifies the value of learning about subgroup effects and treatment heterogeneity.

### Mathematical Framework

For subgroups $s = 1, ..., S$ with treatment effects $\delta_s$:

$$\text{VOI}_{\text{heterogeneity}} = \mathbb{E}_{\delta}\left[\max_d \mathbb{E}_{\theta|\delta}\left[\text{NB}(d, \theta)|\delta\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $\delta$ represents subgroup-specific treatment effects
- $\mathbb{E}_{\theta|\delta}\left[\text{NB}(d, \theta)|\delta\right]$ is the expected net benefit conditional on subgroup effects

### Implementation Details

The implementation accounts for:
1. Hierarchical modeling of subgroup effects
2. Borrowing strength across subgroups
3. Personalized treatment decisions

### Usage Example

```python
import numpy as np
from voiage.methods.heterogeneity import value_heterogeneity

# Define subgroup data
subgroup_data = {
    "subgroups": ["Young", "Middle-aged", "Old"],
    "effects": np.random.normal(loc=[0.15, 0.12, 0.08], scale=[0.05, 0.04, 0.06], size=(1000, 3)),
    "sizes": [300, 400, 300]  # Subgroup population sizes
}

# Calculate value of heterogeneity
heterogeneity_value = value_heterogeneity(subgroup_data)
print(f"Value of heterogeneity: {heterogeneity_value:.2f}")
```

## Sequential and Dynamic VOI

Sequential and dynamic VOI methods analyze information value in sequential decision-making contexts, where information is gathered in stages.

### Mathematical Framework

For decisions at times $t = 1, 2, ..., T$:

$$\text{VOI}_{\text{sequential}} = \mathbb{E}_{D_{1:T}}\left[\sum_{t=1}^T \gamma^t \max_{d_t} \mathbb{E}_{\theta|D_{1:t}}\left[\text{NB}(d_t, \theta)|D_{1:t}\right]\right] - \sum_{t=1}^T \gamma^t \max_{d_t} \mathbb{E}_\theta\left[\text{NB}(d_t, \theta)\right]$$

Where:
- $\gamma$ is the discount factor
- $D_{1:t}$ represents data collected up to time $t$
- $d_t$ represents decisions at time $t$

### Usage Example

```python
import numpy as np
from voiage.methods.sequential import sequential_voi

# Define sequential decision problem
sequential_data = {
    "horizon": 5,
    "discount_rate": 0.03,
    "information_arrival": [100, 200, 150, 180, 120]  # Information arrival at each stage
}

# Calculate sequential VOI
seq_voi = sequential_voi(sequential_data)
print(f"Sequential VOI: {seq_voi:.2f}")
```

## Observational Data VOI

Observational data VOI methods quantify the value of observational studies and real-world evidence in reducing decision uncertainty.

### Mathematical Framework

For observational data $D_{\text{obs}}$:

$$\text{VOI}_{\text{observational}} = \mathbb{E}_{D_{\text{obs}}}\left[\max_d \mathbb{E}_{\theta|D_{\text{obs}}}\left[\text{NB}(d, \theta)|D_{\text{obs}}\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where the likelihood function accounts for observational study design features:
$$p(D_{\text{obs}}|\theta) = \prod_{i=1}^n p(y_i|x_i, \theta) \times \text{selection_bias}(x_i|\theta)$$

### Usage Example

```python
import numpy as np
from voiage.methods.observational import voi_observational

# Define observational study data
observational_data = {
    "sample_size": 5000,
    "confounding_variables": np.random.normal(0, 1, (5000, 3)),
    "bias_adjustment": 0.1  # Estimated selection bias
}

# Calculate observational data VOI
obs_voi = voi_observational(observational_data)
print(f"Observational data VOI: {obs_voi:.2f}")
```

## Best Practices

1. **Model Specification**: Carefully specify model structures for structural uncertainty analysis
2. **Prior Elicitation**: Use expert elicitation for model priors in structural uncertainty
3. **Network Assumptions**: Validate consistency assumptions in network meta-analysis
4. **Adaptive Rules**: Clearly define adaptation rules for adaptive design VOI
5. **Budget Constraints**: Consider realistic budget constraints in portfolio optimization
6. **Subgroup Identification**: Use clinical expertise for meaningful subgroup definitions
7. **Sequential Planning**: Plan information arrival patterns realistically
8. **Observational Bias**: Account for selection bias in observational data VOI

## Validation and Testing

All methods include:
1. **Unit Testing**: Comprehensive unit tests for core algorithms
2. **Integration Testing**: Integration tests with analytical solutions
3. **Validation Examples**: Comparison with established methods
4. **Performance Benchmarks**: Computational performance testing
5. **Edge Case Testing**: Behavior under extreme parameter values

## Computational Considerations

1. **Scalability**: Methods optimized for large-scale problems
2. **Parallel Processing**: Built-in support for parallel computation
3. **Memory Efficiency**: Efficient memory usage for large datasets
4. **Numerical Stability**: Robust numerical methods to prevent overflow/underflow
5. **Convergence Monitoring**: Automatic convergence diagnostics

These advanced methods extend the capabilities of standard VOI analysis to address complex real-world decision problems in health economics and other domains.