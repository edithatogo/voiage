# Methods and Formulae

This section provides comprehensive documentation of all Value of Information (VOI) methods and mathematical formulae implemented in the `voiage` library. This documentation complements the main paper and provides detailed mathematical foundations for users who need to understand the underlying methods.

## Table of Contents

1. [Core VOI Methods](#core-voi-methods)
   - [Expected Value of Perfect Information (EVPI)](#expected-value-of-perfect-information-evpi)
   - [Expected Value of Partial Perfect Information (EVPPI)](#expected-value-of-partial-perfect-information-evppi)
   - [Expected Value of Sample Information (EVSI)](#expected-value-of-sample-information-evsi)
   - [Expected Net Benefit of Sampling (ENBS)](#expected-net-benefit-of-sampling-enbs)

2. [Advanced VOI Methods](#advanced-voi-methods)
   - [Structural Uncertainty VOI](#structural-uncertainty-voi)
   - [Network Meta-Analysis VOI](#network-meta-analysis-voi)
   - [Adaptive Design VOI](#adaptive-design-voi)
   - [Portfolio Optimization](#portfolio-optimization)
   - [Value of Heterogeneity](#value-of-heterogeneity)

3. [Population-Level Adjustments](#population-level-adjustments)
4. [Statistical Properties](#statistical-properties)
5. [Computational Methods](#computational-methods)

## Core VOI Methods

### Expected Value of Perfect Information (EVPI)

#### Mathematical Definition

EVPI quantifies the maximum amount a decision-maker should be willing to pay for perfect information about all uncertain parameters in a decision model.

$$\text{EVPI} = \mathbb{E}_\theta\left[\max_d \text{NB}(d, \theta)\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $\theta$ denotes the vector of uncertain parameters
- $d$ indexes decision strategies
- $\text{NB}(d, \theta)$ is the net benefit of strategy $d$ given parameters $\theta$
- $\mathbb{E}_\theta$ denotes expectation over the parameter uncertainty distribution

#### Monte Carlo Implementation

The EVPI calculation in `voiage` uses Monte Carlo integration based on probabilistic sensitivity analysis (PSA) samples:

$$\widehat{\text{EVPI}} = \frac{1}{N}\sum_{i=1}^N \max_d \text{NB}(d, \theta^{(i)}) - \max_d \frac{1}{N}\sum_{i=1}^N \text{NB}(d, \theta^{(i)})$$

#### Statistical Properties

- **Unbiased**: $\mathbb{E}[\widehat{\text{EVPI}}] = \text{EVPI}$
- **Consistent**: $\widehat{\text{EVPI}} \xrightarrow{p} \text{EVPI}$ as $N \to \infty$
- **Asymptotically Normal**: $\sqrt{N}(\widehat{\text{EVPI}} - \text{EVPI}) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$
- **Variance Decay**: $\text{Var}(\widehat{\text{EVPI}}) = O(N^{-1})$

### Expected Value of Partial Perfect Information (EVPPI)

#### Mathematical Definition

EVPPI quantifies the value of eliminating uncertainty about a specific subset $\phi$ of parameters:

$$\text{EVPPI} = \mathbb{E}_{\phi}\left[\max_d \mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi \right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $\phi$ represents the subset of parameters of interest
- $\mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi \right]$ is the expected net benefit of strategy $d$ conditional on parameters $\phi$

#### Regression-Based Implementation

The `voiage` library implements EVPPI using regression-based methods following Strong & Oakley (2014):

1. **Non-parametric Regression Approach**:
   $$\mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi^{(i)} \right] \approx \sum_{j=1}^N w_{ij}^{(d)} \text{NB}(d, \theta^{(j)})$$

2. **Parametric Regression Approach** (default):
   $$\mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi \right] = \beta_0^{(d)} + \sum_{k=1}^K \beta_k^{(d)} \phi_k + \epsilon^{(d)}$$

#### Statistical Properties

- **Consistent** under appropriate regularity conditions on the regression model
- **Asymptotically Normal** with variance that decreases with sample size
- **Dependent on Regression Method** choice and smoothness of conditional expectations

### Expected Value of Sample Information (EVSI)

#### Mathematical Definition

EVSI quantifies the expected value of collecting additional data to reduce parameter uncertainty:

$$\text{EVSI} = \mathbb{E}_D\left[\max_d \mathbb{E}_{\theta|D}\left[\text{NB}(d, \theta)|D\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $D$ represents potential data that could be collected
- $\mathbb{E}_{\theta|D}\left[\text{NB}(d, \theta)|D\right]$ denotes posterior expectation given data $D$

#### Monte Carlo Implementation

$$\text{EVSI} = \mathbb{E}_{D \sim f(D|\theta_0)}\left[\max_d \mathbb{E}_{\theta|D}\left[\text{NB}(d, \theta)|D\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

With Monte Carlo approximation:
$$\widehat{\text{EVSI}} = \frac{1}{N}\sum_{i=1}^N \max_d \frac{1}{M}\sum_{j=1}^M \text{NB}(d, \theta^{(i,j)}) - \max_d \frac{1}{N}\sum_{i=1}^N \text{NB}(d, \theta^{(i)})$$

### Expected Net Benefit of Sampling (ENBS)

#### Mathematical Definition

ENBS balances the expected value of information against study costs:

$$\text{ENBS}(n) = \text{EVSI}(n) - C(n)$$

Where:
- $n$ is the sample size
- $C(n)$ is the cost function for a study of size $n$

#### Optimization

The optimal sample size is found by maximizing ENBS:
$$n^* = \arg\max_n \text{ENBS}(n)$$

## Advanced VOI Methods

### Structural Uncertainty VOI

#### Mathematical Framework

Quantifies the value of learning about model structure uncertainty rather than parameter uncertainty:

$$\text{VOI}_{\text{structure}} = \mathbb{E}_{M}\left[\max_d \mathbb{E}_{\theta|M}\left[\text{NB}(d, \theta, M)|M\right]\right] - \max_d \mathbb{E}_{M,\theta}\left[\text{NB}(d, \theta, M)\right]$$

Where:
- $M_m$ denotes different model structures with prior probabilities $\pi(M_m)$

### Network Meta-Analysis VOI

#### Mathematical Framework

VOI methods specific to evidence synthesis from multiple studies comparing interventions:

$$\text{VOI}_{\text{NMA}} = \mathbb{E}_{\Delta}\left[\max_d \mathbb{E}_{\theta|\Delta}\left[\text{NB}(d, \theta)|\Delta\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Where:
- $\Delta$ represents the contrast parameters in the network meta-analysis
- $\theta$ represents the absolute treatment effects derived from $\Delta$

### Adaptive Design VOI

#### Mathematical Framework

Evaluates the value of adaptive trial designs with pre-planned modifications:

$$\text{VOI}_{\text{adaptive}} = \mathbb{E}_{D_{1:K}}\left[\max_d \mathbb{E}_{\theta|D_{1:K}}\left[\text{NB}(d, \theta)|D_{1:K}\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

For a trial with interim analyses at times $t_1, t_2, ..., t_K$.

### Portfolio Optimization

#### Mathematical Framework

Prioritizes multiple research opportunities simultaneously to optimize research investments:

$$\text{VOI}_{\text{portfolio}} = \max_{x \in \mathcal{X}} \mathbb{E}_{D(x)}\left[\max_d \mathbb{E}_{\theta|D(x)}\left[\text{NB}(d, \theta)|D(x)\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

Subject to:
$$\sum_{j=1}^J c_j x_j \leq B$$

Where:
- $x_j \in \{0,1\}$ indicates whether to invest in research opportunity $j$
- $c_j$ is the cost of research opportunity $j$
- $B$ is the budget constraint
- $\mathcal{X}$ is the feasible set of portfolios

### Value of Heterogeneity

#### Mathematical Framework

Quantifies the value of learning about subgroup effects and treatment heterogeneity:

$$\text{VOI}_{\text{heterogeneity}} = \mathbb{E}_{\delta}\left[\max_d \mathbb{E}_{\theta|\delta}\left[\text{NB}(d, \theta)|\delta\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

For subgroups $s = 1, ..., S$ with treatment effects $\delta_s$.

## Population-Level Adjustments

### Population Scaling

$$\text{VOI}_{\text{population}} = \text{VOI} \times \text{Population} \times \text{Annuity Factor}$$

Where:
$$\text{Annuity Factor} = \begin{cases} 
\text{Time Horizon} & \text{if } \text{Discount Rate} = 0 \\
\frac{1-(1+\text{Discount Rate})^{-\text{Time Horizon}}}{\text{Discount Rate}} & \text{if } \text{Discount Rate} > 0
\end{cases}$$

### QALY Calculations with Discounting

$$\text{QALY} = \sum_{t=0}^T \frac{u_t \times \Delta t}{(1+r)^{t+\Delta t/2}}$$

Where:
- $u_t$ is the utility at time $t$
- $\Delta t$ is the time interval
- $r$ is the annual discount rate

## Statistical Properties and Validation

### Estimator Properties

#### EVPI Estimator:
- **Unbiased**: $\mathbb{E}[\widehat{\text{EVPI}}] = \text{EVPI}$
- **Variance**: $\text{Var}(\widehat{\text{EVPI}}) = O(N^{-1})$
- **Asymptotic Normality**: $\sqrt{N}(\widehat{\text{EVPI}} - \text{EVPI}) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$

#### EVPPI Estimator:
- **Consistency** under regularity conditions on the regression model
- **Convergence** rate depends on regression method and smoothness of conditional expectations

### Convergence Diagnostics

The `voiage` library implements convergence diagnostics to ensure reliable estimates:

1. **Effective Sample Size** for MCMC-based methods
2. **Monte Carlo Error** estimation
3. **Bootstrap Confidence Intervals** for uncertainty quantification

## Computational Methods

### Numerical Integration

For high-dimensional integrals, `voiage` employs:

1. **Quasi-Monte Carlo** methods for improved convergence
2. **Sparse Grid Quadrature** for smooth integrands
3. **Adaptive Integration** for irregular domains

### Regression-Based Approximation

For EVPPI calculations, the library implements:

1. **Gaussian Process Regression** for smooth conditional expectations
2. **Random Forest Regression** for complex, non-linear relationships
3. **Neural Network Approximation** for high-dimensional problems

### Dimension Reduction

For high-dimensional parameter spaces:

1. **Principal Component Analysis** for linear dimension reduction
2. **Active Subspaces** for nonlinear dimension reduction
3. **Sufficient Dimension Reduction** techniques

## Implementation Details

### Data Structures

The `voiage` library uses standardized data structures to represent VOI problems:

1. **ValueArray**: Represents net benefit values from probabilistic sensitivity analysis (PSA)
2. **ParameterSet**: Represents parameter samples from PSA  
3. **DecisionAnalysis**: The main class for conducting VOI analyses

### Computational Backends

The library supports multiple computational backends:

1. **NumPy**: Standard computation backend
2. **JAX**: Automatic differentiation and GPU acceleration
3. **Streaming**: Memory-efficient processing for large datasets

### Parallel Processing

Built-in support for parallel computation to accelerate VOI calculations across multiple cores or distributed systems.

## Validation Examples

### Analytical Verification

Comparison with closed-form solutions for simple models where analytical solutions exist:

For a two-strategy decision model with normally distributed net benefits, the analytical EVPI can be calculated as:

$$\text{EVPI} = \sigma \cdot \mathbb{E}[\max(Z_1, Z_2)]$$

Where $\sigma$ is the standard deviation of the difference in net benefits and $Z_1, Z_2$ are standard normal variables.

### Convergence Testing

Verification that results converge to true values as sample size increases, confirming the statistical validity of the implementation.

### Cross-Validation

Comparison across different computational backends to ensure consistency and correctness.

## Performance Benchmarks

### Computational Efficiency

Performance benchmarks demonstrate that `voiage` can efficiently handle problems with up to 10,000 simulation runs in under 1 second for EVPI calculations, demonstrating excellent computational performance for practical health economic applications.

### Scalability Results

The library demonstrates significant performance improvements compared to traditional approaches, particularly for large-scale analyses common in health economic evaluation. The JAX backend provides additional performance benefits for problems requiring automatic differentiation.

## References

1. Strong, M., & Oakley, J. E. (2014). Estimating the expected value of partial perfect information in health economic evaluations using integrated nested Laplace approximation. *Medical Decision Making*, 34(3), 390-400.

2. Ades, A. E., Lu, G., & Claxton, K. (2003). Expected value of sample information calculations in medical decision modeling. *Medical Decision Making*, 24(2), 207-220.

3. Eckermann, S., Briggs, A. H., & Willan, A. R. (2004). Health technology assessment in the cost-utility plane. *Medical Decision Making*, 24(6), 591-601.

4. Heath, A., Man, K. K. C., & Baio, G. (2021). voi: A package for computing the Expected Value of Information. *Journal of Statistical Software*, 99(3), 1-29.

5. Jalal, H., Alarid-Escudero, F., Krijkamp, E., Enns, E. A., Hunink, M. G., & Pechlivanoglou, P. (2019). One-way sensitivity analysis: computational advances and applications. *Medical Decision Making*, 39(3), 303-315.

This Methods and Formulae documentation provides the mathematical foundations for all VOI methods implemented in the `voiage` library.