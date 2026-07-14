# Supplementary Methods and Formulae for voiage

This document provides comprehensive mathematical formulae and methodological details for all Value of Information (VOI) methods implemented in the `voiage` library. This supplementary material accompanies the main paper "voiage: A Python Library for Value of Information Analysis" and provides the mathematical foundations for the implemented methods.

## 1. Core VOI Methods

### 1.1 Expected Value of Perfect Information (EVPI)

**Definition**: EVPI quantifies the maximum amount a decision-maker should be willing to pay to eliminate all uncertainty in a decision model.

**Mathematical Formula**:
$$\text{EVPI} = \mathbb{E}_\theta\left[\max_d \text{NB}(d, \theta)\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

**Where**:
- $\text{NB}(d, \theta)$ is the net benefit of strategy $d$ given parameters $\theta$
- $\mathbb{E}_\theta$ denotes expectation over the parameter uncertainty distribution
- $d$ indexes decision strategies

**Implementation Details**:
The EVPI calculation in `voiage` uses Monte Carlo integration based on probabilistic sensitivity analysis (PSA) samples:

$$\widehat{\text{EVPI}} = \frac{1}{N}\sum_{i=1}^N \max_d \text{NB}(d, \theta^{(i)}) - \max_d \frac{1}{N}\sum_{i=1}^N \text{NB}(d, \theta^{(i)})$$

**Statistical Properties**:
- The estimator is unbiased: $\mathbb{E}[\widehat{\text{EVPI}}] = \text{EVPI}$
- Variance decreases at rate $O(N^{-1})$ where $N$ is the number of PSA samples

### 1.2 Expected Value of Partial Perfect Information (EVPPI)

**Definition**: EVPPI quantifies the value of eliminating uncertainty about a specific subset $\phi$ of parameters.

**Mathematical Formula**:
$$\text{EVPPI} = \mathbb{E}_{\phi}\left[\max_d \mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi \right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

**Implementation Details**:
The `voiage` library implements EVPPI using regression-based methods following Strong & Oakley (2014):

1. **Non-parametric Regression Approach**:
   $$\mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi^{(i)} \right] \approx \sum_{j=1}^N w_{ij}^{(d)} \text{NB}(d, \theta^{(j)})$$

2. **Parametric Regression Approach** (default):
   $$\mathbb{E}_{\theta | \phi}\left[\text{NB}(d, \theta) | \phi \right] = \beta_0^{(d)} + \sum_{k=1}^K \beta_k^{(d)} \phi_k + \epsilon^{(d)}$$

**Statistical Properties**:
- Consistency under appropriate regularity conditions on the regression model
- Asymptotic normality with variance that decreases with sample size

### 1.3 Expected Value of Sample Information (EVSI)

**Definition**: EVSI quantifies the expected value of collecting additional data to reduce parameter uncertainty.

**Mathematical Formula**:
$$\text{EVSI} = \mathbb{E}_D\left[\max_d \mathbb{E}_{\theta|D}\left[\text{NB}(d, \theta)|D\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

**Where**:
- $D$ represents potential data that could be collected
- $\mathbb{E}_{\theta|D}$ denotes posterior expectation given data $D$

**Implementation Details**:
The `voiage` library implements EVSI using flexible frameworks for various data-generating processes:

1. **General Framework**:
   $$\text{EVSI} = \mathbb{E}_{D \sim f(D|\theta_0)}\left[\max_d \mathbb{E}_{\theta|D}\left[\text{NB}(d, \theta)|D\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

2. **Monte Carlo Implementation**:
   $$\widehat{\text{EVSI}} = \frac{1}{N}\sum_{i=1}^N \max_d \frac{1}{M}\sum_{j=1}^M \text{NB}(d, \theta^{(i,j)}) - \max_d \frac{1}{N}\sum_{i=1}^N \text{NB}(d, \theta^{(i)})$$

## 2. Advanced VOI Methods

### 2.1 Structural Uncertainty VOI

**Definition**: Quantifies the value of learning about model structure uncertainty rather than parameter uncertainty.

**Mathematical Framework**:
Let $M_m$ denote different model structures, with prior probabilities $\pi(M_m)$. The structural uncertainty VOI is:

$$\text{VOI}_{\text{structure}} = \mathbb{E}_{M}\left[\max_d \mathbb{E}_{\theta|M}\left[\text{NB}(d, \theta, M)|M\right]\right] - \max_d \mathbb{E}_{M,\theta}\left[\text{NB}(d, \theta, M)\right]$$

**Implementation Approach**:
The `voiage` library implements structural uncertainty VOI using mixture models and Bayesian model averaging techniques.

### 2.2 Network Meta-Analysis VOI

**Definition**: VOI methods specific to evidence synthesis from multiple studies comparing interventions.

**Mathematical Framework**:
For a network of $K$ interventions with $S$ studies, the network meta-analysis VOI accounts for indirect evidence:

$$\text{VOI}_{\text{NMA}} = \mathbb{E}_{\Delta}\left[\max_d \mathbb{E}_{\theta|\Delta}\left[\text{NB}(d, \theta)|\Delta\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

**Where**:
- $\Delta$ represents the contrast parameters in the network meta-analysis
- $\theta$ represents the absolute treatment effects derived from $\Delta$

### 2.3 Adaptive Design VOI

**Definition**: Evaluates the value of adaptive trial designs with pre-planned modifications.

**Mathematical Framework**:
For a trial with interim analyses at times $t_1, t_2, ..., t_K$:

$$\text{VOI}_{\text{adaptive}} = \mathbb{E}_{D_{1:K}}\left[\max_d \mathbb{E}_{\theta|D_{1:K}}\left[\text{NB}(d, \theta)|D_{1:K}\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

**Implementation Details**:
The `voiage` library accounts for adaptation rules and potential modifications to trial design.

### 2.4 Portfolio Optimization VOI

**Definition**: Prioritizes multiple research opportunities simultaneously to optimize research investments.

**Mathematical Framework**:
For a portfolio of $J$ research opportunities with budget constraint $B$:

$$\text{VOI}_{\text{portfolio}} = \max_{x \in \mathcal{X}} \mathbb{E}_{D(x)}\left[\max_d \mathbb{E}_{\theta|D(x)}\left[\text{NB}(d, \theta)|D(x)\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

**Subject to**:
$$\sum_{j=1}^J c_j x_j \leq B$$

**Where**:
- $x_j \in \{0,1\}$ indicates whether to invest in research opportunity $j$
- $c_j$ is the cost of research opportunity $j$
- $\mathcal{X}$ is the feasible set of portfolios

### 2.5 Value of Heterogeneity

**Definition**: Quantifies the value of learning about subgroup effects and treatment heterogeneity.

**Mathematical Framework**:
For subgroups $s = 1, ..., S$ with treatment effects $\delta_s$:

$$\text{VOI}_{\text{heterogeneity}} = \mathbb{E}_{\delta}\left[\max_d \mathbb{E}_{\theta|\delta}\left[\text{NB}(d, \theta)|\delta\right]\right] - \max_d \mathbb{E}_\theta\left[\text{NB}(d, \theta)\right]$$

## 3. Population-Level Adjustments

### 3.1 Population Scaling

**Formula**:
$$\text{VOI}_{\text{population}} = \text{VOI} \times \text{Population} \times \text{Annuity Factor}$$

**Where**:
$$\text{Annuity Factor} = \begin{cases} 
\text{Time Horizon} & \text{if } \text{Discount Rate} = 0 \\
\frac{1-(1+\text{Discount Rate})^{-\text{Time Horizon}}}{\text{Discount Rate}} & \text{if } \text{Discount Rate} > 0
\end{cases}$$

### 3.2 QALY Calculations with Discounting

**Formula**:
$$\text{QALY} = \sum_{t=0}^T \frac{u_t \times \Delta t}{(1+r)^{t+\Delta t/2}}$$

**Where**:
- $u_t$ is the utility at time $t$
- $\Delta t$ is the time interval
- $r$ is the annual discount rate

## 4. Statistical Properties and Validation

### 4.1 Estimator Properties

**EVPI Estimator**:
- Unbiased: $\mathbb{E}[\widehat{\text{EVPI}}] = \text{EVPI}$
- Variance: $\text{Var}(\widehat{\text{EVPI}}) = O(N^{-1})$
- Asymptotic Normality: $\sqrt{N}(\widehat{\text{EVPI}} - \text{EVPI}) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$

**EVPPI Estimator**:
- Consistency under regularity conditions
- Convergence rate depends on regression method and smoothness of conditional expectations

### 4.2 Convergence Diagnostics

The `voiage` library implements convergence diagnostics to ensure reliable estimates:

1. **Effective Sample Size** for MCMC-based methods
2. **Monte Carlo Error** estimation
3. **Bootstrap Confidence Intervals** for uncertainty quantification

## 5. Computational Methods

### 5.1 Numerical Integration

For high-dimensional integrals, `voiage` employs:

1. **Quasi-Monte Carlo** methods for improved convergence
2. **Sparse Grid Quadrature** for smooth integrands
3. **Adaptive Integration** for irregular domains

### 5.2 Regression-Based Approximation

For EVPPI calculations, the library implements:

1. **Gaussian Process Regression** for smooth conditional expectations
2. **Random Forest Regression** for complex, non-linear relationships
3. **Neural Network Approximation** for high-dimensional problems

### 5.3 Dimension Reduction

For high-dimensional parameter spaces:

1. **Principal Component Analysis** for linear dimension reduction
2. **Active Subspaces** for nonlinear dimension reduction
3. **Sufficient Dimension Reduction** techniques

This supplementary document provides the mathematical foundations for all VOI methods implemented in the `voiage` library. For implementation details and usage examples, please refer to the main paper and the library documentation.