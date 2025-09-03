# Portfolio Optimization

This guide covers portfolio optimization functionality in voiage, which helps prioritize research investments based on their expected value.

## Overview

Portfolio optimization in voiage allows you to determine the optimal allocation of research budget across multiple candidate studies to maximize the expected value of information.

## Key Concepts

### PortfolioStudy
Represents a single candidate study within a research portfolio with properties:
- `name`: Identifier for the study
- `cost`: Cost of conducting the study
- `value`: Expected value of the study (e.g., EVPPI)

### PortfolioSpec
Defines a portfolio of candidate research studies for optimization with properties:
- `studies`: List of PortfolioStudy objects
- `budget`: Total available budget for research

## Usage Example

```python
import numpy as np
from voiage.methods.portfolio import portfolio_voi, PortfolioSpec, PortfolioStudy

# Define candidate studies
studies = [
    PortfolioStudy(name="Study A", cost=100, value=50),
    PortfolioStudy(name="Study B", cost=200, value=120),
    PortfolioStudy(name="Study C", cost=150, value=80),
    PortfolioStudy(name="Study D", cost=300, value=200)
]

# Define portfolio specification
portfolio_spec = PortfolioSpec(studies=studies, budget=400)

# Optimize portfolio using greedy algorithm
optimal_portfolio = portfolio_voi(portfolio_spec, optimization_method="greedy")

print("Optimal portfolio:")
for study in optimal_portfolio.selected_studies:
    print(f"  {study.name}: Cost={study.cost}, Value={study.value}")
print(f"Total value: {optimal_portfolio.total_value}")
print(f"Total cost: {optimal_portfolio.total_cost}")
```

## Optimization Methods

### Greedy Algorithm
The default optimization method that selects studies based on value-to-cost ratio:

```python
optimal_portfolio = portfolio_voi(portfolio_spec, optimization_method="greedy")
```

### Exhaustive Search
For small portfolios, you can use exhaustive search to find the globally optimal solution:

```python
optimal_portfolio = portfolio_voi(portfolio_spec, optimization_method="exhaustive")
```

## Advanced Usage

### Custom Value Functions
You can provide a custom function to calculate study values:

```python
def custom_value_calculator(study):
    # Custom logic to calculate value
    return study.value * 1.2  # Example adjustment

optimal_portfolio = portfolio_voi(
    portfolio_spec, 
    study_value_calculator=custom_value_calculator
)
```

## Best Practices

1. **Study Definition**: Clearly define each candidate study with realistic costs and values
2. **Budget Constraints**: Ensure the budget constraint reflects real-world limitations
3. **Value Assessment**: Use robust methods to estimate the value of each study
4. **Sensitivity Analysis**: Perform sensitivity analysis on study costs and values