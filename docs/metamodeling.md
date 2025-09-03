# Metamodeling in voiage

The voiage library provides several metamodeling approaches for Value of Information analysis. Metamodels (also known as surrogate models) are used to approximate complex models with simpler, faster-to-evaluate functions.

## Available Metamodels

### 1. Random Forest Metamodel

A Random Forest regressor that can capture non-linear relationships and interactions between parameters.

```python
from voiage.metamodels import RandomForestMetamodel

model = RandomForestMetamodel(n_estimators=100, random_state=42)
model.fit(x, y)
predictions = model.predict(x)
```

### 2. Generalized Additive Model (GAM)

A GAM that uses spline functions to model non-linear relationships for each parameter.

```python
from voiage.metamodels import GAMMetamodel

model = GAMMetamodel(n_splines=10)
model.fit(x, y)
predictions = model.predict(x)
```

### 3. Bayesian Additive Regression Trees (BART)

A Bayesian non-parametric approach that uses a sum of regression trees with Bayesian inference.

```python
from voiage.metamodels import BARTMetamodel

model = BARTMetamodel(num_trees=50)
model.fit(x, y)
predictions = model.predict(x)
```

## Diagnostics and Validation

### Calculate Diagnostics

The `calculate_diagnostics` function provides comprehensive metrics for evaluating metamodel performance:

```python
from voiage.metamodels import calculate_diagnostics

diagnostics = calculate_diagnostics(model, x, y)
# Returns: {"r2", "rmse", "mae", "mean_residual", "std_residual", "n_samples"}
```

### Cross-Validation

The `cross_validate` function performs k-fold cross-validation:

```python
from voiage.metamodels import cross_validate

cv_results = cross_validate(RandomForestMetamodel, x, y, cv_folds=5)
# Returns cross-validation statistics
```

### Model Comparison

The `compare_metamodels` function compares multiple models using cross-validation:

```python
from voiage.metamodels import compare_metamodels

models = [RandomForestMetamodel, GAMMetamodel, BARTMetamodel]
comparison = compare_metamodels(models, x, y, cv_folds=3)
```

## Example Usage

```python
import numpy as np
import xarray as xr
from voiage.schema import ParameterSet
from voiage.metamodels import RandomForestMetamodel, calculate_diagnostics

# Create sample data
n_samples = 100
param1 = np.random.rand(n_samples)
param2 = np.random.rand(n_samples)
y = 2 * param1 + 3 * param2 + np.random.randn(n_samples) * 0.1

# Create ParameterSet
data = {
    "param1": ("n_samples", param1),
    "param2": ("n_samples", param2),
}
x = ParameterSet(dataset=xr.Dataset(data))

# Fit and evaluate model
model = RandomForestMetamodel(n_estimators=50)
model.fit(x, y)

# Make predictions
predictions = model.predict(x)

# Calculate diagnostics
diagnostics = calculate_diagnostics(model, x, y)
print(f"R²: {diagnostics['r2']:.4f}")
print(f"RMSE: {diagnostics['rmse']:.4f}")
```

## API Reference

### Metamodel Protocol

All metamodels implement the `Metamodel` protocol with the following methods:

- `fit(x: ParameterSet, y: np.ndarray) -> None`: Fit the model to data
- `predict(x: ParameterSet) -> np.ndarray`: Make predictions
- `score(x: ParameterSet, y: np.ndarray) -> float`: Calculate R² score
- `rmse(x: ParameterSet, y: np.ndarray) -> float`: Calculate RMSE

### Utility Functions

- `calculate_diagnostics(model: Metamodel, x: ParameterSet, y: np.ndarray) -> dict`
- `cross_validate(model: Metamodel, x: ParameterSet, y: np.ndarray, cv_folds: int = 5) -> dict`
- `compare_metamodels(models: list, x: ParameterSet, y: np.ndarray, cv_folds: int = 5) -> dict`

## Dependencies

Some metamodels require additional packages:

- Random Forest: `scikit-learn`
- GAM: `pygam`
- BART: `pymc` and `pymc-bart`

Install optional dependencies:
```bash
pip install scikit-learn pygam pymc pymc-bart
```