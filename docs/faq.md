# Frequently Asked Questions (FAQ)

This document addresses common questions and usage patterns for the voiage library.

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation and Setup](#installation-and-setup)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Integration with Other Tools](#integration-with-other-tools)

## General Questions

### What is Value of Information (VOI) analysis?

Value of Information (VOI) analysis is a decision-analytic framework used to quantify the expected benefit of reducing uncertainty in decision models. It helps decision-makers determine whether additional research or data collection is economically worthwhile.

The main VOI metrics include:
- **EVPI (Expected Value of Perfect Information)**: The maximum amount one should be willing to pay to eliminate all uncertainty
- **EVPPI (Expected Value of Partial Perfect Information)**: The value of eliminating uncertainty for specific parameters
- **EVSI (Expected Value of Sample Information)**: The value of a specific proposed study

### What makes voiage different from other VOI libraries?

voiage offers several unique features:

1. **Comprehensive Implementation**: Covers a wide range of VOI analyses including basic EVPI/EVPPI to advanced methods like adaptive design EVSI and portfolio VOI
2. **Modern Python API**: Leverages the Python scientific computing ecosystem (NumPy, SciPy, Pandas, xarray)
3. **Extensible Architecture**: Designed for easy extension with new methods and metamodels
4. **Performance Optimizations**: Includes chunked processing, caching, parallel processing, and GPU acceleration
5. **Cross-Domain Support**: Not limited to health economics; applicable to business strategy, environmental policy, etc.
6. **Advanced Features**: Includes streaming data support, incremental computation, and memory optimization

### Who should use voiage?

voiage is intended for:
- Health economists and HTA practitioners
- Decision analysts and operations researchers
- Researchers in business strategy and environmental policy
- Anyone interested in quantifying the value of information in decision-making

## Installation and Setup

### How do I install voiage?

You can install voiage using pip:

```bash
pip install voiage
```

For development or the latest features, you can install from source:

```bash
git clone https://github.com/your-username/voiage.git
cd voiage
pip install -e .
```

### What are the system requirements?

voiage requires:
- Python 3.7 or higher
- NumPy, SciPy, Pandas, xarray
- Matplotlib for plotting (optional)
- scikit-learn for some advanced metamodels (optional)
- JAX for GPU acceleration (optional)
- PyTorch for deep learning metamodels (optional)

### How do I install optional dependencies?

To install all optional dependencies:

```bash
pip install voiage[all]
```

To install specific optional dependencies:

```bash
# For plotting
pip install voiage[plot]

# For advanced metamodels
pip install voiage[metamodels]

# For GPU acceleration
pip install voiage[gpu]

# For deep learning metamodels
pip install voiage[deep-learning]
```

## Basic Usage

### How do I perform a basic EVPI calculation?

Here's a simple example of calculating EVPI:

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

# Create sample net benefit data (samples x strategies)
net_benefits = np.array([
    [100, 120, 90],   # Sample 1
    [110, 100, 130],  # Sample 2
    [90, 140, 110],   # Sample 3
    # ... more samples
])

# Create ValueArray
value_array = ValueArray.from_numpy(net_benefits)

# Create decision analysis
analysis = DecisionAnalysis(value_array)

# Calculate EVPI
evpi_result = analysis.evpi()
print(f"EVPI: {evpi_result}")
```

### How do I include population scaling in my calculations?

You can include population parameters to scale your VOI results:

```python
# Calculate population-adjusted EVPI
population_evpi = analysis.evpi(
    population=100000,      # Population size
    time_horizon=10,        # Time horizon in years
    discount_rate=0.03      # Annual discount rate
)
print(f"Population EVPI: {population_evpi}")
```

### How do I perform EVPPI calculations?

EVPPI requires parameter samples:

```python
from voiage.schema import ParameterSet

# Create parameter samples
parameters = {
    'param1': np.array([1.0, 1.2, 0.8, 1.1]),
    'param2': np.array([0.5, 0.6, 0.4, 0.5])
}
parameter_set = ParameterSet.from_numpy_or_dict(parameters)

# Create decision analysis with parameters
analysis = DecisionAnalysis(value_array, parameter_set)

# Calculate EVPPI
evppi_result = analysis.evppi()
print(f"EVPPI: {evppi_result}")
```

## Advanced Features

### How do I use streaming data support?

Streaming data support allows continuous VOI updates as new data arrives:

```python
# Create analysis with streaming support
analysis = DecisionAnalysis(value_array, streaming_window_size=1000)

# Get streaming EVPI generator
evpi_generator = analysis.streaming_evpi()

# Update with new data
new_data = np.random.randn(200, 3) * 100 + 5000
new_value_array = ValueArray.from_numpy(new_data)
analysis.update_with_new_data(new_value_array)

# Get updated EVPI
updated_evpi = next(evpi_generator)
print(f"Updated EVPI: {updated_evpi}")
```

### How do I enable caching for repeated calculations?

Caching can significantly speed up repeated calculations:

```python
# Create analysis with caching enabled
analysis = DecisionAnalysis(value_array, enable_caching=True)

# First calculation (no cache)
result1 = analysis.evpi()

# Second calculation (uses cache)
result2 = analysis.evpi()

# Results should be identical
assert result1 == result2
```

### How do I use incremental computation for large datasets?

For large datasets, use chunked processing:

```python
# For large datasets, use chunked processing
large_evpi = analysis.evpi(chunk_size=10000)
```

### How do I leverage GPU acceleration?

If you have compatible hardware, GPU acceleration can speed up computations:

```python
from voiage.core.gpu_acceleration import is_gpu_available

if is_gpu_available():
    # GPU acceleration is available
    evpi_result = analysis.evpi(use_gpu=True)
```

## Performance Optimization

### How can I optimize memory usage for large analyses?

For large-scale analyses, consider these memory optimization techniques:

```python
from voiage.core.memory_optimization import MemoryOptimizer

# Create memory optimizer
optimizer = MemoryOptimizer()

# Optimize data types
optimized_array = optimizer.optimize_array_dtype(large_net_benefits)

# Use chunked computation
chunked_results = optimizer.chunked_computation(
    your_function, 
    large_data, 
    chunk_size=10000
)
```

### How do I use parallel processing?

Parallel processing can speed up Monte Carlo simulations:

```python
from voiage.parallel.monte_carlo import parallel_monte_carlo_simulation

# Run parallel Monte Carlo simulation
result = parallel_monte_carlo_simulation(
    model_func=your_model_function,
    psa_prior=parameter_set,
    trial_design=trial_design,
    n_simulations=10000,
    n_workers=4,  # Number of parallel workers
    use_processes=True  # Use processes instead of threads
)
```

### What are the best practices for performance?

1. **Use appropriate data types**: Use float32 instead of float64 when precision allows
2. **Enable caching**: For repeated calculations with the same data
3. **Use chunked processing**: For large datasets that don't fit in memory
4. **Leverage parallel processing**: For Monte Carlo simulations and other embarrassingly parallel tasks
5. **Use GPU acceleration**: When available, for computationally intensive operations
6. **Monitor memory usage**: Keep an eye on memory consumption, especially with large datasets

## Troubleshooting

### I'm getting dtype validation errors. How do I fix this?

Ensure your arrays use the correct data type (typically float64):

```python
# Make sure to use float64 dtype
net_benefits = np.array([[100.0, 120.0], [110.0, 100.0]], dtype=np.float64)
```

### My EVPPI calculation is slow. How can I speed it up?

EVPPI can be computationally intensive. Consider these optimizations:

1. **Subsample for regression fitting**:
```python
evppi_result = analysis.evppi(n_regression_samples=1000)
```

2. **Use incremental computation**:
```python
evppi_result = analysis.evppi(chunk_size=5000)
```

3. **Use a more efficient regression model**:
```python
from sklearn.linear_model import Ridge
evppi_result = analysis.evppi(regression_model=Ridge)
```

### I'm getting memory errors with large datasets. What can I do?

For memory issues with large datasets:

1. **Use chunked processing**:
```python
result = analysis.evpi(chunk_size=10000)
```

2. **Optimize data types**:
```python
# Use float32 instead of float64 if precision allows
net_benefits = net_benefits.astype(np.float32)
```

3. **Use memory-efficient computation**:
```python
from voiage.core.memory_optimization import memory_efficient_evpi_computation
result = memory_efficient_evpi_computation(large_net_benefits)
```

### My plotting functions aren't working. What's wrong?

Make sure you have matplotlib installed:

```bash
pip install matplotlib
```

Then import the plotting modules:

```python
from voiage.plot import ceac, voi_curves
```

## Integration with Other Tools

### How do I integrate voiage with JAX?

voiage has built-in JAX support for GPU/TPU acceleration:

```python
# Set JAX backend
from voiage.backends import set_backend
set_backend("jax")

# Use JIT compilation
analysis = DecisionAnalysis(value_array, use_jit=True)
```

### How do I use voiage with PyTorch?

PyTorch integration is available for deep learning metamodels:

```python
from voiage.metamodels import PyTorchNNMetamodel

# Create PyTorch-based metamodel
metamodel = PyTorchNNMetamodel()
```

### How do I export results for use in other tools?

You can export results in standard formats:

```python
import pandas as pd

# Convert results to pandas DataFrame
results_df = pd.DataFrame({
    'EVPI': [evpi_result],
    'EVPPI': [evppi_result]
})

# Save to CSV
results_df.to_csv('voi_results.csv', index=False)
```

### Can I use voiage with R?

While voiage is a Python library, you can call it from R using the `reticulate` package:

```r
# In R
library(reticulate)
voiage <- import("voiage")
```

## Additional Resources

For more detailed information, check out:
- [User Guide](user_guide/index.md)
- [API Reference](api/index.md)
- [Examples](../examples/)
- [Tutorials](../examples/interactive_tutorial.ipynb)
- [Visualization Gallery](../examples/visualization_gallery.ipynb)