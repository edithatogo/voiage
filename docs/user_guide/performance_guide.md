# Performance Optimization Guide

This guide provides recommendations for optimizing performance when using voiage for Value of Information analysis.

## Overview

voiage is designed for performance, but there are several strategies you can use to optimize your analyses further, especially when working with large datasets or complex models.

## Data Structure Optimization

### Use Appropriate Data Types

Ensure you're using the most efficient data types for your analysis:

```python
import numpy as np
from voiage.config import DEFAULT_DTYPE

# Use the default data type for consistency and performance
data = np.array(your_data, dtype=DEFAULT_DTYPE)
```

### Efficient Data Loading

When loading large datasets, consider using chunked loading:

```python
import xarray as xr

# Load data in chunks to manage memory usage
dataset = xr.open_dataset('large_dataset.nc', chunks={'n_samples': 1000})
```

## Computational Optimization

### Sample Size Management

For large PSA datasets, consider using a subset for initial analysis:

```python
from voiage.analysis import DecisionAnalysis

# Use a subset of samples for initial exploration
subset_indices = np.random.choice(len(all_samples), 1000, replace=False)
subset_data = all_samples[subset_indices]

analysis = DecisionAnalysis(nb_array=subset_data)
```

### Regression Sample Control

For EVPPI calculations, control the number of samples used for regression:

```python
# Use fewer samples for regression to speed up computation
evppi_result = analysis.evppi(n_regression_samples=500)
```

## Backend Optimization

### NumPy Backend

The default NumPy backend is optimized for most use cases:

```python
# Ensure you're using the NumPy backend (default)
analysis = DecisionAnalysis(nb_array=data, backend="numpy")
```

### JAX Backend

voiage supports JAX for performance optimization, which can provide significant speedups for large datasets:

```python
# Use the JAX backend for improved performance
analysis = DecisionAnalysis(nb_array=data, backend="jax")

# Enable JIT compilation for even better performance
analysis_jit = DecisionAnalysis(nb_array=data, backend="jax", use_jit=True)

# Compare performance
import time

# NumPy backend
start = time.time()
evpi_numpy = analysis_numpy.evpi()
time_numpy = time.time() - start

# JAX backend
start = time.time()
evpi_jax = analysis_jax.evpi()
time_jax = time.time() - start

# JAX backend with JIT
start = time.time()
evpi_jax_jit = analysis_jit.evpi()
time_jax_jit = time.time() - start

print(f"NumPy: {time_numpy:.4f}s")
print(f"JAX: {time_jax:.4f}s")
print(f"JAX + JIT: {time_jax_jit:.4f}s")
```

#### Performance Benefits of JAX

The JAX backend provides several performance benefits:

1. **Just-In-Time (JIT) Compilation**: Functions are compiled to optimized machine code
2. **Vectorization**: Operations are automatically vectorized for better performance
3. **GPU/TPU Support**: Can leverage accelerators when available
4. **Automatic Differentiation**: Enables gradient-based optimizations

#### When to Use JAX

Consider using the JAX backend when:

- Working with large datasets (>10,000 samples)
- Performing repeated calculations
- You have access to GPU/TPU hardware
- You need maximum computational performance

#### Installation Requirements

To use the JAX backend, you need to install JAX:

```bash
pip install jax jaxlib
```

## Parallel Processing

### Using Multiple Cores

For operations that support it, use multiple cores:

```python
import multiprocessing as mp

# Example of parallel processing for multiple analyses
def run_analysis(data_chunk):
    analysis = DecisionAnalysis(nb_array=data_chunk)
    return analysis.evpi()

# Split data into chunks
chunks = np.array_split(large_dataset, mp.cpu_count())

# Process in parallel
with mp.Pool() as pool:
    results = pool.map(run_analysis, chunks)
```

## Memory Management

### Efficient Memory Usage

Monitor and manage memory usage for large analyses:

```python
import psutil
import gc

# Check memory usage
memory_usage = psutil.virtual_memory().percent
print(f"Memory usage: {memory_usage}%")

# Force garbage collection if needed
if memory_usage > 80:
    gc.collect()
```

## Profiling and Benchmarking

### Performance Profiling

Use Python's profiling tools to identify bottlenecks:

```python
import cProfile

# Profile your analysis
cProfile.run('analysis.evpi()', 'evpi_profile.stats')
```

### Benchmarking

Compare performance of different approaches:

```python
import time

# Benchmark different approaches
start_time = time.time()
result1 = analysis.evpi()
time1 = time.time() - start_time

start_time = time.time()
result2 = analysis.evpi(n_regression_samples=500)
time2 = time.time() - start_time

print(f"Full sample EVPI: {time1:.2f}s")
print(f"Subsampled EVPI: {time2:.2f}s")
```

## Best Practices

1. **Start Small**: Begin with smaller datasets to test your approach
2. **Profile Regularly**: Regularly profile your code to identify performance issues
3. **Optimize Iteratively**: Make incremental improvements based on profiling results
4. **Use Appropriate Hardware**: Ensure you're using appropriate hardware for your analysis needs
5. **Cache Results**: Cache expensive computations when possible
6. **Leverage JAX**: For large-scale analyses, consider using the JAX backend with JIT compilation
7. **Monitor Memory**: Keep an eye on memory usage, especially with large datasets