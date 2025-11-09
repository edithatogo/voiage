# JAX Development Guide for voiage v0.3.0

This document provides guidance for developing and testing JAX backend integration in voiage.

## Environment Setup

### JAX Configuration
JAX development environment is configured through `jax_config.json`. Key settings:

- **Device**: CPU by default (GPU when available)
- **JIT Compilation**: Enabled for performance
- **Debug Mode**: Enable during development
- **Memory Monitoring**: Track memory usage during testing

### Development Tools

#### JAX Development Utilities
```bash
# Check JAX health
python jax_dev_utils.py --health

# Start development context with JIT enabled
python jax_dev_utils.py --context

# Show JAX configuration
python jax_dev_utils.py --config
```

#### Environment Context Manager
```python
from jax_dev_utils import JaxDevContext

with JaxDevContext(enable_jit=True, debug_mode=True):
    # JAX development with JIT compilation
    result = your_jax_function(data)
```

## Testing

### JAX Test Fixtures
The test fixtures provide:

- **JAX devices**: Available CPU/GPU devices
- **Test arrays**: JAX arrays in different sizes
- **Comparison data**: Paired JAX/NumPy arrays for testing
- **Test helper**: Utilities for array comparison and benchmarking

### Running JAX Tests
```bash
# Run JAX-specific tests
pytest -m jax

# Run with JAX health check
pytest --jax-health

# Benchmark JAX vs NumPy performance
pytest --benchmark-only
```

### JAX Test Examples

```python
def test_jax_evpi(jax_test_arrays, jax_helper):
    """Test JAX EVPI implementation."""
    net_benefit = jax_test_arrays["medium"]
    
    # Your JAX implementation
    jax_backend = get_backend("jax")
    result = jax_backend.evpi(net_benefit)
    
    # Validate result
    assert isinstance(result, (float, jax.Array))
    assert result >= 0  # EVPI should be non-negative

def test_jax_numba_equivalence(comparison_data, jax_helper):
    """Test that JAX and NumPy implementations give same results."""
    data = comparison_data["medium"]
    
    # JAX implementation
    jax_backend = get_backend("jax")
    jax_result = jax_backend.evpi(data["jax"])
    
    # NumPy implementation  
    numpy_backend = get_backend("numpy")
    numpy_result = numpy_backend.evpi(data["numpy"])
    
    # Compare results
    jax_helper.assert_array_close(jax_result, numpy_result, rtol=1e-5)
```

## Development Workflow

### 1. Development Environment
```bash
# Set up JAX development environment
python jax_dev_setup.py

# Verify JAX installation
python jax_dev_utils.py --health
```

### 2. Writing JAX Code
```python
import jax
import jax.numpy as jnp
import jax.random as jrandom

@jax.jit  # Enable JIT compilation
def your_jax_function(data):
    # JAX-optimized implementation
    return jnp.sum(data ** 2)
```

### 3. Testing and Validation
```python
# Use test fixtures
def test_your_function(jax_test_arrays, jax_helper):
    result = your_function(jax_test_arrays["medium"])
    jax_helper.assert_array_close(result, expected_result)
    
    # Benchmark performance
    benchmark = jax_helper.benchmark_function(your_function, jax_test_arrays["large"])
    print(f"Execution time: {benchmark['time']:.4f}s")
```

### 4. Performance Optimization
```python
# Use JIT compilation
@jax.jit
def optimized_function(data):
    return jnp.sum(data ** 2)

# Use JAX vectorization
def vectorized_function(data_batch):
    return jax.vmap(your_function)(data_batch)
```

## Common Patterns

### Array Operations
```python
# JAX-style array creation
key = jrandom.key(42)
data = jrandom.normal(key, (100, 100))

# JAX-style mathematical operations  
result = jnp.sum(data ** 2 + 1)

# JAX-style broadcasting
result = jnp.mean(data, axis=0)
```

### JIT Compilation
```python
@jax.jit
def compute_evpi(net_benefit):
    max_nb = jnp.max(net_benefit, axis=1)
    expected_nb_options = jnp.mean(net_benefit, axis=0)
    max_expected_nb = jnp.max(expected_nb_options)
    expected_max_nb = jnp.mean(max_nb)
    return expected_max_nb - max_expected_nb
```

### Error Handling
```python
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Graceful fallback
if not JAX_AVAILABLE:
    # Use NumPy implementation
    backend = get_backend("numpy")
else:
    backend = get_backend("jax")
```

## Performance Tips

1. **JIT Compilation**: Compile expensive functions with `@jax.jit`
2. **JAX Arrays**: Work with JAX arrays for best performance
3. **Device Placement**: Place arrays on appropriate devices
4. **Vectorization**: Use `jax.vmap` for batch operations
5. **Memory Efficiency**: Use `jax.lax.cond` for conditional logic

## Troubleshooting

### Common Issues

1. **JAX Import Errors**
   - Install JAX: `pip install jax jaxlib`
   - For GPU: `pip install jax[cuda]`

2. **Device Not Available**
   - Check device availability: `python jax_dev_utils.py --health`
   - Set device explicitly: `jax.devices("cpu")`

3. **Performance Issues**
   - Enable JIT compilation with `@jax.jit`
   - Use JAX arrays instead of NumPy arrays
   - Check memory usage with `jax.config.update("jax_enable_x64", True)`

### Debugging
```python
# Enable JAX debug logging
os.environ["JAX_DEBUG_MIN_ITER_LEVEL"] = "1"

# Check device placement
print(f"Array device: {array.device()}")

# Monitor memory usage
jax.config.update("jax_enable_compilation_cache", True)
```

## Testing Strategy

### Unit Tests
- Test individual JAX functions
- Compare JAX vs NumPy results
- Validate numerical accuracy

### Integration Tests  
- Test JAX backend integration
- Test DecisionAnalysis class with JAX
- Test end-to-end workflows

### Performance Tests
- Benchmark JAX vs NumPy
- Test with different array sizes
- Validate JIT compilation benefits

This guide should be updated as JAX integration evolves.
