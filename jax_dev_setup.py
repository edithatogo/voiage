#!/usr/bin/env python3
"""
JAX Development Environment Setup Script for voiage v0.3.0

This script sets up a JAX-optimized development environment including:
- JAX-specific development tools
- Performance monitoring
- Development utilities
- Testing configurations

Usage:
    python jax_dev_setup.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def create_jax_config():
    """Create JAX configuration file."""
    config = {
        "jax": {
            "version": "0.4.38",
            "device": "cpu",  # Will be updated based on availability
            "xla_flags": [
                "--xla_cpu_enable_fast_math=true",
                "--xla_cpu_enable_fast_min_max=true"
            ],
            "enable_jit": True,
            "array_types": ["jax.numpy", "numpy"]
        },
        "development": {
            "use_jit_by_default": False,  # Enable during development
            "enable_debug_logging": True,
            "profile_jit_compilation": True,
            "memory_efficient_mode": False
        },
        "performance": {
            "enable_pmap": True,
            "enable_vmap": True,
            "chunk_size": 10000,  # For large array processing
            "cache_size": 100  # For compiled functions
        },
        "monitoring": {
            "enable_memory_monitoring": True,
            "enable_performance_profiling": True,
            "log_device_placement": True
        }
    }
    
    config_file = Path("jax_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ JAX configuration saved to {config_file}")
    return config

def create_jax_development_scripts():
    """Create JAX-specific development scripts."""
    
    # JAX development script
    dev_script = '''#!/usr/bin/env python3
"""
JAX Development Utilities for voiage
"""

import os
import sys
import warnings
from typing import Optional

# JAX imports with proper error handling
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    print(f"Warning: JAX not available: {e}")

# Environment configuration
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_math=true")

class JaxDevContext:
    """Context manager for JAX development."""
    
    def __init__(self, enable_jit: bool = True, debug_mode: bool = False):
        self.enable_jit = enable_jit
        self.debug_mode = debug_mode
        self.original_flags = os.environ.get("XLA_FLAGS", "")
    
    def __enter__(self):
        if self.debug_mode:
            # Enable JAX debug logging
            os.environ["JAX_DEBUG_MIN_ITER_LEVEL"] = "1"
            warnings.filterwarnings("ignore")
        
        if JAX_AVAILABLE and self.enable_jit:
            print("🔧 JAX Development Context: JIT compilation enabled")
            # JIT-compiled versions will be created on-demand
        else:
            print("🔧 JAX Development Context: JAX not available or JIT disabled")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original environment
        if "JAX_DEBUG_MIN_ITER_LEVEL" in os.environ:
            del os.environ["JAX_DEBUG_MIN_ITER_LEVEL"]

def create_jax_test_array(shape: tuple, dtype: str = "float64") -> "jnp.ndarray":
    """Create a JAX array for testing."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX not available")
    
    # Use a fixed seed for reproducible tests
    key = jrandom.key(42)
    if dtype == "float64":
        return jrandom.normal(key, shape, dtype=jnp.float64)
    elif dtype == "float32":
        return jrandom.normal(key, shape, dtype=jnp.float32)
    else:
        return jrandom.normal(key, shape)

def benchmark_jax_function(func, array_sizes: list, **kwargs):
    """Benchmark a JAX function with different array sizes."""
    if not JAX_AVAILABLE:
        print("JAX not available for benchmarking")
        return
    
    results = {}
    for size in array_sizes:
        test_array = create_jax_test_array((size, size))
        print(f"Benchmarking with {size}x{size} array...")
        
        import time
        start_time = time.perf_counter()
        result = func(test_array, **kwargs)
        end_time = time.perf_counter()
        
        results[size] = {
            "time": end_time - start_time,
            "result_shape": getattr(result, 'shape', 'scalar')
        }
    
    return results

# Development utilities
def check_jax_health():
    """Check JAX health and performance."""
    if not JAX_AVAILABLE:
        print("❌ JAX not available")
        return False
    
    try:
        # Test basic functionality
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        c = a + b
        
        # Test JIT compilation
        @jax.jit
        def test_function(x):
            return jnp.sum(x**2)
        
        result = test_function(jnp.array([1, 2, 3]))
        
        print("✅ JAX health check passed")
        print(f"   Version: {jax.__version__}")
        print(f"   Devices: {len(jax.devices())}")
        print(f"   Test result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX health check failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JAX Development Utilities")
    parser.add_argument("--health", action="store_true", help="Run JAX health check")
    parser.add_argument("--context", action="store_true", help="Start JAX development context")
    parser.add_argument("--config", action="store_true", help="Show JAX configuration")
    
    args = parser.parse_args()
    
    if args.health:
        check_jax_health()
    elif args.context:
        with JaxDevContext(enable_jit=True, debug_mode=True):
            print("🔧 JAX development context active")
            print("   Use Ctrl+C to exit")
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\\n🛑 Exiting JAX development context")
    elif args.config:
        print("JAX Development Configuration")
        print("-" * 40)
        print("Environment variables:")
        for key, value in os.environ.items():
            if key.startswith("XLA_") or key.startswith("JAX_"):
                print(f"  {key}: {value}")
        print(f"\\nJAX available: {JAX_AVAILABLE}")
        if JAX_AVAILABLE:
            print(f"JAX version: {jax.__version__}")
            print(f"JAX devices: {jax.devices()}")
'''
    
    with open("jax_dev_utils.py", "w") as f:
        f.write(dev_script)
    
    # Make executable
    os.chmod("jax_dev_utils.py", 0o755)
    print("✅ JAX development utilities created: jax_dev_utils.py")

def create_test_fixtures():
    """Create JAX-specific test fixtures."""
    
    test_fixtures = '''#!/usr/bin/env python3
"""
JAX Test Fixtures for voiage
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# JAX fixtures
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

@pytest.fixture(scope="session")
def jax_devices():
    """Provide JAX devices information."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    devices = jax.devices()
    return {
        "cpu": [d for d in devices if "cpu" in str(d).lower()],
        "gpu": [d for d in devices if "gpu" in str(d).lower()],
        "all": devices
    }

@pytest.fixture
def jax_key():
    """Provide a JAX random key for reproducible tests."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    return jrandom.key(42)

@pytest.fixture
def jax_test_arrays(jax_key):
    """Provide JAX test arrays of different sizes."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    def create_arrays(size):
        return jrandom.normal(jax_key, (size, size), dtype=jnp.float64)
    
    return {
        "small": create_arrays(10),
        "medium": create_arrays(100),
        "large": create_arrays(1000)
    }

@pytest.fixture
def numpy_test_arrays():
    """Provide NumPy test arrays for comparison."""
    np.random.seed(42)
    
    def create_arrays(size):
        return np.random.randn(size, size).astype(np.float64)
    
    return {
        "small": create_arrays(10),
        "medium": create_arrays(100),
        "large": create_arrays(1000)
    }

@pytest.fixture
def comparison_data(jax_test_arrays, numpy_test_arrays):
    """Provide paired JAX and NumPy arrays for comparison testing."""
    return {
        "small": {
            "jax": jax_test_arrays["small"],
            "numpy": numpy_test_arrays["small"]
        },
        "medium": {
            "jax": jax_test_arrays["medium"],
            "numpy": numpy_test_arrays["medium"]
        },
        "large": {
            "jax": jax_test_arrays["large"],
            "numpy": numpy_test_arrays["large"]
        }
    }

@pytest.fixture
def evpi_test_data():
    """Provide test data for EVPI calculations."""
    np.random.seed(42)
    
    # Small dataset
    small_nb = np.random.randn(100, 4) * 100 + 1000
    
    # Medium dataset  
    medium_nb = np.random.randn(1000, 6) * 100 + 1000
    
    return {
        "small": small_nb,
        "medium": medium_nb,
        "option_names": ["Option A", "Option B", "Option C", "Option D", "Option E", "Option F"],
        "parameter_names": [f"param_{i}" for i in range(5)]
    }

class JaxTestHelper:
    """Helper class for JAX testing."""
    
    def __init__(self, jax_key):
        self.key = jax_key
    
    def create_test_array(self, shape, dtype=jnp.float64):
        """Create a test array with the given shape and dtype."""
        return jrandom.normal(self.key, shape, dtype=dtype)
    
    def assert_array_close(self, jax_array, numpy_array, rtol=1e-6, atol=1e-6):
        """Assert that JAX and NumPy arrays are close."""
        if JAX_AVAILABLE:
            jax_result = jnp.asarray(jax_array)
            np_result = numpy_array
        else:
            jax_result = jax_array
            np_result = numpy_array
        
        np.testing.assert_allclose(jax_result, np_result, rtol=rtol, atol=atol)
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function with timing."""
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        return {
            "result": result,
            "time": end_time - start_time
        }

@pytest.fixture
def jax_helper(jax_key):
    """Provide a JAX test helper instance."""
    return JaxTestHelper(jax_key)

# Skip JAX tests if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
'''
    
    with open("test_fixtures_jax.py", "w") as f:
        f.write(test_fixtures)
    
    print("✅ JAX test fixtures created: test_fixtures_jax.py")

def create_development_readme():
    """Create development README for JAX integration."""
    
    readme = '''# JAX Development Guide for voiage v0.3.0

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
'''
    
    with open("JAX_DEVELOPMENT_GUIDE.md", "w") as f:
        f.write(readme)
    
    print("✅ JAX development guide created: JAX_DEVELOPMENT_GUIDE.md")

def main():
    """Main setup function."""
    print("🛠️  JAX Development Environment Setup")
    print("=" * 50)
    
    # Create JAX configuration
    config = create_jax_config()
    
    # Create development scripts
    create_jax_development_scripts()
    
    # Create test fixtures
    create_test_fixtures()
    
    # Create development guide
    create_development_readme()
    
    # Final setup verification
    print("\n🔍 Final Setup Verification")
    print("=" * 50)
    
    # Check if JAX is available
    try:
        import jax
        print(f"✅ JAX Version: {jax.__version__}")
        print(f"✅ JAX Devices: {len(jax.devices())}")
    except ImportError:
        print("❌ JAX not available - install with: pip install jax jaxlib")
    
    # Verify created files
    files_created = [
        "jax_config.json",
        "jax_dev_utils.py", 
        "test_fixtures_jax.py",
        "JAX_DEVELOPMENT_GUIDE.md"
    ]
    
    print(f"\n📁 Created Files:")
    for file in files_created:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (not created)")
    
    print(f"\n🎯 JAX Development Environment Setup Complete!")
    print(f"🚀 Ready to proceed with JAX backend implementation!")
    
    return True

if __name__ == "__main__":
    main()