#!/usr/bin/env python3
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
                print("\n🛑 Exiting JAX development context")
    elif args.config:
        print("JAX Development Configuration")
        print("-" * 40)
        print("Environment variables:")
        for key, value in os.environ.items():
            if key.startswith("XLA_") or key.startswith("JAX_"):
                print(f"  {key}: {value}")
        print(f"\nJAX available: {JAX_AVAILABLE}")
        if JAX_AVAILABLE:
            print(f"JAX version: {jax.__version__}")
            print(f"JAX devices: {jax.devices()}")
