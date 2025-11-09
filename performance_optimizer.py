#!/usr/bin/env python3
"""
Phase 1.4: Performance Profiling & Optimization
Comprehensive performance optimization module for voiage JAX backend
"""

import jax
import jax.numpy as jnp
import time
import gc
import os
from typing import Optional, Dict, Any, Callable
from functools import wraps

class PerformanceOptimizer:
    """Advanced performance optimization utilities for JAX backend."""
    
    def __init__(self, enable_64bit: bool = True, enable_gpu: bool = True):
        """Initialize performance optimizer."""
        self.enable_64bit = enable_64bit
        self.enable_gpu = enable_gpu
        self._optimize_jax_config()
        
    def _optimize_jax_config(self):
        """Optimize JAX configuration for better performance."""
        # Enable 64-bit precision
        if self.enable_64bit:
            jax.config.update("jax_enable_x64", True)
            print("✅ JAX 64-bit precision enabled")
            
        # Enable GPU acceleration
        if self.enable_gpu:
            # JAX will automatically use GPU if available
            try:
                devices = jax.devices()
                if len(devices) > 1:
                    print(f"✅ GPU acceleration enabled: {len(devices)} devices available")
                else:
                    print("ℹ️  GPU acceleration: Using CPU (no GPU devices found)")
            except Exception as e:
                print(f"⚠️  GPU acceleration check failed: {e}")
        
        # Set memory pool for GPU
        if self.enable_gpu and 'CUDA_VISIBLE_DEVICES' in os.environ:
            print("✅ CUDA environment detected")
    
    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile function performance with detailed metrics."""
        # Force garbage collection before profiling
        gc.collect()
        
        # Warm-up run (for JIT compilation)
        try:
            _ = func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️  Warm-up run failed: {e}")
        
        # Get memory before
        gc.collect()
        
        # Timed runs
        num_runs = 10
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"❌ Run {i+1} failed: {e}")
                continue
        
        if not times:
            return None
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        results = {
            'function': func.__name__,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'num_runs': len(times),
            'std_time': (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        }
        
        return results
    
    def optimize_jit_function(self, func: Callable) -> Callable:
        """Optimize a function for JAX JIT compilation."""
        
        @wraps(func)
        def optimized_func(*args, **kwargs):
            # Add memory optimization hints
            with jax.default_matmul_precision("float32"):
                return func(*args, **kwargs)
        
        # Compile with optimizations
        compiled = jax.jit(optimized_func)
        
        return compiled
    
    def memory_efficient_computation(self, arrays_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory-efficient computation for large arrays."""
        optimized = {}
        
        for name, array in arrays_dict.items():
            if hasattr(array, 'device_buffer'):
                # JAX array
                # Use in-place operations where possible
                optimized[name] = jax.lax.stop_gradient(array)
            else:
                # NumPy array
                optimized[name] = jnp.asarray(array)
        
        return optimized


def enable_advanced_jax_features():
    """Enable advanced JAX features for better performance."""
    # Set compile optimizations
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    
    # Enable optimizations
    jax.config.update("jax_enable_x64", True)
    
    # Set parallel operations (removed invalid config)
    # jax.config.update("jax_numpy_promotion", "standard")  # Invalid config option
    
    print("✅ Advanced JAX features enabled")
    print(f"   • 64-bit precision: Enabled")
    print(f"   • Compilation cache: /tmp/jax_cache")
    print(f"   • Advanced features: Configured")


def create_performance_profile() -> Dict[str, Any]:
    """Create detailed performance profile of JAX environment."""
    profile = {
        'jax_version': jax.__version__,
        'jaxlib_version': jax.lib.__version__,
        'devices': [str(device) for device in jax.devices()],
        'device_count': len(jax.devices()),
        'device_types': list(set(str(device.device_kind) for device in jax.devices())),
        'memory_info': {},
        'config': dict(jax.config.values)
    }
    
    # Get memory info for each device
    for device in jax.devices():
        try:
            if hasattr(device, 'memory_stats'):
                profile['memory_info'][str(device)] = device.memory_stats()
        except Exception:
            pass
    
    return profile


if __name__ == "__main__":
    # Test the performance optimizer
    print("🔧 Phase 1.4 Performance Profiling & Optimization")
    print("=" * 55)
    
    # Enable advanced features
    enable_advanced_jax_features()
    
    # Create performance profile
    profile = create_performance_profile()
    
    print(f"\n📊 JAX Environment Profile:")
    print(f"   Version: {profile['jax_version']}")
    print(f"   Devices: {profile['device_count']}")
    print(f"   Device Types: {profile['device_types']}")
    
    # Test simple computation
    def test_computation():
        # Create test data
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (1000, 1000))
        y = jax.random.normal(key, (1000, 1000))
        
        # Matrix multiplication test
        z = jnp.dot(x, y)
        return jnp.mean(z)
    
    optimizer = PerformanceOptimizer()
    results = optimizer.profile_function(test_computation)
    
    if results:
        print(f"\n🧮 Performance Test Results:")
        print(f"   Average time: {results['avg_time']:.6f}s")
        print(f"   Min time: {results['min_time']:.6f}s")
        print(f"   Max time: {results['max_time']:.6f}s")
        print(f"   Std dev: {results['std_time']:.6f}s")
    
    print("\n✅ Phase 1.4 Performance Optimization Ready!")