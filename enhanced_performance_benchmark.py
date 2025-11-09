#!/usr/bin/env python3
"""
Phase 1.4 Enhanced Performance Benchmark
Tests optimized JAX backend performance with advanced features enabled
"""

import sys
import os
import time
import numpy as np
import gc

# Add the current directory to the path
sys.path.insert(0, '/Users/doughnut/GitHub/voiage')
sys.path.insert(0, '/Users/doughnut/GitHub/voiage/voiage')

def enable_jax_optimizations():
    """Enable JAX optimizations for Phase 1.4."""
    import jax
    import jax.numpy as jnp
    
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    
    # Set compilation cache for faster JIT compilation
    os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
    
    # Enable memory optimizations
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    
    print("✅ JAX optimizations enabled:")
    print("   • 64-bit precision: Enabled")
    print("   • Compilation cache: /tmp/jax_cache")
    print("   • Memory optimizations: Configured")

def run_enhanced_benchmark():
    """Run enhanced performance benchmark with optimizations."""
    print("🚀 Phase 1.4 Enhanced Performance Benchmark")
    print("=" * 60)
    
    # Enable optimizations first
    enable_jax_optimizations()
    
    try:
        import jax.numpy as jnp
        print("✅ JAX available for enhanced performance testing")
    except ImportError as e:
        print(f"❌ JAX not available: {e}")
        return False

    from voiage.analysis import DecisionAnalysis
    from voiage.schema import ValueArray, ParameterSet
    
    # Enhanced test configurations for larger datasets
    configs = [
        {"name": "Small Dataset", "n_samples": 5000, "n_strategies": 8, "n_params": 5},
        {"name": "Medium Dataset", "n_samples": 50000, "n_strategies": 15, "n_params": 8},
        {"name": "Large Dataset", "n_samples": 500000, "n_strategies": 25, "n_params": 10},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing {config['name']}:")
        print(f"   Samples: {config['n_samples']:,}")
        print(f"   Strategies: {config['n_strategies']}")
        print(f"   Parameters: {config['n_params']}")
        
        # Force garbage collection
        gc.collect()
        
        # Generate test data
        np.random.seed(42)
        
        # NumPy data
        numpy_nb = np.random.normal(1000, 200, (config['n_samples'], config['n_strategies']))
        numpy_params = {
            f'param_{i}': np.random.normal(i, 0.5, config['n_samples'])
            for i in range(config['n_params'])
        }
        
        # JAX data with 64-bit precision
        jax_nb = jnp.array(numpy_nb, dtype=jnp.float64)
        jax_params = {
            name: jnp.array(values, dtype=jnp.float64)
            for name, values in numpy_params.items()
        }
        
        # Create DecisionAnalysis instances
        da_numpy = DecisionAnalysis(numpy_nb, numpy_params, backend="numpy")
        
        # Create JAX instances with optimizations
        da_jax = DecisionAnalysis(
            ValueArray.from_jax(jax_nb),
            ParameterSet.from_jax(jax_params),
            backend="jax"
        )
        
        # Enhanced JAX with JIT
        da_jax_jit = DecisionAnalysis(
            ValueArray.from_jax(jax_nb),
            ParameterSet.from_jax(jax_params),
            backend="jax", 
            use_jit=True
        )
        
        # Test EVPI performance
        print(f"\n   🧮 EVPI Performance Test:")
        
        # NumPy timing
        start_time = time.time()
        for _ in range(3):  # Multiple runs for better averaging
            evpi_numpy = da_numpy.evpi()
        numpy_evpi_time = (time.time() - start_time) / 3
        print(f"      NumPy (avg): {numpy_evpi_time:.4f}s")
        
        # JAX timing (warm-up for JIT)
        start_time = time.time()
        evpi_jax_warmup = da_jax.evpi()
        warmup_time = time.time() - start_time
        
        # JAX timing (actual runs)
        start_time = time.time()
        for _ in range(3):
            evpi_jax = da_jax.evpi()
        jax_evpi_time = (time.time() - start_time) / 3
        print(f"      JAX (avg): {jax_evpi_time:.4f}s")
        print(f"      Warm-up: {warmup_time:.4f}s")
        
        # JAX JIT timing
        start_time = time.time()
        evpi_jax_jit_warmup = da_jax_jit.evpi()
        jit_warmup_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(3):
            evpi_jax_jit = da_jax_jit.evpi()
        jax_jit_time = (time.time() - start_time) / 3
        print(f"      JAX JIT (avg): {jax_jit_time:.4f}s")
        print(f"      JIT Warm-up: {jit_warmup_time:.4f}s")
        
        # Speedup calculations
        speedup_jax = numpy_evpi_time / jax_evpi_time if jax_evpi_time > 0 else 0
        speedup_jit = numpy_evpi_time / jax_jit_time if jax_jit_time > 0 else 0
        
        print(f"\n   📈 Enhanced Performance Analysis:")
        print(f"      JAX Speedup: {speedup_jax:.2f}x")
        print(f"      JAX JIT Speedup: {speedup_jit:.2f}x")
        
        # Target check
        if speedup_jax >= 10.0 or speedup_jit >= 10.0:
            print(f"      🎯 Target Achieved: 10x+ speedup")
        elif speedup_jax >= 5.0 or speedup_jit >= 5.0:
            print(f"      📈 Good Progress: 5x+ speedup")
        else:
            print(f"      💡 Further optimization needed")
        
        # Verify numerical consistency
        evpi_diff = abs(evpi_numpy - float(evpi_jax))
        evpi_diff_jit = abs(evpi_numpy - float(evpi_jax_jit))
        print(f"   🔍 Numerical Check:")
        print(f"      JAX difference: {evpi_diff:.2e} (should be < 1e-3)")
        print(f"      JAX JIT difference: {evpi_diff_jit:.2e} (should be < 1e-3)")
        
        results.append({
            'config': config['name'],
            'samples': config['n_samples'],
            'numpy_time': numpy_evpi_time,
            'jax_time': jax_evpi_time,
            'jax_jit_time': jax_jit_time,
            'speedup_jax': speedup_jax,
            'speedup_jit': speedup_jit,
            'evpi_diff': evpi_diff,
            'evpi_diff_jit': evpi_diff_jit
        })
    
    # Final summary
    print(f"\n🏆 Phase 1.4 Enhanced Performance Summary")
    print("=" * 60)
    
    avg_speedup_jax = sum(r['speedup_jax'] for r in results) / len(results)
    avg_speedup_jit = sum(r['speedup_jit'] for r in results) / len(results)
    
    print(f"   Average JAX speedup: {avg_speedup_jax:.2f}x")
    print(f"   Average JAX JIT speedup: {avg_speedup_jit:.2f}x")
    
    # Overall assessment
    if avg_speedup_jax >= 10.0 or avg_speedup_jit >= 10.0:
        assessment = "🎯 EXCELLENT"
        status = "Phase 1.4: Performance Optimization - SUCCESS"
    elif avg_speedup_jax >= 5.0 or avg_speedup_jit >= 5.0:
        assessment = "📈 GOOD"
        status = "Phase 1.4: Performance Optimization - PROGRESS"
    elif avg_speedup_jax >= 2.0 or avg_speedup_jit >= 2.0:
        assessment = "✅ PARTIAL"
        status = "Phase 1.4: Performance Optimization - PARTIAL SUCCESS"
    else:
        assessment = "⚠️  NEEDS WORK"
        status = "Phase 1.4: Performance Optimization - OPTIMIZATION NEEDED"
    
    print(f"   Overall Assessment: {assessment}")
    print(f"   {status}")
    
    return True, results

if __name__ == "__main__":
    success, results = run_enhanced_benchmark()
    
    if success:
        print(f"\n✅ Enhanced benchmark completed successfully!")
        print(f"💡 Consider: Memory profiling, GPU utilization, or further JIT optimization")
    else:
        print(f"\n❌ Enhanced benchmark failed!")