#!/usr/bin/env python3
"""
Phase 1.3 Performance Optimization Benchmark
Tests JAX backend performance vs NumPy baseline for >10x speedup target
"""

import sys
import os
import time
import numpy as np

# Add the current directory to the path
sys.path.insert(0, '/Users/doughnut/GitHub/voiage')

def benchmark_phase1_3_performance():
    """Comprehensive performance benchmark for Phase 1.3 JAX optimization."""
    print("🚀 Phase 1.3 Performance Optimization Benchmark")
    print("=" * 65)
    
    try:
        import jax.numpy as jnp
        print("✅ JAX available for performance testing")
    except ImportError as e:
        print(f"❌ JAX not available: {e}")
        return False
    
    from voiage.analysis import DecisionAnalysis
    from voiage.schema import ValueArray, ParameterSet
    
    # Performance test configurations
    configs = [
        {"name": "Small Dataset", "n_samples": 1000, "n_strategies": 5, "n_params": 3},
        {"name": "Medium Dataset", "n_samples": 10000, "n_strategies": 10, "n_params": 5},
        {"name": "Large Dataset", "n_samples": 100000, "n_strategies": 20, "n_params": 8},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing {config['name']}:")
        print(f"   Samples: {config['n_samples']:,}")
        print(f"   Strategies: {config['n_strategies']}")
        print(f"   Parameters: {config['n_params']}")
        
        # Generate test data
        np.random.seed(42)
        
        # NumPy data
        numpy_nb = np.random.normal(1000, 200, (config['n_samples'], config['n_strategies']))
        numpy_params = {
            f'param_{i}': np.random.normal(i, 0.5, config['n_samples']) 
            for i in range(config['n_params'])
        }
        
        # JAX data
        jax_nb = jnp.array(numpy_nb, dtype=jnp.float32)
        jax_params = {
            name: jnp.array(values, dtype=jnp.float32) 
            for name, values in numpy_params.items()
        }
        
        # Create DecisionAnalysis instances
        da_numpy = DecisionAnalysis(numpy_nb, numpy_params, backend="numpy")
        da_jax = DecisionAnalysis(ValueArray.from_jax(jax_nb), 
                                ParameterSet.from_jax(jax_params), 
                                backend="jax")
        
        # Test EVPI performance
        print(f"\n   🧮 EVPI Performance Test:")
        
        # NumPy timing
        start_time = time.time()
        evpi_numpy = da_numpy.evpi()
        numpy_evpi_time = time.time() - start_time
        print(f"      NumPy: {numpy_evpi_time:.4f}s")
        
        # JAX timing (first run - may include compilation)
        start_time = time.time()
        evpi_jax_1 = da_jax.evpi()
        jax_evpi_time_1 = time.time() - start_time
        print(f"      JAX (1st): {jax_evpi_time_1:.4f}s")
        
        # JAX timing (subsequent run - compiled)
        start_time = time.time()
        evpi_jax_2 = da_jax.evpi()
        jax_evpi_time_2 = time.time() - start_time
        print(f"      JAX (cached): {jax_evpi_time_2:.4f}s")
        
        # JAX with JIT
        da_jax_jit = DecisionAnalysis(ValueArray.from_jax(jax_nb), 
                                    ParameterSet.from_jax(jax_params), 
                                    backend="jax", use_jit=True)
        
        start_time = time.time()
        evpi_jax_jit_1 = da_jax_jit.evpi()
        jax_jit_time_1 = time.time() - start_time
        
        start_time = time.time()
        evpi_jax_jit_2 = da_jax_jit.evpi()
        jax_jit_time_2 = time.time() - start_time
        
        print(f"      JAX JIT (1st): {jax_jit_time_1:.4f}s")
        print(f"      JAX JIT (cached): {jax_jit_time_2:.4f}s")
        
        # Speedup calculations
        speedup_1 = numpy_evpi_time / jax_evpi_time_1 if jax_evpi_time_1 > 0 else 0
        speedup_2 = numpy_evpi_time / jax_evpi_time_2 if jax_evpi_time_2 > 0 else 0
        speedup_jit_1 = numpy_evpi_time / jax_jit_time_1 if jax_jit_time_1 > 0 else 0
        speedup_jit_2 = numpy_evpi_time / jax_jit_time_2 if jax_jit_time_2 > 0 else 0
        
        print(f"   📈 Speedup Analysis:")
        print(f"      JAX (first run): {speedup_1:.2f}x")
        print(f"      JAX (cached): {speedup_2:.2f}x")
        print(f"      JAX JIT (first run): {speedup_jit_1:.2f}x")
        print(f"      JAX JIT (cached): {speedup_jit_2:.2f}x")
        
        # Verify numerical consistency
        evpi_diff = abs(evpi_numpy - evpi_jax_1)
        print(f"   🔍 Numerical Check:")
        print(f"      Difference: {evpi_diff:.2e} (should be < 1e-3)")
        
        results.append({
            'config': config['name'],
            'samples': config['n_samples'],
            'numpy_time': numpy_evpi_time,
            'jax_time_1': jax_evpi_time_1,
            'jax_time_2': jax_evpi_time_2,
            'jax_jit_time_1': jax_jit_time_1,
            'jax_jit_time_2': jax_jit_time_2,
            'speedup_1': speedup_1,
            'speedup_2': speedup_2,
            'speedup_jit_1': speedup_jit_1,
            'speedup_jit_2': speedup_jit_2,
            'evpi_numpy': evpi_numpy,
            'evpi_jax': evpi_jax_1,
            'evpi_diff': evpi_diff
        })
        
        if evpi_diff > 1e-3:
            print(f"   ⚠️  Warning: Numerical difference may be too large")
    
    # Summary
    print(f"\n🎯 Phase 1.3 Performance Summary")
    print("=" * 50)
    
    target_speedup = 10.0  # 10x speedup target
    met_target = False
    
    for result in results:
        print(f"\n{result['config']} ({result['samples']:,} samples):")
        print(f"  JAX (cached): {result['speedup_2']:.2f}x speedup")
        print(f"  JAX JIT (cached): {result['speedup_jit_2']:.2f}x speedup")
        
        if result['speedup_2'] >= target_speedup or result['speedup_jit_2'] >= target_speedup:
            print(f"  ✅ Meets 10x target!")
            met_target = True
        else:
            print(f"  ❌ Below 10x target")
    
    print(f"\n🏆 Overall Results:")
    if met_target:
        print(f"  ✅ Phase 1.3 Performance Target: ACHIEVED!")
        print(f"  🎉 JAX optimization provides significant performance benefits")
    else:
        print(f"  📈 Phase 1.3 Performance Target: Partial")
        print(f"  💡 Consider: larger datasets, GPU usage, or further optimization")
    
    print(f"\n📋 Performance Insights:")
    avg_speedup_2 = np.mean([r['speedup_2'] for r in results])
    avg_speedup_jit_2 = np.mean([r['speedup_jit_2'] for r in results])
    print(f"  Average JAX speedup (cached): {avg_speedup_2:.2f}x")
    print(f"  Average JAX JIT speedup (cached): {avg_speedup_jit_2:.2f}x")
    
    # Data size impact
    print(f"\n📈 Scaling Analysis:")
    for result in results:
        efficiency = result['speedup_2'] / (result['samples'] / 1000)  # speedup per 1000 samples
        print(f"  {result['config']}: {efficiency:.2f} speedup efficiency")
    
    return met_target, results


if __name__ == "__main__":
    success, results = benchmark_phase1_3_performance()
    if success:
        print(f"\n🎯 Phase 1.3: Performance Optimization - ✅ TARGET ACHIEVED")
    else:
        print(f"\n🎯 Phase 1.3: Performance Optimization - 📈 PARTIAL SUCCESS")
        print(f"💡 Consider advancing to Phase 1.3: Advanced JAX Features")
    
    sys.exit(0 if success else 1)