#!/usr/bin/env python3
"""
Test script for Phase 1.3: DecisionAnalysis Integration.
Tests automatic JAX backend selection in DecisionAnalysis class.
"""

import sys
import os
import numpy as np

# Add the current directory to the path so we can import voiage
sys.path.insert(0, '/Users/doughnut/GitHub/voiage')

def test_decision_analysis_jax_integration():
    """Test automatic JAX backend selection in DecisionAnalysis."""
    print("🔧 Testing DecisionAnalysis JAX Integration (Phase 1.3)")
    print("=" * 65)
    
    try:
        import jax.numpy as jnp
        print("✅ JAX imported successfully")
    except ImportError as e:
        print(f"❌ JAX not available: {e}")
        return False
    
    from voiage.analysis import DecisionAnalysis
    from voiage.schema import ValueArray, ParameterSet
    from voiage.backends import JaxBackend, NumpyBackend
    
    # Test data setup
    np.random.seed(42)
    jax_seed = 42
    
    n_samples = 100
    n_strategies = 3
    n_parameters = 4
    
    print(f"\n📊 Test Configuration:")
    print(f"   Samples: {n_samples}")
    print(f"   Strategies: {n_strategies}")
    print(f"   Parameters: {n_parameters}")
    
    # ===============================
    # Test Automatic Backend Selection
    # ===============================
    print(f"\n🧪 Testing Automatic Backend Selection")
    print("-" * 45)
    
    # Test 1: NumPy arrays should select NumPy backend
    print(f"   Test 1: NumPy arrays → NumPy backend")
    numpy_nb = np.random.normal(1000, 200, (n_samples, n_strategies))
    numpy_params = {
        'param1': np.random.normal(0, 1, n_samples),
        'param2': np.random.normal(1, 2, n_samples)
    }
    
    da_numpy = DecisionAnalysis(numpy_nb, numpy_params, backend=None)
    backend_type = type(da_numpy.backend).__name__
    print(f"   Selected backend: {backend_type}")
    if backend_type == "NumpyBackend":
        print(f"   ✅ Correctly selected NumPy backend")
    else:
        print(f"   ❌ Expected NumPyBackend, got {backend_type}")
        return False
    
    # Test 2: JAX ValueArray should select JAX backend
    print(f"\n   Test 2: JAX ValueArray → JAX backend")
    jax_nb = jnp.array(numpy_nb, dtype=jnp.float32)
    value_array_jax = ValueArray.from_jax(jax_nb)
    
    da_jax_va = DecisionAnalysis(value_array_jax, numpy_params, backend=None)
    backend_type = type(da_jax_va.backend).__name__
    print(f"   Selected backend: {backend_type}")
    if backend_type == "JaxBackend":
        print(f"   ✅ Correctly selected JAX backend")
    else:
        print(f"   ❌ Expected JaxBackend, got {backend_type}")
        return False
    
    # Test 3: JAX ParameterSet should select JAX backend
    print(f"\n   Test 3: JAX ParameterSet → JAX backend")
    jax_params = {name: jnp.array(values, dtype=jnp.float32) for name, values in numpy_params.items()}
    param_set_jax = ParameterSet.from_jax(jax_params)
    
    da_jax_params = DecisionAnalysis(numpy_nb, param_set_jax, backend=None)
    backend_type = type(da_jax_params.backend).__name__
    print(f"   Selected backend: {backend_type}")
    if backend_type == "JaxBackend":
        print(f"   ✅ Correctly selected JAX backend")
    else:
        print(f"   ❌ Expected JaxBackend, got {backend_type}")
        return False
    
    # Test 4: Both JAX arrays should select JAX backend
    print(f"\n   Test 4: JAX ValueArray + JAX ParameterSet → JAX backend")
    da_both_jax = DecisionAnalysis(value_array_jax, param_set_jax, backend=None)
    backend_type = type(da_both_jax.backend).__name__
    print(f"   Selected backend: {backend_type}")
    if backend_type == "JaxBackend":
        print(f"   ✅ Correctly selected JAX backend")
    else:
        print(f"   ❌ Expected JaxBackend, got {backend_type}")
        return False
    
    # Test 5: Explicit backend override should work
    print(f"\n   Test 5: Explicit backend override")
    da_explicit = DecisionAnalysis(numpy_nb, numpy_params, backend="numpy")
    backend_type = type(da_explicit.backend).__name__
    print(f"   Selected backend: {backend_type}")
    if backend_type == "NumpyBackend":
        print(f"   ✅ Explicit backend override works")
    else:
        print(f"   ❌ Expected NumPyBackend, got {backend_type}")
        return False
    
    # ===============================
    # Test JAX Backend Functionality
    # ===============================
    print(f"\n🚀 Testing JAX Backend Functionality")
    print("-" * 40)
    
    # Test EVPI calculation with JAX backend
    print(f"   Testing EVPI calculation...")
    try:
        evpi_result = da_both_jax.evpi()
        print(f"   ✅ EVPI: {evpi_result:.4f}")
    except Exception as e:
        print(f"   ❌ EVPI calculation failed: {e}")
        return False
    
    # Test EVPPI calculation with JAX backend
    print(f"   Testing EVPPI calculation...")
    try:
        evppi_result = da_both_jax.evppi(["param1", "param2"])
        print(f"   ✅ EVPPI: {evppi_result:.4f}")
    except Exception as e:
        print(f"   ❌ EVPPI calculation failed: {e}")
        return False
    
    # Test ENBS calculation with JAX backend
    print(f"   Testing ENBS calculation...")
    try:
        enbs_result = da_both_jax.enbs(research_cost=50.0)
        print(f"   ✅ ENBS: {enbs_result:.4f}")
    except Exception as e:
        print(f"   ❌ ENBS calculation failed: {e}")
        return False
    
    # ===============================
    # Test Numerical Consistency
    # ===============================
    print(f"\n🔍 Testing Numerical Consistency")
    print("-" * 35)
    
    # Compare results between NumPy and JAX backends
    da_numpy_comparison = DecisionAnalysis(numpy_nb, numpy_params, backend="numpy")
    
    # EVPI consistency
    evpi_numpy = da_numpy_comparison.evpi()
    evpi_jax = da_both_jax.evpi()
    evpi_diff = abs(evpi_numpy - evpi_jax)
    print(f"   EVPI - NumPy: {evpi_numpy:.6f}, JAX: {evpi_jax:.6f}, Diff: {evpi_diff:.2e}")
    if evpi_diff < 1e-4:
        print(f"   ✅ EVPI numerical consistency verified")
    else:
        print(f"   ❌ EVPI numerical inconsistency detected")
        return False
    
    # EVPPI consistency
    evppi_numpy = da_numpy_comparison.evppi(["param1", "param2"])
    evppi_jax = da_both_jax.evppi(["param1", "param2"])
    evppi_diff = abs(evppi_numpy - evppi_jax)
    print(f"   EVPPI - NumPy: {evppi_numpy:.6f}, JAX: {evppi_jax:.6f}, Diff: {evppi_diff:.2e}")
    if evppi_diff < 1e-3:  # Slightly higher tolerance for EVPPI due to regression
        print(f"   ✅ EVPPI numerical consistency verified")
    else:
        print(f"   ❌ EVPPI numerical inconsistency detected")
        return False
    
    # ENBS consistency
    enbs_numpy = da_numpy_comparison.enbs(research_cost=50.0)
    enbs_jax = da_both_jax.enbs(research_cost=50.0)
    enbs_diff = abs(enbs_numpy - enbs_jax)
    print(f"   ENBS - NumPy: {enbs_numpy:.6f}, JAX: {enbs_jax:.6f}, Diff: {enbs_diff:.2e}")
    if enbs_diff < 1e-4:
        print(f"   ✅ ENBS numerical consistency verified")
    else:
        print(f"   ❌ ENBS numerical inconsistency detected")
        return False
    
    # ===============================
    # Test JAX JIT Compilation
    # ===============================
    print(f"\n⚡ Testing JAX JIT Compilation")
    print("-" * 30)
    
    # Create DecisionAnalysis with JIT enabled
    da_jit = DecisionAnalysis(value_array_jax, param_set_jax, backend=None, use_jit=True)
    backend_type = type(da_jit.backend).__name__
    print(f"   Backend: {backend_type}, JIT enabled: {da_jit.use_jit}")
    
    if backend_type == "JaxBackend" and da_jit.use_jit:
        print(f"   ✅ JIT configuration successful")
    else:
        print(f"   ❌ JIT configuration failed")
        return False
    
    # Test JIT compilation performance (first call will compile)
    import time
    
    # Warm up and measure JIT compilation
    print(f"   Testing JIT compilation...")
    start_time = time.time()
    evpi_jit_1 = da_jit.evpi()
    compile_time = time.time() - start_time
    print(f"   First call (includes compilation): {compile_time:.4f}s")
    
    # Subsequent calls should be faster
    start_time = time.time()
    evpi_jit_2 = da_jit.evpi()
    run_time = time.time() - start_time
    print(f"   Second call (cached): {run_time:.4f}s")
    
    print(f"   JIT result: {evpi_jit_1:.6f}")
    print(f"   ✅ JIT compilation test completed")
    
    print(f"\n🎉 All DecisionAnalysis JAX integration tests passed!")
    print(f"   ✅ Automatic backend selection: NumPy and JAX")
    print(f"   ✅ JAX backend functionality: EVPI, EVPPI, ENBS")
    print(f"   ✅ Numerical consistency verified")
    print(f"   ✅ JAX JIT compilation enabled")
    print(f"   ✅ Explicit backend override supported")
    
    return True


if __name__ == "__main__":
    success = test_decision_analysis_jax_integration()
    if success:
        print(f"\n🎯 Phase 1.3 DecisionAnalysis Integration: ✅ COMPLETE")
    else:
        print(f"\n❌ Phase 1.3 DecisionAnalysis Integration: ❌ FAILED")
        sys.exit(1)