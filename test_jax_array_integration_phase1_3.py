#!/usr/bin/env python3
"""
Test script for Phase 1.3: JAX Array Integration.
Tests the new JAX array support in ValueArray and ParameterSet.
"""

import sys
import os
import numpy as np

# Add the current directory to the path so we can import voiage
sys.path.insert(0, '/Users/doughnut/GitHub/voiage')

def test_jax_array_integration():
    """Test JAX array integration in ValueArray and ParameterSet."""
    print("🔧 Testing JAX Array Integration (Phase 1.3)")
    print("=" * 60)
    
    try:
        import jax.numpy as jnp
        print("✅ JAX imported successfully")
    except ImportError as e:
        print(f"❌ JAX not available: {e}")
        return False
    
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
    # Test ValueArray JAX Integration
    # ===============================
    print(f"\n🧪 Testing ValueArray JAX Integration")
    print("-" * 40)
    
    from voiage.schema import ValueArray
    
    # Create test data
    numpy_nb = np.random.normal(1000, 200, (n_samples, n_strategies))
    jax_nb = jnp.array(numpy_nb, dtype=jnp.float32)
    
    print(f"   Original NumPy shape: {numpy_nb.shape}")
    print(f"   JAX array shape: {jax_nb.shape}")
    print(f"   JAX array dtype: {jax_nb.dtype}")
    
    # Test from_numpy with JAX array
    print(f"\n   Testing ValueArray.from_numpy with JAX array...")
    try:
        value_array_from_jax = ValueArray.from_numpy(jax_nb)
        print(f"   ✅ from_numpy(JAX) successful")
        print(f"   Values shape: {value_array_from_jax.values.shape}")
        print(f"   Values type: {type(value_array_from_jax.values)}")
    except Exception as e:
        print(f"   ❌ from_numpy(JAX) failed: {e}")
        return False
    
    # Test from_jax
    print(f"\n   Testing ValueArray.from_jax...")
    try:
        value_array_direct = ValueArray.from_jax(jax_nb)
        print(f"   ✅ from_jax successful")
        print(f"   Values shape: {value_array_direct.values.shape}")
        print(f"   Values type: {type(value_array_direct.values)}")
    except Exception as e:
        print(f"   ❌ from_jax failed: {e}")
        return False
    
    # Test jax_values property
    print(f"\n   Testing jax_values property...")
    try:
        jax_values = value_array_from_jax.jax_values
        if jax_values is not None:
            print(f"   ✅ jax_values successful")
            print(f"   JAX values shape: {jax_values.shape}")
            print(f"   JAX values dtype: {jax_values.dtype}")
            print(f"   Is JAX array: {hasattr(jax_values, 'device')}")
        else:
            print(f"   ⚠️  jax_values returned None")
    except Exception as e:
        print(f"   ❌ jax_values failed: {e}")
        return False
    
    # Verify numerical consistency
    print(f"\n   Verifying numerical consistency...")
    numpy_result = value_array_from_jax.values
    jax_result = value_array_from_jax.jax_values
    
    if jax_result is not None:
        max_diff = np.max(np.abs(numpy_result - np.asarray(jax_result)))
        print(f"   Max difference: {max_diff:.2e}")
        if max_diff < 1e-6:
            print(f"   ✅ Numerical consistency verified")
        else:
            print(f"   ❌ Numerical inconsistency detected")
            return False
    
    # ===============================
    # Test ParameterSet JAX Integration
    # ===============================
    print(f"\n🧪 Testing ParameterSet JAX Integration")
    print("-" * 40)
    
    from voiage.schema import ParameterSet
    
    # Create test parameter data
    numpy_params = {
        'param1': np.random.normal(0, 1, n_samples),
        'param2': np.random.normal(1, 2, n_samples),
        'param3': np.random.normal(-0.5, 0.5, n_samples),
        'param4': np.random.normal(2, 0.8, n_samples)
    }
    
    jax_params = {name: jnp.array(values, dtype=jnp.float32) for name, values in numpy_params.items()}
    
    print(f"   Parameter names: {list(numpy_params.keys())}")
    for name in numpy_params:
        print(f"   {name} - NumPy: {numpy_params[name].shape}, JAX: {jax_params[name].shape}")
    
    # Test from_numpy_or_dict with JAX arrays
    print(f"\n   Testing ParameterSet.from_numpy_or_dict with JAX arrays...")
    try:
        param_set_from_jax = ParameterSet.from_numpy_or_dict(jax_params)
        print(f"   ✅ from_numpy_or_dict(JAX) successful")
        print(f"   Parameter names: {param_set_from_jax.parameter_names}")
        print(f"   Number of samples: {param_set_from_jax.n_samples}")
    except Exception as e:
        print(f"   ❌ from_numpy_or_dict(JAX) failed: {e}")
        return False
    
    # Test from_jax with dictionary
    print(f"\n   Testing ParameterSet.from_jax with dictionary...")
    try:
        param_set_direct = ParameterSet.from_jax(jax_params)
        print(f"   ✅ from_jax successful")
        print(f"   Parameter names: {param_set_direct.parameter_names}")
        print(f"   Number of samples: {param_set_direct.n_samples}")
    except Exception as e:
        print(f"   ❌ from_jax failed: {e}")
        return False
    
    # Test from_jax with JAX array
    jax_param_array = jnp.column_stack([jax_params[name] for name in ['param1', 'param2', 'param3', 'param4']])
    print(f"\n   Testing ParameterSet.from_jax with JAX array...")
    try:
        param_set_array = ParameterSet.from_jax(jax_param_array)
        print(f"   ✅ from_jax(array) successful")
        print(f"   Parameter names: {param_set_array.parameter_names}")
        print(f"   Number of samples: {param_set_array.n_samples}")
    except Exception as e:
        print(f"   ❌ from_jax(array) failed: {e}")
        return False
    
    # Test jax_parameters property
    print(f"\n   Testing jax_parameters property...")
    try:
        jax_parameters = param_set_from_jax.jax_parameters
        if jax_parameters is not None:
            print(f"   ✅ jax_parameters successful")
            for name, values in jax_parameters.items():
                print(f"   {name} - JAX shape: {values.shape}, dtype: {values.dtype}")
                print(f"   {name} - Is JAX array: {hasattr(values, 'device')}")
        else:
            print(f"   ⚠️  jax_parameters returned None")
    except Exception as e:
        print(f"   ❌ jax_parameters failed: {e}")
        return False
    
    # Verify numerical consistency for parameters
    print(f"\n   Verifying numerical consistency for parameters...")
    numpy_params_result = param_set_from_jax.parameters
    jax_params_result = param_set_from_jax.jax_parameters
    
    if jax_params_result is not None:
        for name in numpy_params_result:
            max_diff = np.max(np.abs(numpy_params_result[name] - np.asarray(jax_params_result[name])))
            print(f"   {name} - Max difference: {max_diff:.2e}")
            if max_diff > 1e-6:
                print(f"   ❌ Numerical inconsistency detected in {name}")
                return False
        print(f"   ✅ All parameter numerical consistency verified")
    
    # ===============================
    # Test Integration with JAX Backend
    # ===============================
    print(f"\n🔗 Testing JAX Backend Integration")
    print("-" * 40)
    
    from voiage.backends import JaxBackend
    
    backend = JaxBackend()
    print(f"   ✅ JAX backend initialized")
    
    # Test EVPI with JAX ValueArray
    print(f"\n   Testing EVPI with JAX ValueArray...")
    try:
        evpi_result = backend.evpi(value_array_from_jax.jax_values)
        print(f"   ✅ EVPI calculation successful: {evpi_result:.4f}")
    except Exception as e:
        print(f"   ❌ EVPI calculation failed: {e}")
        return False
    
    # Test EVPPI with JAX ParameterSet
    print(f"\n   Testing EVPPI with JAX ParameterSet...")
    try:
        parameters_of_interest = ['param1', 'param2']
        evppi_result = backend.evppi(
            value_array_from_jax.jax_values, 
            {name: jax_params_result[name] for name in parameters_of_interest}, 
            parameters_of_interest
        )
        print(f"   ✅ EVPPI calculation successful: {evppi_result:.4f}")
    except Exception as e:
        print(f"   ❌ EVPPI calculation failed: {e}")
        return False
    
    print(f"\n🎉 All JAX array integration tests passed!")
    print(f"   ✅ ValueArray.from_numpy: JAX array support")
    print(f"   ✅ ValueArray.from_jax: Direct JAX array creation")
    print(f"   ✅ ValueArray.jax_values: JAX array property")
    print(f"   ✅ ParameterSet.from_numpy_or_dict: JAX array support")
    print(f"   ✅ ParameterSet.from_jax: Direct JAX array creation")
    print(f"   ✅ ParameterSet.jax_parameters: JAX array property")
    print(f"   ✅ JAX Backend integration: EVPI and EVPPI working")
    
    return True


def performance_test_jax_vs_numpy():
    """Compare performance between NumPy and JAX array operations."""
    print(f"\n🚀 Performance Test: JAX vs NumPy Arrays")
    print("=" * 50)
    
    try:
        import jax.numpy as jnp
        import time
        from voiage.schema import ValueArray, ParameterSet
    except ImportError as e:
        print(f"❌ Cannot import required modules: {e}")
        return False
    
    # Setup test data
    np.random.seed(42)
    n_samples = 1000
    n_strategies = 5
    
    # Create test data
    numpy_nb = np.random.normal(1000, 200, (n_samples, n_strategies))
    jax_nb = jnp.array(numpy_nb, dtype=jnp.float32)
    
    # Test ValueArray creation performance
    print(f"📊 ValueArray Creation Performance:")
    
    # NumPy
    start_time = time.time()
    for _ in range(100):
        va_numpy = ValueArray.from_numpy(numpy_nb)
    numpy_time = time.time() - start_time
    print(f"   NumPy: {numpy_time:.4f}s (100 iterations)")
    
    # JAX
    start_time = time.time()
    for _ in range(100):
        va_jax = ValueArray.from_numpy(jax_nb)
    jax_time = time.time() - start_time
    print(f"   JAX: {jax_time:.4f}s (100 iterations)")
    
    # JAX direct
    start_time = time.time()
    for _ in range(100):
        va_jax_direct = ValueArray.from_jax(jax_nb)
    jax_direct_time = time.time() - start_time
    print(f"   JAX direct: {jax_direct_time:.4f}s (100 iterations)")
    
    # Test jax_values property performance
    print(f"\n📊 JAX Values Property Performance:")
    
    va_test = ValueArray.from_numpy(jax_nb)
    
    # NumPy values
    start_time = time.time()
    for _ in range(1000):
        _ = va_test.values
    numpy_values_time = time.time() - start_time
    print(f"   values: {numpy_values_time:.4f}s (1000 iterations)")
    
    # JAX values
    start_time = time.time()
    for _ in range(1000):
        _ = va_test.jax_values
    jax_values_time = time.time() - start_time
    print(f"   jax_values: {jax_values_time:.4f}s (1000 iterations)")
    
    print(f"\n📈 Performance Summary:")
    print(f"   ValueArray creation (JAX vs NumPy): {jax_time/numpy_time:.2f}x")
    print(f"   JAX direct vs JAX via NumPy: {jax_direct_time/jax_time:.2f}x")
    print(f"   JAX values property overhead: {jax_values_time/numpy_values_time:.2f}x")
    
    return True


if __name__ == "__main__":
    success = test_jax_array_integration()
    if success:
        performance_test_jax_vs_numpy()
        print(f"\n🎯 Phase 1.3 JAX Array Integration: ✅ COMPLETE")
    else:
        print(f"\n❌ Phase 1.3 JAX Array Integration: ❌ FAILED")
        sys.exit(1)