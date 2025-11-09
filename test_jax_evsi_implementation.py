#!/usr/bin/env python3
"""
Test script to verify JAX EVSI implementation is working correctly.
"""

import numpy as np
import jax.numpy as jnp
import jax
import sys
import os

# Add current directory to path
sys.path.insert(0, '/Users/doughnut/GitHub/voiage')

# Test JAX backend availability
try:
    # Import directly from the backends.py file
    import importlib.util
    spec = importlib.util.spec_from_file_location("backends_module", "/Users/doughnut/GitHub/voiage/voiage/backends.py")
    backends_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backends_module)
    
    get_backend = backends_module.get_backend
    JAX_AVAILABLE = backends_module.JAX_AVAILABLE
    print(f"JAX Available (from backends.py): {JAX_AVAILABLE}")
    
    if JAX_AVAILABLE:
        backend = get_backend('jax')
        print(f"JAX Backend: {backend}")
        print(f"Backend type: {type(backend)}")
        
        # Test simple EVPI calculation
        nb_array = jnp.array([[100, 200, 150], [110, 190, 160], [105, 195, 155]])
        evpi_result = backend.evpi(nb_array)
        print(f"EVPI Test Result: {evpi_result}")
        
        # Test if EVSI method is available
        if hasattr(backend, 'evsi'):
            print("✓ EVSI method is available")
            print("EVSI method implementation:")
            import inspect
            print(inspect.getsource(backend.evsi))
        else:
            print("✗ EVSI method not found")
            
        # Test if JAX helper methods are available
        helper_methods = ['_evsi_two_loop_jax', '_evsi_regression_jax', 
                         '_simulate_trial_data_jax', '_bayesian_update_jax']
        for method in helper_methods:
            if hasattr(backend, method):
                print(f"✓ {method} is available")
            else:
                print(f"✗ {method} not found")
                
    else:
        print("JAX is not available")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()