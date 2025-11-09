
# Performance Profiling and Optimization Tools
import time
import functools
import numpy as np
from typing import Dict, List

class JaxPerformanceProfiler:
    """Profile and optimize JAX computations."""
    
    def __init__(self):
        self.profiles = {}
        self.timings = {}
        
    def profile_function(self, func):
        """Profile function execution time."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            func_name = func.__name__
            if func_name not in self.timings:
                self.timings[func_name] = []
            self.timings[func_name].append(end_time - start_time)
            
            return result
        return wrapper
    
    def compare_implementations(self, numpy_func, jax_func, test_data, n_runs=10):
        """Compare NumPy vs JAX implementations."""
        results = {
            'numpy_times': [],
            'jax_times': [],
            'speedups': []
        }
        
        # Warm up JAX
        jax_func(*test_data)
        
        for i in range(n_runs):
            # Time NumPy
            start = time.time()
            _ = numpy_func(*test_data)
            numpy_time = time.time() - start
            results['numpy_times'].append(numpy_time)
            
            # Time JAX
            start = time.time()
            _ = jax_func(*test_data)
            jax_time = time.time() - start
            results['jax_times'].append(jax_time)
            
            # Calculate speedup
            speedup = numpy_time / jax_time if jax_time > 0 else 0
            results['speedups'].append(speedup)
            
        return results
    
    def memory_usage_analysis(self, func, *args, **kwargs):
        """Analyze memory usage of a function."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run function
        result = func(*args, **kwargs)
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_increase_mb': memory_after - memory_before,
            'result': result
        }
    
    def get_performance_report(self):
        """Generate performance report."""
        report = {}
        for func_name, times in self.timings.items():
            report[func_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'calls': len(times)
            }
        return report
