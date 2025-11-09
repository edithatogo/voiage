
# Integration of Advanced Features with Main JAX Backend
# This would modify voiage/backends.py to include the advanced features

import sys
import os
sys.path.append('/Users/doughnut/GitHub/voiage/voiage/backends')

# Import advanced features
from .advanced_jax_regression import JaxAdvancedRegression
from .gpu_acceleration import GpuAcceleration
from .performance_profiler import JaxPerformanceProfiler

# Add to JaxBackend class (would be added to voiage/backends.py)
try:
    from .base import JaxBackend  # Or wherever JaxBackend is defined
except ImportError:
    # Create a base class for dev/testing purposes only
    class JaxBackend:
        pass

class JaxAdvancedBackend(JaxBackend):
    """Extended JAX backend with advanced features."""
    
    def __init__(self):
        super().__init__()
        self.regression_model = JaxAdvancedRegression()
        self.gpu_utils = GpuAcceleration()
        self.profiler = JaxPerformanceProfiler()
        
    def evppi_advanced(self, net_benefit_array, parameter_samples, parameters_of_interest, 
                      method="polynomial", degree=2, **kwargs):
        """Advanced EVPPI calculation with enhanced regression."""
        return self.evppi_advanced_core(net_benefit_array, parameter_samples, 
                                       parameters_of_interest, method, degree, **kwargs)
    
    def get_gpu_info(self):
        """Get GPU information for optimization."""
        return self.gpu_utils.get_memory_info()
    
    def profile_evppi(self, net_benefit_array, parameter_samples, parameters_of_interest):
        """Profile EVPPI calculation performance."""
        return self.profiler.memory_usage_analysis(
            self.evppi_advanced, net_benefit_array, parameter_samples, parameters_of_interest
        )
