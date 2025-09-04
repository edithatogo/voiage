"""Fluent API for Value of Information analysis."""

from typing import Any, Dict, Optional, Union, Generator
import numpy as np

from voiage.analysis import DecisionAnalysis
from voiage.schema import ParameterSet, ValueArray


class FluentDecisionAnalysis(DecisionAnalysis):
    """A class to represent a decision analysis problem with fluent API support."""
    
    def __init__(
        self,
        nb_array: Union[np.ndarray, ValueArray],
        parameter_samples: Optional[
            Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]
        ] = None,
        backend: Optional[str] = None,
        use_jit: bool = False,
        streaming_window_size: Optional[int] = None,
        enable_caching: bool = False,
    ):
        super().__init__(
            nb_array=nb_array,
            parameter_samples=parameter_samples,
            backend=backend,
            use_jit=use_jit,
            streaming_window_size=streaming_window_size,
            enable_caching=enable_caching,
        )
        # Store results for method chaining
        self._last_evpi_result = None
        self._last_evppi_result = None
        self._last_evsi_result = None
    
    def with_parameters(self, parameter_samples: Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]) -> "FluentDecisionAnalysis":
        """
        Set parameter samples for the analysis.
        
        Args:
            parameter_samples: Parameter samples for EVPPI calculation
            
        Returns:
            FluentDecisionAnalysis: Self for method chaining
        """
        if isinstance(parameter_samples, ParameterSet):
            self.parameter_samples = parameter_samples
        elif isinstance(parameter_samples, (dict, np.ndarray)):
            self.parameter_samples = ParameterSet.from_numpy_or_dict(parameter_samples)
        else:
            raise TypeError(f"`parameter_samples` must be a NumPy array, ParameterSet, or Dict. Got {type(parameter_samples)}.")
        return self
    
    def with_backend(self, backend: str) -> "FluentDecisionAnalysis":
        """
        Set the computational backend.
        
        Args:
            backend: Backend name ('numpy', 'jax', etc.)
            
        Returns:
            FluentDecisionAnalysis: Self for method chaining
        """
        from voiage.backends import get_backend
        self.backend = get_backend(backend)
        return self
    
    def with_jit(self, use_jit: bool = True) -> "FluentDecisionAnalysis":
        """
        Enable or disable JIT compilation.
        
        Args:
            use_jit: Whether to use JIT compilation
            
        Returns:
            FluentDecisionAnalysis: Self for method chaining
        """
        self.use_jit = use_jit
        return self
    
    def with_streaming(self, window_size: int) -> "FluentDecisionAnalysis":
        """
        Enable streaming data support with specified window size.
        
        Args:
            window_size: Size of the streaming window
            
        Returns:
            FluentDecisionAnalysis: Self for method chaining
        """
        self.streaming_window_size = window_size
        self._initialize_streaming_buffers()
        return self
    
    def with_caching(self, enable: bool = True) -> "FluentDecisionAnalysis":
        """
        Enable or disable caching.
        
        Args:
            enable: Whether to enable caching
            
        Returns:
            FluentDecisionAnalysis: Self for method chaining
        """
        self.enable_caching = enable
        self._cache = {} if enable else None
        return self
    
    def add_data(self, new_nb_data: Union[np.ndarray, ValueArray], 
                 new_parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None) -> "FluentDecisionAnalysis":
        """
        Add new data to the analysis (for streaming support).
        
        Args:
            new_nb_data: New net benefit data to add
            new_parameter_samples: New parameter samples corresponding to the net benefit data
            
        Returns:
            FluentDecisionAnalysis: Self for method chaining
        """
        self.update_with_new_data(new_nb_data, new_parameter_samples)
        return self
    
    def calculate_evpi(self, 
                       population: Optional[float] = None,
                       time_horizon: Optional[float] = None,
                       discount_rate: Optional[float] = None,
                       chunk_size: Optional[int] = None) -> "FluentDecisionAnalysis":
        """
        Calculate the Expected Value of Perfect Information (EVPI).
        
        Args:
            population: The relevant population size
            time_horizon: The relevant time horizon in years
            discount_rate: The annual discount rate
            chunk_size: Size of chunks for incremental computation
            
        Returns:
            FluentDecisionAnalysis: Self for method chaining
        """
        self._last_evpi_result = self.evpi(
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            chunk_size=chunk_size
        )
        return self
    
    def calculate_evppi(self,
                        population: Optional[float] = None,
                        time_horizon: Optional[float] = None,
                        discount_rate: Optional[float] = None,
                        n_regression_samples: Optional[int] = None,
                        regression_model: Optional[Any] = None,
                        chunk_size: Optional[int] = None) -> "FluentDecisionAnalysis":
        """
        Calculate the Expected Value of Partial Perfect Information (EVPPI).
        
        Args:
            population: Population size for scaling
            time_horizon: Time horizon for scaling
            discount_rate: Discount rate for scaling
            n_regression_samples: Number of samples to use for fitting the regression model
            regression_model: An unfitted scikit-learn compatible regression model
            chunk_size: Size of chunks for incremental computation
            
        Returns:
            FluentDecisionAnalysis: Self for method chaining
        """
        self._last_evppi_result = self.evppi(
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            n_regression_samples=n_regression_samples,
            regression_model=regression_model,
            chunk_size=chunk_size
        )
        return self
    
    def get_evpi_result(self) -> Optional[float]:
        """
        Get the last calculated EVPI result.
        
        Returns:
            Optional[float]: Last EVPI result or None if not calculated
        """
        return self._last_evpi_result
    
    def get_evppi_result(self) -> Optional[float]:
        """
        Get the last calculated EVPPI result.
        
        Returns:
            Optional[float]: Last EVPPI result or None if not calculated
        """
        return self._last_evppi_result
    
    def get_results(self) -> Dict[str, Optional[float]]:
        """
        Get all calculated results.
        
        Returns:
            Dict[str, Optional[float]]: Dictionary of results
        """
        return {
            "evpi": self._last_evpi_result,
            "evppi": self._last_evppi_result
        }
    
    # Context manager support
    def __enter__(self) -> "FluentDecisionAnalysis":
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        pass


# Factory function for creating fluent analysis objects
def create_analysis(nb_array: Union[np.ndarray, ValueArray],
                   parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None,
                   **kwargs) -> FluentDecisionAnalysis:
    """
    Create a FluentDecisionAnalysis object.
    
    Args:
        nb_array: Net benefit array
        parameter_samples: Parameter samples for EVPPI calculation
        **kwargs: Additional arguments for DecisionAnalysis constructor
        
    Returns:
        FluentDecisionAnalysis: New fluent analysis object
    """
    return FluentDecisionAnalysis(
        nb_array=nb_array,
        parameter_samples=parameter_samples,
        **kwargs
    )


# Example usage:
# 
# # Fluent API usage
# results = (create_analysis(net_benefits)
#            .with_parameters(parameters)
#            .with_backend("jax")
#            .with_jit()
#            .with_caching()
#            .calculate_evpi(population=100000, time_horizon=10, discount_rate=0.03)
#            .calculate_evppi()
#            .get_results())
# 
# # Or step by step
# analysis = create_analysis(net_benefits)
# analysis = analysis.with_parameters(parameters)
# analysis = analysis.with_backend("jax").with_jit().with_caching()
# analysis = analysis.calculate_evpi(population=100000, time_horizon=10, discount_rate=0.03)
# analysis = analysis.calculate_evppi()
# results = analysis.get_results()