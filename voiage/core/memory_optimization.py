"""Memory optimization utilities for Value of Information analysis."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import gc
import psutil
import warnings

from voiage.schema import ValueArray, ParameterSet


class MemoryOptimizer:
    """Utility class for memory optimization in VOI analyses."""
    
    def __init__(self, memory_limit_mb: Optional[float] = None):
        """
        Initialize the memory optimizer.
        
        Args:
            memory_limit_mb: Memory limit in MB. If None, uses 80% of available memory.
        """
        if memory_limit_mb is None:
            # Use 80% of available memory as default limit
            memory_limit_mb = psutil.virtual_memory().total * 0.8 / (1024 * 1024)
        
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.current_memory_usage = 0.0
        
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in bytes.
        
        Returns:
            float: Current memory usage in bytes
        """
        return psutil.virtual_memory().used
    
    def get_available_memory(self) -> float:
        """
        Get available memory in bytes.
        
        Returns:
            float: Available memory in bytes
        """
        return self.memory_limit_bytes - self.get_memory_usage()
    
    def is_memory_available(self, required_bytes: float) -> bool:
        """
        Check if required memory is available.
        
        Args:
            required_bytes: Required memory in bytes
            
        Returns:
            bool: True if memory is available, False otherwise
        """
        return self.get_available_memory() >= required_bytes
    
    def optimize_array_dtype(self, arr: np.ndarray) -> np.ndarray:
        """
        Optimize array data type to reduce memory usage.
        
        Args:
            arr: Input array
            
        Returns:
            np.ndarray: Array with optimized data type
        """
        # If array is already float32, return as is
        if arr.dtype == np.float32:
            return arr
            
        # Check if we can safely convert to float32
        if arr.dtype == np.float64:
            # Check if values can be represented in float32 without significant loss
            float32_arr = arr.astype(np.float32)
            if np.allclose(arr, float32_arr, rtol=1e-6):
                return float32_arr
                
        # For integer types, check if we can use smaller types
        if np.issubdtype(arr.dtype, np.integer):
            # Try to find the smallest integer type that can hold all values
            if np.min(arr) >= np.iinfo(np.int8).min and np.max(arr) <= np.iinfo(np.int8).max:
                return arr.astype(np.int8)
            elif np.min(arr) >= np.iinfo(np.int16).min and np.max(arr) <= np.iinfo(np.int16).max:
                return arr.astype(np.int16)
            elif np.min(arr) >= np.iinfo(np.int32).min and np.max(arr) <= np.iinfo(np.int32).max:
                return arr.astype(np.int32)
                
        # If no optimization is possible, return original array
        return arr
    
    def chunk_large_array(self, arr: np.ndarray, max_chunk_size: int) -> List[np.ndarray]:
        """
        Split a large array into chunks to reduce memory usage.
        
        Args:
            arr: Input array
            max_chunk_size: Maximum size of each chunk in number of rows
            
        Returns:
            List[np.ndarray]: List of array chunks
        """
        if arr.shape[0] <= max_chunk_size:
            return [arr]
            
        chunks = []
        for i in range(0, arr.shape[0], max_chunk_size):
            end_idx = min(i + max_chunk_size, arr.shape[0])
            chunks.append(arr[i:end_idx])
            
        return chunks
    
    def estimate_memory_usage(self, obj: Any) -> float:
        """
        Estimate memory usage of an object in bytes.
        
        Args:
            obj: Object to estimate memory usage for
            
        Returns:
            float: Estimated memory usage in bytes
        """
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (ValueArray, ParameterSet)):
            if hasattr(obj, 'dataset') and hasattr(obj.dataset, 'nbytes'):
                return obj.dataset.nbytes
        elif isinstance(obj, dict):
            total = 0
            for key, value in obj.items():
                total += self.estimate_memory_usage(key)
                total += self.estimate_memory_usage(value)
            return total
        elif isinstance(obj, (list, tuple)):
            total = 0
            for item in obj:
                total += self.estimate_memory_usage(item)
            return total
            
        # For other objects, use a rough estimate
        return len(str(obj)) if hasattr(obj, '__str__') else 0
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
        
    def monitor_memory_usage(self, warning_threshold: float = 0.8) -> None:
        """
        Monitor memory usage and warn if it exceeds threshold.
        
        Args:
            warning_threshold: Threshold for memory usage warning (0.0 to 1.0)
        """
        usage_ratio = self.get_memory_usage() / self.memory_limit_bytes
        if usage_ratio > warning_threshold:
            warnings.warn(
                f"Memory usage is high: {usage_ratio:.1%} of limit. "
                f"Consider optimizing memory usage or increasing limit.",
                ResourceWarning
            )


def optimize_value_array(value_array: ValueArray) -> ValueArray:
    """
    Optimize a ValueArray for memory usage.
    
    Args:
        value_array: Input ValueArray
        
    Returns:
        ValueArray: Optimized ValueArray
    """
    optimizer = MemoryOptimizer()
    
    # Get the underlying numpy array
    values = value_array.values
    
    # Optimize the data type
    optimized_values = optimizer.optimize_array_dtype(values)
    
    # If optimization was possible, create a new ValueArray
    if optimized_values.dtype != values.dtype:
        return ValueArray.from_numpy(optimized_values)
    
    # Otherwise, return the original
    return value_array


def optimize_parameter_set(parameter_set: ParameterSet) -> ParameterSet:
    """
    Optimize a ParameterSet for memory usage.
    
    Args:
        parameter_set: Input ParameterSet
        
    Returns:
        ParameterSet: Optimized ParameterSet
    """
    optimizer = MemoryOptimizer()
    
    # Get the parameters as a dictionary
    params = parameter_set.parameters
    
    # Optimize each parameter array
    optimized_params = {}
    for name, values in params.items():
        optimized_params[name] = optimizer.optimize_array_dtype(values)
    
    # Create a new ParameterSet with optimized parameters
    return ParameterSet.from_numpy_or_dict(optimized_params)


def chunked_computation(
    func: Callable[[np.ndarray], Any],
    data: np.ndarray,
    chunk_size: Optional[int] = None,
    memory_optimizer: Optional[MemoryOptimizer] = None
) -> List[Any]:
    """
    Perform computation on data in chunks to reduce memory usage.
    
    Args:
        func: Function to apply to each chunk
        data: Input data array
        chunk_size: Size of each chunk. If None, determined automatically.
        memory_optimizer: MemoryOptimizer instance. If None, creates a new one.
        
    Returns:
        List[Any]: List of results from each chunk
    """
    if memory_optimizer is None:
        memory_optimizer = MemoryOptimizer()
    
    # If chunk_size is not specified, determine it based on available memory
    if chunk_size is None:
        # Estimate memory usage of the data
        data_memory = memory_optimizer.estimate_memory_usage(data)
        
        # If data is already small enough, process it directly
        if memory_optimizer.is_memory_available(data_memory * 2):  # Allow for result storage
            return [func(data)]
        
        # Otherwise, estimate chunk size based on available memory
        available_memory = memory_optimizer.get_available_memory()
        chunk_size = max(1, int(len(data) * (available_memory / (data_memory * 3))))
    
    # Split data into chunks
    chunks = memory_optimizer.chunk_large_array(data, chunk_size)
    
    # Process each chunk
    results = []
    for chunk in chunks:
        result = func(chunk)
        results.append(result)
        
        # Monitor memory usage
        memory_optimizer.monitor_memory_usage()
    
    return results


def memory_efficient_evpi_computation(
    net_benefit_array: np.ndarray,
    chunk_size: Optional[int] = None
) -> float:
    """
    Compute EVPI in a memory-efficient way using chunked computation.
    
    Args:
        net_benefit_array: 2D array of net benefits (samples x strategies)
        chunk_size: Size of chunks for processing. If None, determined automatically.
        
    Returns:
        float: Computed EVPI value
    """
    optimizer = MemoryOptimizer()
    
    # If array is small enough, compute directly
    if chunk_size is None:
        array_memory = optimizer.estimate_memory_usage(net_benefit_array)
        if optimizer.is_memory_available(array_memory * 4):  # Allow for intermediate arrays
            # Standard EVPI computation
            max_nb = np.max(net_benefit_array, axis=1)
            expected_nb_options = np.mean(net_benefit_array, axis=0)
            max_expected_nb = np.max(expected_nb_options)
            expected_max_nb = np.mean(max_nb)
            return float(expected_max_nb - max_expected_nb)
    
    # For large arrays, use chunked computation
    n_samples, n_strategies = net_benefit_array.shape
    
    if chunk_size is None:
        # Determine chunk size based on available memory
        sample_memory = net_benefit_array[0:1].nbytes * n_strategies  # Rough estimate
        available_memory = optimizer.get_available_memory()
        chunk_size = max(1, int(available_memory / (sample_memory * 4)))
    
    # Initialize accumulators for incremental computation
    max_nb_sum = 0.0
    strategy_sums = np.zeros(n_strategies, dtype=net_benefit_array.dtype)
    n_processed = 0
    
    # Process data in chunks
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = net_benefit_array[start_idx:end_idx]
        chunk_size_actual = chunk.shape[0]
        
        # Calculate max net benefit for each sample in chunk
        chunk_max_nb = np.max(chunk, axis=1)
        max_nb_sum += np.sum(chunk_max_nb)
        
        # Calculate sum of net benefits for each strategy
        chunk_strategy_sums = np.sum(chunk, axis=0)
        strategy_sums += chunk_strategy_sums
        
        n_processed += chunk_size_actual
    
    # Calculate final results
    expected_max_nb = max_nb_sum / n_processed
    expected_nb_options = strategy_sums / n_processed
    max_expected_nb = np.max(expected_nb_options)
    
    evpi = expected_max_nb - max_expected_nb
    return float(evpi)


# Example usage function
def example_memory_optimization():
    """Example of how to use memory optimization utilities."""
    # Create a large sample dataset
    np.random.seed(42)
    large_net_benefits = np.random.randn(10000, 5).astype(np.float64)
    
    print(f"Original array size: {large_net_benefits.nbytes / (1024*1024):.2f} MB")
    print(f"Original dtype: {large_net_benefits.dtype}")
    
    # Create memory optimizer
    optimizer = MemoryOptimizer()
    
    # Optimize array dtype
    optimized_array = optimizer.optimize_array_dtype(large_net_benefits)
    print(f"Optimized array size: {optimized_array.nbytes / (1024*1024):.2f} MB")
    print(f"Optimized dtype: {optimized_array.dtype}")
    
    # Check available memory
    available_memory = optimizer.get_available_memory()
    print(f"Available memory: {available_memory / (1024*1024):.2f} MB")
    
    # Perform memory-efficient EVPI computation
    evpi_result = memory_efficient_evpi_computation(large_net_benefits, chunk_size=1000)
    print(f"EVPI computed using memory-efficient method: {evpi_result}")
    
    # Force garbage collection
    optimizer.force_garbage_collection()
    print("Garbage collection completed")


if __name__ == "__main__":
    example_memory_optimization()