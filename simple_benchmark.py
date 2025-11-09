#!/usr/bin/env python3
"""
Simplified Performance Benchmark for voiage - Establishing NumPy Backend Baselines

This script establishes baseline performance metrics for the current NumPy backend
before implementing JAX backend integration. Results will be used to measure
JAX performance improvements.

Usage:
    python simple_benchmark.py [--sizes small,medium,large] [--iterations 5]
"""

import time
import sys
import os
import psutil
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voiage'))

# Import voiage modules
try:
    from voiage.analysis import DecisionAnalysis
    from voiage.schema import ValueArray, ParameterSet
    from voiage.backends import get_backend, NumpyBackend
    print("✅ Successfully imported voiage modules")
except ImportError as e:
    print(f"❌ Error importing voiage modules: {e}")
    sys.exit(1)

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    method_name: str
    dataset_size: str
    execution_time: float
    memory_usage_mb: float
    memory_peak_mb: float
    parameters: int
    options: int
    samples: int
    backend: str

class PerformanceMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current and peak memory usage in MB."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_info = self.process.memory_info()
        peak_memory = memory_info.rss / 1024 / 1024  # MB
        
        # Try to get peak memory if available (Unix-like systems)
        try:
            with open(f'/proc/{self.process.pid}/status') as f:
                for line in f:
                    if 'VmPeak' in line:
                        peak_memory = float(line.split()[1]) / 1024  # Convert kB to MB
                        break
        except (FileNotFoundError, ValueError):
            # VmPeak not available, use current memory as peak
            pass
            
        return current_memory, peak_memory

def create_test_data(size: str = "medium") -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Create test data for benchmarking.
    
    Parameters
    ----------
    size : str
        Size of test data: "small", "medium", "large"
    
    Returns
    -------
    Tuple[np.ndarray, List[str], List[str]]
        Net benefit array, parameter names, option names
    """
    np.random.seed(42)  # For reproducible results
    
    if size == "small":
        n_params, n_options, n_samples = 3, 4, 100
    elif size == "medium":
        n_params, n_options, n_samples = 5, 6, 1000
    elif size == "large":
        n_params, n_options, n_samples = 10, 8, 10000
    else:
        raise ValueError(f"Unknown size: {size}")
    
    # Create net benefit array (samples x options)
    net_benefit_array = np.random.randn(n_samples, n_options) * 100 + 1000
    
    parameter_names = [f"param_{i}" for i in range(n_params)]
    option_names = [f"option_{i}" for i in range(n_options)]
    
    return net_benefit_array, parameter_names, option_names

def benchmark_backend_evpi(data: Tuple, monitor: PerformanceMonitor, iterations: int = 5) -> BenchmarkResult:
    """Benchmark EVPI calculation using backend directly."""
    net_benefit_array, param_names, option_names = data
    results = []
    
    for _ in range(iterations):
        current_memory, peak_memory = monitor.get_memory_usage()
        start_time = time.perf_counter()
        
        # Calculate EVPI using NumPy backend directly
        backend = get_backend("numpy")
        evpi_result = backend.evpi(net_benefit_array)
        
        end_time = time.perf_counter()
        current_memory, peak_memory = monitor.get_memory_usage()
        
        results.append(BenchmarkResult(
            method_name="Backend_EVPI",
            dataset_size="",
            execution_time=end_time - start_time,
            memory_usage_mb=current_memory,
            memory_peak_mb=peak_memory,
            parameters=len(param_names),
            options=len(option_names),
            samples=len(net_benefit_array),
            backend="numpy"
        ))
    
    # Return average results
    avg_time = np.mean([r.execution_time for r in results])
    avg_memory = np.mean([r.memory_usage_mb for r in results])
    avg_peak_memory = np.max([r.memory_peak_mb for r in results])
    
    return BenchmarkResult(
        method_name="Backend_EVPI",
        dataset_size="",
        execution_time=avg_time,
        memory_usage_mb=avg_memory,
        memory_peak_mb=avg_peak_memory,
        parameters=len(param_names),
        options=len(option_names),
        samples=len(net_benefit_array),
        backend="numpy"
    )

def benchmark_decision_analysis_evpi(data: Tuple, monitor: PerformanceMonitor, iterations: int = 5) -> BenchmarkResult:
    """Benchmark EVPI calculation using DecisionAnalysis class."""
    net_benefit_array, param_names, option_names = data
    results = []
    
    for _ in range(iterations):
        current_memory, peak_memory = monitor.get_memory_usage()
        start_time = time.perf_counter()
        
        # Create DecisionAnalysis instance and calculate EVPI
        value_array = ValueArray.from_numpy(net_benefit_array, option_names)
        analysis = DecisionAnalysis(value_array)
        evpi_result = analysis.evpi()
        
        end_time = time.perf_counter()
        current_memory, peak_memory = monitor.get_memory_usage()
        
        results.append(BenchmarkResult(
            method_name="DecisionAnalysis_EVPI",
            dataset_size="",
            execution_time=end_time - start_time,
            memory_usage_mb=current_memory,
            memory_peak_mb=peak_memory,
            parameters=len(param_names),
            options=len(option_names),
            samples=len(net_benefit_array),
            backend="numpy"
        ))
    
    # Return average results
    avg_time = np.mean([r.execution_time for r in results])
    avg_memory = np.mean([r.memory_usage_mb for r in results])
    avg_peak_memory = np.max([r.memory_peak_mb for r in results])
    
    return BenchmarkResult(
        method_name="DecisionAnalysis_EVPI",
        dataset_size="",
        execution_time=avg_time,
        memory_usage_mb=avg_memory,
        memory_peak_mb=avg_peak_memory,
        parameters=len(param_names),
        options=len(option_names),
        samples=len(net_benefit_array),
        backend="numpy"
    )

def benchmark_decision_analysis_evppi(data: Tuple, monitor: PerformanceMonitor, iterations: int = 5) -> BenchmarkResult:
    """Benchmark EVPPI calculation using DecisionAnalysis class."""
    net_benefit_array, param_names, option_names = data
    results = []
    
    for _ in range(iterations):
        current_memory, peak_memory = monitor.get_memory_usage()
        start_time = time.perf_counter()
        
        # Create DecisionAnalysis instance and calculate EVPPI
        value_array = ValueArray.from_numpy(net_benefit_array, option_names)
        
        # Create parameter samples for EVPPI
        param_samples = {
            name: np.random.randn(len(net_benefit_array)) for name in param_names
        }
        analysis = DecisionAnalysis(value_array, parameter_samples=param_samples)
        evppi_result = analysis.evppi(parameters_of_interest=[param_names[0]])
        
        end_time = time.perf_counter()
        current_memory, peak_memory = monitor.get_memory_usage()
        
        results.append(BenchmarkResult(
            method_name="DecisionAnalysis_EVPPI",
            dataset_size="",
            execution_time=end_time - start_time,
            memory_usage_mb=current_memory,
            memory_peak_mb=peak_memory,
            parameters=len(param_names),
            options=len(option_names),
            samples=len(net_benefit_array),
            backend="numpy"
        ))
    
    # Return average results
    avg_time = np.mean([r.execution_time for r in results])
    avg_memory = np.mean([r.memory_usage_mb for r in results])
    avg_peak_memory = np.max([r.memory_peak_mb for r in results])
    
    return BenchmarkResult(
        method_name="DecisionAnalysis_EVPPI",
        dataset_size="",
        execution_time=avg_time,
        memory_usage_mb=avg_memory,
        memory_peak_mb=avg_peak_memory,
        parameters=len(param_names),
        options=len(option_names),
        samples=len(net_benefit_array),
        backend="numpy"
    )

def run_comprehensive_benchmark(sizes: List[str] = ["small", "medium", "large"], iterations: int = 5) -> Dict[str, List[BenchmarkResult]]:
    """Run comprehensive benchmark across all sizes and methods."""
    print("🚀 Starting voiage Performance Benchmark")
    print("=" * 50)
    
    all_results = {}
    
    for size in sizes:
        print(f"\n📊 Benchmarking {size.upper()} dataset...")
        print(f"   Dataset parameters: {size} (samples, options, parameters)")
        
        # Create test data
        data = create_test_data(size)
        net_benefit_array, param_names, option_names = data
        
        print(f"   Net benefit array shape: {net_benefit_array.shape}")
        print(f"   Parameters: {param_names}")
        print(f"   Options: {option_names}")
        
        # Initialize performance monitor
        monitor = PerformanceMonitor()
        
        size_results = []
        
        # Benchmark each method
        methods = [
            ("Backend_EVPI", benchmark_backend_evpi),
            ("DecisionAnalysis_EVPI", benchmark_decision_analysis_evpi),
            ("DecisionAnalysis_EVPPI", benchmark_decision_analysis_evppi),
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"   🔄 Benchmarking {method_name}...")
                result = method_func(data, monitor, iterations)
                result.dataset_size = size
                size_results.append(result)
                
                print(f"      ⏱️  {result.execution_time:.4f}s")
                print(f"      💾 {result.memory_peak_mb:.2f} MB peak")
                
            except Exception as e:
                print(f"      ❌ Error in {method_name}: {e}")
                continue
        
        all_results[size] = size_results
        
        print(f"   ✅ Completed {size.upper()} dataset")
    
    return all_results

def save_results(results: Dict[str, List[BenchmarkResult]], output_file: str = "baseline_performance_results.json"):
    """Save benchmark results to JSON file."""
    serialized_results = {}
    
    for size, size_results in results.items():
        serialized_results[size] = []
        for result in size_results:
            serialized_results[size].append({
                "method_name": result.method_name,
                "dataset_size": result.dataset_size,
                "execution_time": result.execution_time,
                "memory_usage_mb": result.memory_usage_mb,
                "memory_peak_mb": result.memory_peak_mb,
                "parameters": result.parameters,
                "options": result.options,
                "samples": result.samples,
                "backend": result.backend
            })
    
    with open(output_file, 'w') as f:
        json.dump(serialized_results, f, indent=2)
    
    print(f"💾 Results saved to {output_file}")

def print_summary(results: Dict[str, List[BenchmarkResult]]):
    """Print a summary of benchmark results."""
    print("\n📈 PERFORMANCE BASELINE SUMMARY")
    print("=" * 50)
    
    for size, size_results in results.items():
        print(f"\n🔹 {size.upper()} Dataset Results:")
        print("-" * 30)
        
        for result in size_results:
            print(f"   {result.method_name}:")
            print(f"      ⏱️  Time: {result.execution_time:.4f}s")
            print(f"      💾 Peak Memory: {result.memory_peak_mb:.2f} MB")
            print(f"      📊 Data: {result.samples} samples × {result.options} options")
    
    # Calculate performance scaling
    print(f"\n📊 Performance Scaling Analysis:")
    print("-" * 30)
    
    if "small" in results and "medium" in results and "large" in results:
        small_backend = next(r.execution_time for r in results["small"] if r.method_name == "Backend_EVPI")
        medium_backend = next(r.execution_time for r in results["medium"] if r.method_name == "Backend_EVPI")
        large_backend = next(r.execution_time for r in results["large"] if r.method_name == "Backend_EVPI")
        
        print(f"   Backend EVPI Time Scaling:")
        print(f"      Small → Medium: {medium_backend/small_backend:.1f}x")
        print(f"      Medium → Large: {large_backend/medium_backend:.1f}x")
        print(f"      Small → Large: {large_backend/small_backend:.1f}x")

def main():
    """Main benchmarking function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="voiage Performance Benchmark")
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large"],
                      help="Dataset sizes to benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of iterations per benchmark")
    parser.add_argument("--output", default="baseline_performance_results.json",
                      help="Output file for results")
    
    args = parser.parse_args()
    
    print("🎯 voiage Performance Benchmarking")
    print(f"   Dataset sizes: {args.sizes}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Backend: NumPy (baseline)")
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark(args.sizes, args.iterations)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output)
    
    print(f"\n✅ Benchmark completed successfully!")
    print(f"📁 Results saved to {args.output}")
    print(f"🚀 Ready for JAX backend implementation comparison!")

if __name__ == "__main__":
    main()