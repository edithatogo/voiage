"""Tests for the fluent API in Value of Information analysis."""

import numpy as np
import pytest
import xarray as xr

from voiage.fluent import FluentDecisionAnalysis, create_analysis
from voiage.schema import ValueArray, ParameterSet


def test_fluent_analysis_creation():
    """Test creating a FluentDecisionAnalysis object."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Test creation with net benefits only
    analysis = FluentDecisionAnalysis(value_array)
    assert isinstance(analysis, FluentDecisionAnalysis)
    assert analysis.nb_array is not None
    
    # Test creation with parameters
    params_data = {
        'param1': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'param2': np.array([0.5, 1.5, 2.5], dtype=np.float64)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(params_data)
    
    analysis = FluentDecisionAnalysis(value_array, parameter_set)
    assert isinstance(analysis, FluentDecisionAnalysis)
    assert analysis.parameter_samples is not None


def test_fluent_with_parameters():
    """Test setting parameters with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create analysis without parameters
    analysis = FluentDecisionAnalysis(value_array)
    assert analysis.parameter_samples is None
    
    # Add parameters using fluent API
    params_data = {
        'param1': np.array([1.0, 2.0], dtype=np.float64),
        'param2': np.array([0.5, 1.5], dtype=np.float64)
    }
    
    result = analysis.with_parameters(params_data)
    assert result is analysis  # Should return self for chaining
    assert analysis.parameter_samples is not None
    assert 'param1' in analysis.parameter_samples.parameters
    assert 'param2' in analysis.parameter_samples.parameters


def test_fluent_with_backend():
    """Test setting backend with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create analysis
    analysis = FluentDecisionAnalysis(value_array)
    
    # Set backend using fluent API
    result = analysis.with_backend("numpy")
    assert result is analysis  # Should return self for chaining
    assert analysis.backend is not None


def test_fluent_with_jit():
    """Test setting JIT compilation with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create analysis
    analysis = FluentDecisionAnalysis(value_array)
    
    # Enable JIT using fluent API
    result = analysis.with_jit(True)
    assert result is analysis  # Should return self for chaining
    assert analysis.use_jit is True
    
    # Disable JIT using fluent API
    result = analysis.with_jit(False)
    assert result is analysis  # Should return self for chaining
    assert analysis.use_jit is False


def test_fluent_with_streaming():
    """Test setting streaming support with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create analysis
    analysis = FluentDecisionAnalysis(value_array)
    
    # Enable streaming using fluent API
    result = analysis.with_streaming(100)
    assert result is analysis  # Should return self for chaining
    assert analysis.streaming_window_size == 100
    assert analysis._streaming_data_buffer is not None


def test_fluent_with_caching():
    """Test setting caching with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create analysis
    analysis = FluentDecisionAnalysis(value_array)
    
    # Enable caching using fluent API
    result = analysis.with_caching(True)
    assert result is analysis  # Should return self for chaining
    assert analysis.enable_caching is True
    assert analysis._cache is not None
    
    # Disable caching using fluent API
    result = analysis.with_caching(False)
    assert result is analysis  # Should return self for chaining
    assert analysis.enable_caching is False
    assert analysis._cache is None


def test_fluent_add_data():
    """Test adding data with fluent API."""
    # Create initial test data
    initial_net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(initial_net_benefits)
    
    # Create analysis
    analysis = FluentDecisionAnalysis(value_array)
    
    # Add new data using fluent API
    new_net_benefits = np.array([[110.0, 110.0]], dtype=np.float64)
    new_value_array = ValueArray.from_numpy(new_net_benefits)
    
    result = analysis.add_data(new_value_array)
    assert result is analysis  # Should return self for chaining
    # Check that data was added (shape should be larger)
    assert analysis.nb_array.values.shape[0] > initial_net_benefits.shape[0]


def test_fluent_calculate_evpi():
    """Test calculating EVPI with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create analysis
    analysis = FluentDecisionAnalysis(value_array)
    
    # Calculate EVPI using fluent API
    result = analysis.calculate_evpi()
    assert result is analysis  # Should return self for chaining
    assert analysis.get_evpi_result() is not None
    assert isinstance(analysis.get_evpi_result(), float)
    assert analysis.get_evpi_result() >= 0


def test_fluent_calculate_evppi():
    """Test calculating EVPPI with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create parameters
    params_data = {
        'param1': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'param2': np.array([0.5, 1.5, 2.5], dtype=np.float64)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(params_data)
    
    # Create analysis with parameters
    analysis = FluentDecisionAnalysis(value_array, parameter_set)
    
    # Calculate EVPPI using fluent API
    result = analysis.calculate_evppi()
    assert result is analysis  # Should return self for chaining
    assert analysis.get_evppi_result() is not None
    assert isinstance(analysis.get_evppi_result(), float)
    assert analysis.get_evppi_result() >= 0


def test_fluent_get_results():
    """Test getting results with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create parameters
    params_data = {
        'param1': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'param2': np.array([0.5, 1.5, 2.5], dtype=np.float64)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(params_data)
    
    # Create analysis and calculate results
    results = (FluentDecisionAnalysis(value_array, parameter_set)
               .calculate_evpi()
               .calculate_evppi()
               .get_results())
    
    assert isinstance(results, dict)
    assert "evpi" in results
    assert "evppi" in results
    assert results["evpi"] is not None
    assert results["evppi"] is not None
    assert isinstance(results["evpi"], float)
    assert isinstance(results["evppi"], float)


def test_create_analysis_factory():
    """Test the create_analysis factory function."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    
    # Test factory function
    analysis = create_analysis(net_benefits)
    assert isinstance(analysis, FluentDecisionAnalysis)
    
    # Test factory function with parameters
    params_data = {
        'param1': np.array([1.0, 2.0], dtype=np.float64),
    }
    analysis = create_analysis(net_benefits, params_data)
    assert isinstance(analysis, FluentDecisionAnalysis)
    assert analysis.parameter_samples is not None


def test_fluent_chaining():
    """Test method chaining with fluent API."""
    # Create test data
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0]], dtype=np.float64)
    params_data = {
        'param1': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'param2': np.array([0.5, 1.5, 2.5], dtype=np.float64)
    }
    
    # Test full method chaining
    analysis = (create_analysis(net_benefits)
                .with_parameters(params_data)
                .with_backend("numpy")
                .with_jit()
                .with_caching()
                .with_streaming(100)
                .calculate_evpi()
                .calculate_evppi())
    
    assert isinstance(analysis, FluentDecisionAnalysis)
    assert analysis.parameter_samples is not None
    assert analysis.backend is not None
    assert analysis.use_jit is True
    assert analysis.enable_caching is True
    assert analysis.streaming_window_size == 100
    assert analysis.get_evpi_result() is not None
    assert analysis.get_evppi_result() is not None


if __name__ == "__main__":
    test_fluent_analysis_creation()
    test_fluent_with_parameters()
    test_fluent_with_backend()
    test_fluent_with_jit()
    test_fluent_with_streaming()
    test_fluent_with_caching()
    test_fluent_add_data()
    test_fluent_calculate_evpi()
    test_fluent_calculate_evppi()
    test_fluent_get_results()
    test_create_analysis_factory()
    test_fluent_chaining()
    print("All fluent API tests passed!")