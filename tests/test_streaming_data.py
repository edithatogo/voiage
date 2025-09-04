"""Tests for streaming data support in DecisionAnalysis."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet
import xarray as xr


def test_streaming_data_initialization():
    """Test initialization of DecisionAnalysis with streaming support."""
    # Create test data with float64 dtype
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0], [95.0, 125.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Test initialization without streaming
    analysis1 = DecisionAnalysis(value_array)
    assert analysis1.streaming_window_size is None
    assert analysis1._streaming_data_buffer is None
    
    # Test initialization with streaming
    analysis2 = DecisionAnalysis(value_array, streaming_window_size=10)
    assert analysis2.streaming_window_size == 10
    assert analysis2._streaming_data_buffer is not None
    assert analysis2._streaming_data_buffer.maxlen == 10


def test_update_with_new_data_without_streaming():
    """Test updating data without streaming buffers."""
    # Create initial data with float64 dtype
    initial_net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(initial_net_benefits)
    analysis = DecisionAnalysis(value_array)
    
    # Add new data with float64 dtype
    new_net_benefits = np.array([[110.0, 110.0], [95.0, 125.0]], dtype=np.float64)
    new_value_array = ValueArray.from_numpy(new_net_benefits)
    analysis.update_with_new_data(new_value_array)
    
    # Check that data was appended correctly
    expected_net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0], [95.0, 125.0]], dtype=np.float64)
    np.testing.assert_array_equal(analysis.nb_array.values, expected_net_benefits)


def test_update_with_new_data_with_streaming():
    """Test updating data with streaming buffers."""
    # Create initial data with float64 dtype
    initial_net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(initial_net_benefits)
    analysis = DecisionAnalysis(value_array, streaming_window_size=3)
    
    # Add new data with float64 dtype
    new_net_benefits = np.array([[110.0, 110.0], [95.0, 125.0], [105.0, 115.0], [85.0, 135.0]], dtype=np.float64)
    new_value_array = ValueArray.from_numpy(new_net_benefits)
    analysis.update_with_new_data(new_value_array)
    
    # With window size 3, only the last 3 samples should be kept
    # The buffer should contain: [95.0, 125.0], [105.0, 115.0], [85.0, 135.0]
    expected_net_benefits = np.array([[95.0, 125.0], [105.0, 115.0], [85.0, 135.0]], dtype=np.float64)
    np.testing.assert_array_equal(analysis.nb_array.values, expected_net_benefits)


def test_update_with_new_data_with_parameters():
    """Test updating data with parameter samples."""
    # Create initial data with parameters and float64 dtype
    initial_net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(initial_net_benefits)
    
    # Create parameter samples with float64 dtype
    param_dict = {
        "param1": np.array([1.0, 2.0], dtype=np.float64),
        "param2": np.array([0.5, 1.5], dtype=np.float64)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(param_dict)
    
    analysis = DecisionAnalysis(value_array, parameter_set, streaming_window_size=3)
    
    # Add new data with parameters and float64 dtype
    new_net_benefits = np.array([[110.0, 110.0], [95.0, 125.0]], dtype=np.float64)
    new_value_array = ValueArray.from_numpy(new_net_benefits)
    
    new_param_dict = {
        "param1": np.array([3.0, 4.0], dtype=np.float64),
        "param2": np.array([2.5, 3.5], dtype=np.float64)
    }
    new_parameter_set = ParameterSet.from_numpy_or_dict(new_param_dict)
    
    analysis.update_with_new_data(new_value_array, new_parameter_set)
    
    # Check that data was updated correctly
    expected_net_benefits = np.array([[110.0, 110.0], [95.0, 125.0]], dtype=np.float64)  # Last 2 samples due to window size
    np.testing.assert_array_equal(analysis.nb_array.values, expected_net_benefits)
    
    # Check that parameters were updated correctly
    assert analysis.parameter_samples is not None
    assert "param1" in analysis.parameter_samples.parameters
    assert "param2" in analysis.parameter_samples.parameters


def test_streaming_evpi_generator():
    """Test the streaming EVPI generator."""
    # Create test data with float64 dtype
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0], [95.0, 125.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    analysis = DecisionAnalysis(value_array)
    
    # Get the streaming EVPI generator
    evpi_generator = analysis.streaming_evpi()
    
    # Get first value
    evpi_value = next(evpi_generator)
    assert isinstance(evpi_value, float)
    assert evpi_value >= 0


def test_streaming_evppi_generator():
    """Test the streaming EVPPI generator."""
    # Create test data with parameters and float64 dtype
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0], [110.0, 110.0], [95.0, 125.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    
    # Create parameter samples with float64 dtype
    param_dict = {
        "param1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "param2": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(param_dict)
    
    analysis = DecisionAnalysis(value_array, parameter_set)
    
    # Get the streaming EVPPI generator
    evppi_generator = analysis.streaming_evppi()
    
    # Get first value (this might raise an exception if sklearn is not available)
    try:
        evppi_value = next(evppi_generator)
        assert isinstance(evppi_value, float)
        assert evppi_value >= 0
    except ImportError:
        # If sklearn is not available, skip this test
        pytest.skip("scikit-learn not available")


def test_streaming_with_large_data():
    """Test streaming with larger datasets."""
    # Create initial data with float64 dtype
    initial_net_benefits = np.random.rand(100, 3).astype(np.float64) * 100
    value_array = ValueArray.from_numpy(initial_net_benefits)
    analysis = DecisionAnalysis(value_array, streaming_window_size=50)
    
    # Add large amount of new data with float64 dtype
    new_net_benefits = np.random.rand(200, 3).astype(np.float64) * 100
    new_value_array = ValueArray.from_numpy(new_net_benefits)
    analysis.update_with_new_data(new_value_array)
    
    # Check that only the window size amount of data is kept
    assert analysis.nb_array.values.shape[0] == 50


if __name__ == "__main__":
    test_streaming_data_initialization()
    test_update_with_new_data_without_streaming()
    test_update_with_new_data_with_streaming()
    test_update_with_new_data_with_parameters()
    test_streaming_evpi_generator()
    test_streaming_evppi_generator()
    test_streaming_with_large_data()
    print("All streaming data tests passed!")