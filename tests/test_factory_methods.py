"""Tests for factory methods for common Value of Information analysis patterns."""

import numpy as np
import pytest

from voiage.factory import (
    create_standard_analysis, create_streaming_analysis, create_healthcare_analysis,
    create_environmental_analysis, create_financial_analysis, create_large_scale_analysis,
    create_metamodel_analysis, create_configured_analysis
)
from voiage.config_objects import VOIAnalysisConfig, HealthcareConfig
from voiage.analysis import DecisionAnalysis
from voiage.fluent import FluentDecisionAnalysis
from voiage.schema import ParameterSet, ValueArray
import xarray as xr


def test_create_standard_analysis():
    """Test creating a standard VOI analysis."""
    # Create test data
    net_benefits = np.random.randn(100, 3).astype(np.float64)
    param_data = {
        'param1': np.random.randn(100).astype(np.float64),
        'param2': np.random.randn(100).astype(np.float64)
    }
    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in param_data.items()},
        coords={"n_samples": np.arange(len(param_data['param1']))},
    )
    parameters = ParameterSet(dataset=dataset)
    
    # Create standard analysis
    analysis = create_standard_analysis(
        nb_array=net_benefits,
        parameter_samples=parameters,
        use_jit=True,
        backend="numpy",
        enable_caching=True
    )
    
    # Check that we got the right type of object
    assert isinstance(analysis, DecisionAnalysis)
    
    # Check that parameters were set correctly
    assert analysis.use_jit == True
    assert analysis.enable_caching == True
    
    # Check that data was set correctly
    assert isinstance(analysis.nb_array, ValueArray)
    assert analysis.parameter_samples is not None


def test_create_streaming_analysis():
    """Test creating a streaming VOI analysis."""
    # Create test data
    net_benefits = np.random.randn(100, 3).astype(np.float64)
    
    # Create streaming analysis
    analysis = create_streaming_analysis(
        nb_array=net_benefits,
        window_size=500,
        update_frequency=50,
        use_jit=True,
        backend="numpy"
    )
    
    # Check that we got the right type of object
    assert isinstance(analysis, FluentDecisionAnalysis)
    
    # Check that streaming parameters were set correctly
    assert analysis.streaming_window_size == 500
    
    # Check that other parameters were set correctly
    assert analysis.use_jit == True
    assert analysis.enable_caching == True  # Enabled by default in streaming


def test_create_healthcare_analysis():
    """Test creating a healthcare VOI analysis."""
    # Create test data
    net_benefits = np.random.randn(100, 3).astype(np.float64)
    
    # Create healthcare analysis
    analysis = create_healthcare_analysis(
        nb_array=net_benefits,
        use_jit=True
    )
    
    # Check that we got the right type of object
    assert isinstance(analysis, DecisionAnalysis)
    
    # Check that parameters were set correctly
    assert analysis.use_jit == True


def test_create_environmental_analysis():
    """Test creating an environmental VOI analysis."""
    # Create test data
    net_benefits = np.random.randn(100, 3).astype(np.float64)
    
    # Create environmental analysis
    analysis = create_environmental_analysis(
        nb_array=net_benefits,
        carbon_intensity=0.6,
        energy_consumption=15000,
        water_intensity=0.15
    )
    
    # Check that we got the right type of object
    assert isinstance(analysis, DecisionAnalysis)
    
    # Check that parameters were set correctly
    assert analysis.use_jit == True
    assert analysis.enable_caching == True


def test_create_financial_analysis():
    """Test creating a financial VOI analysis."""
    # Create test data
    net_benefits = np.random.randn(100, 3).astype(np.float64)
    
    # Create financial analysis
    analysis = create_financial_analysis(
        nb_array=net_benefits,
        var_confidence_level=0.99,
        cvar_confidence_level=0.99,
        mc_n_simulations=5000
    )
    
    # Check that we got the right type of object
    assert isinstance(analysis, DecisionAnalysis)
    
    # Check that parameters were set correctly
    assert analysis.use_jit == True
    assert analysis.enable_caching == True


def test_create_large_scale_analysis():
    """Test creating a large-scale VOI analysis."""
    # Create test data
    net_benefits = np.random.randn(100, 3).astype(np.float64)
    
    # Create large-scale analysis
    analysis = create_large_scale_analysis(
        nb_array=net_benefits,
        chunk_size=5000,
        n_workers=4,
        memory_limit_mb=1024
    )
    
    # Check that we got the right type of object
    assert isinstance(analysis, FluentDecisionAnalysis)
    
    # Check that parameters were set correctly
    assert analysis.use_jit == True
    assert analysis.enable_caching == True


def test_create_metamodel_analysis():
    """Test creating a metamodel VOI analysis."""
    # Create test data
    net_benefits = np.random.randn(100, 3).astype(np.float64)
    
    # Create metamodel analysis
    analysis = create_metamodel_analysis(
        nb_array=net_benefits,
        method="gp",
        n_samples=5000,
        n_folds=3
    )
    
    # Check that we got the right type of object
    assert isinstance(analysis, DecisionAnalysis)
    
    # Check that parameters were set correctly
    assert analysis.use_jit == False  # Metamodels don't use JIT by default


def test_create_configured_analysis():
    """Test creating a VOI analysis from a configuration object."""
    # Create configuration
    config = VOIAnalysisConfig(
        use_jit=True,
        backend="numpy",
        enable_caching=True,
        streaming_window_size=1000
    )
    
    # Create analysis from configuration
    analysis = create_configured_analysis(config)
    
    # Check that we got the right type of object
    assert isinstance(analysis, DecisionAnalysis)
    
    # Check that parameters were set correctly
    assert analysis.use_jit == True
    assert analysis.enable_caching == True
    assert analysis.streaming_window_size == 1000


def test_healthcare_config_validation():
    """Test that healthcare configuration validates parameters correctly."""
    # This should work fine
    config = HealthcareConfig(
        qaly_discount_rate=0.03,
        cost_discount_rate=0.03
    )
    
    # This should raise a ValueError
    with pytest.raises(ValueError):
        HealthcareConfig(
            qaly_discount_rate=1.5,  # Invalid - should be between 0 and 1
            cost_discount_rate=0.03
        )


if __name__ == "__main__":
    test_create_standard_analysis()
    test_create_streaming_analysis()
    test_create_healthcare_analysis()
    test_create_environmental_analysis()
    test_create_financial_analysis()
    test_create_large_scale_analysis()
    test_create_metamodel_analysis()
    test_create_configured_analysis()
    test_healthcare_config_validation()
    print("All factory method tests passed!")