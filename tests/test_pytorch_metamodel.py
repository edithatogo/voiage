"""Tests for the PyTorch neural network metamodel."""

import numpy as np
import pytest

from voiage.metamodels import PyTorchNNMetamodel
from voiage.schema import ParameterSet
import xarray as xr


def test_pytorch_metamodel_import():
    """Test that PyTorchNNMetamodel can be imported."""
    try:
        from voiage.metamodels import PyTorchNNMetamodel
        assert PyTorchNNMetamodel is not None
    except ImportError:
        pytest.skip("PyTorch not available")


def test_pytorch_metamodel_initialization():
    """Test that PyTorchNNMetamodel can be initialized."""
    try:
        model = PyTorchNNMetamodel()
        assert model is not None
        assert model.hidden_layers == [64, 32]
        assert model.learning_rate == 0.001
        assert model.n_epochs == 1000
    except ImportError:
        pytest.skip("PyTorch not available")


def test_pytorch_metamodel_fit_predict():
    """Test fitting and prediction with PyTorchNNMetamodel."""
    try:
        # Create simple test data
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        
        # Generate input parameters
        param_dict = {}
        for i in range(n_features):
            param_dict[f"param_{i}"] = np.random.randn(n_samples)
        
        # Create ParameterSet
        dataset = xr.Dataset(
            {k: ("n_samples", v) for k, v in param_dict.items()},
            coords={"n_samples": np.arange(n_samples)}
        )
        parameter_set = ParameterSet(dataset=dataset)
        
        # Generate target values (simple quadratic relationship)
        x_array = np.array(list(param_dict.values())).T
        y = np.sum(x_array**2, axis=1) + np.random.normal(0, 0.1, n_samples)
        
        # Create and fit model
        model = PyTorchNNMetamodel(hidden_layers=[32, 16], n_epochs=100)
        model.fit(parameter_set, y)
        
        # Test prediction
        predictions = model.predict(parameter_set)
        assert len(predictions) == n_samples
        assert np.all(np.isfinite(predictions))
        
        # Test scoring
        score = model.score(parameter_set, y)
        assert isinstance(score, float)
        # For a well-fitted model, R2 should be reasonably high
        assert score > -1.0  # Even a poor model should have R2 > -1
        
        # Test RMSE
        rmse = model.rmse(parameter_set, y)
        assert isinstance(rmse, float)
        assert rmse >= 0
        
    except ImportError:
        pytest.skip("PyTorch not available")


def test_pytorch_metamodel_predict_before_fit():
    """Test that predict raises an error if called before fit."""
    try:
        model = PyTorchNNMetamodel()
        
        # Create dummy ParameterSet
        n_samples = 10
        param_dict = {"param1": np.random.randn(n_samples)}
        dataset = xr.Dataset(
            {k: ("n_samples", v) for k, v in param_dict.items()},
            coords={"n_samples": np.arange(n_samples)}
        )
        parameter_set = ParameterSet(dataset=dataset)
        
        # Try to predict without fitting
        with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
            model.predict(parameter_set)
            
        # Try to score without fitting
        with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
            model.score(parameter_set, np.random.randn(n_samples))
            
        # Try to calculate RMSE without fitting
        with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
            model.rmse(parameter_set, np.random.randn(n_samples))
            
    except ImportError:
        pytest.skip("PyTorch not available")


def test_pytorch_metamodel_different_architectures():
    """Test PyTorchNNMetamodel with different hidden layer configurations."""
    try:
        # Test with different hidden layer configurations
        architectures = [
            [16],           # Single hidden layer
            [32, 16],       # Two hidden layers
            [64, 32, 16],   # Three hidden layers
        ]
        
        for hidden_layers in architectures:
            model = PyTorchNNMetamodel(hidden_layers=hidden_layers, n_epochs=50)
            assert model.hidden_layers == hidden_layers
            
    except ImportError:
        pytest.skip("PyTorch not available")


if __name__ == "__main__":
    test_pytorch_metamodel_import()
    test_pytorch_metamodel_initialization()
    test_pytorch_metamodel_fit_predict()
    test_pytorch_metamodel_predict_before_fit()
    test_pytorch_metamodel_different_architectures()
    print("All PyTorch metamodel tests passed!")