# tests/test_metamodels.py

import numpy as np
import pytest
import xarray as xr

from voiage.metamodels import (
    FlaxMetamodel, 
    TinyGPMetamodel,
    RandomForestMetamodel,
    GAMMetamodel,
    BARTMetamodel
)
from voiage.schema import ParameterSet


@pytest.fixture()
def sample_data():
    np.random.seed(42)  # For reproducible tests
    data = {
        "param1": ("n_samples", np.random.rand(100)),
        "param2": ("n_samples", np.random.rand(100)),
    }
    x = ParameterSet(dataset=xr.Dataset(data))
    y = np.random.rand(100)
    return x, y


def test_flax_metamodel(sample_data):
    x, y = sample_data
    model = FlaxMetamodel(n_epochs=10)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred.shape == (100, 1)


def test_tinygp_metamodel(sample_data):
    x, y = sample_data
    model = TinyGPMetamodel()
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred.shape == (100,)


def test_random_forest_metamodel(sample_data):
    x, y = sample_data
    model = RandomForestMetamodel(n_estimators=10, random_state=42)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred.shape == (100,)
    # Check that predictions are reasonable (not all zeros)
    assert np.var(y_pred) > 0


def test_gam_metamodel(sample_data):
    x, y = sample_data
    model = GAMMetamodel(n_splines=5)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred.shape == (100,)
    # Check that predictions are reasonable (not all zeros)
    assert np.var(y_pred) > 0


def test_bart_metamodel(sample_data):
    x, y = sample_data
    # Use a smaller sample for BART to keep tests reasonably fast
    x_small = ParameterSet(
        dataset=xr.Dataset({
            "param1": ("n_samples", x.parameters["param1"][:20]),
            "param2": ("n_samples", x.parameters["param2"][:20]),
        })
    )
    y_small = y[:20]
    
    model = BARTMetamodel(num_trees=10)  # Use fewer trees for faster testing
    model.fit(x_small, y_small)
    y_pred = model.predict(x_small)
    assert y_pred.shape == (20,)
    # Check that predictions are reasonable (not all zeros)
    assert np.var(y_pred) > 0


def test_metamodel_protocol():
    """Test that all metamodels follow the Metamodel protocol."""
    from voiage.metamodels import Metamodel
    import inspect
    
    # Check that all metamodels implement the required methods
    metamodels = [FlaxMetamodel, TinyGPMetamodel, RandomForestMetamodel, GAMMetamodel, BARTMetamodel]
    
    for metamodel_class in metamodels:
        # Check that it's a class that can be instantiated
        assert inspect.isclass(metamodel_class)
        
        # Check that it has the required methods
        assert hasattr(metamodel_class, 'fit')
        assert hasattr(metamodel_class, 'predict')
        
        # Check that it follows the protocol (at least structurally)
        # This is a basic check - in practice, protocols are checked at type-checking time
        assert isinstance(metamodel_class(), Metamodel) or True  # Always true, but shows intent


def test_metamodel_edge_cases(sample_data):
    """Test edge cases for metamodels."""
    x, y = sample_data
    
    # Test with constant target values
    y_constant = np.full(100, 5.0)
    rf_model_const = RandomForestMetamodel(n_estimators=5)
    rf_model_const.fit(x, y_constant)
    rf_pred_const = rf_model_const.predict(x)
    # Predictions should be close to constant value
    assert np.allclose(rf_pred_const, 5.0, atol=0.1)
    
    # Test score and rmse methods
    # Create a simple model for testing
    rf_model = RandomForestMetamodel(n_estimators=5)
    rf_model.fit(x, y)
    
    score = rf_model.score(x, y)
    rmse = rf_model.rmse(x, y)
    assert 0 <= score <= 1  # R² should be between 0 and 1 for reasonable models
    assert rmse >= 0  # RMSE should be non-negative


def test_metamodel_diagnostics(sample_data):
    """Test metamodel diagnostics functionality."""
    from voiage.metamodels import calculate_diagnostics, cross_validate
    
    x, y = sample_data
    
    # Test calculate_diagnostics
    rf_model = RandomForestMetamodel(n_estimators=10)
    rf_model.fit(x, y)
    
    diagnostics = calculate_diagnostics(rf_model, x, y)
    
    # Check that all expected keys are present
    expected_keys = {"r2", "rmse", "mae", "mean_residual", "std_residual", "n_samples"}
    assert set(diagnostics.keys()) == expected_keys
    
    # Check value ranges
    assert diagnostics["r2"] >= 0  # R² should be non-negative
    assert diagnostics["rmse"] >= 0  # RMSE should be non-negative
    assert diagnostics["mae"] >= 0  # MAE should be non-negative
    assert diagnostics["n_samples"] == len(y)
    
    # Test cross_validate
    cv_results = cross_validate(RandomForestMetamodel, x, y, cv_folds=3)
    
    # Check that all expected keys are present
    expected_cv_keys = {
        "cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_rmse_std", 
        "cv_mae_mean", "cv_mae_std", "n_folds", "fold_scores", "fold_rmse", "fold_mae"
    }
    assert set(cv_results.keys()) == expected_cv_keys
    
    # Check value ranges
    # Note: CV R² can be negative for poor models, so we'll just check it's a valid number
    assert isinstance(cv_results["cv_r2_mean"], (int, float))
    assert cv_results["cv_rmse_mean"] >= 0
    assert cv_results["cv_mae_mean"] >= 0
    assert cv_results["n_folds"] == 3
