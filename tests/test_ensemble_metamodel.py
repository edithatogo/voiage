"""Tests for the ensemble metamodel."""

import numpy as np
import pytest
import xarray as xr

from voiage.metamodels import EnsembleMetamodel, GAMMetamodel, RandomForestMetamodel
from voiage.schema import ParameterSet


def create_test_data(n_samples=100, n_features=3):
    """Create test data for metamodel testing."""
    np.random.seed(42)

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

    # Generate target values (simple quadratic relationship with noise)
    x_array = np.array(list(param_dict.values())).T
    y = np.sum(x_array**2, axis=1) + np.random.normal(0, 0.1, n_samples)

    return parameter_set, y


def test_ensemble_metamodel_initialization():
    """Test that EnsembleMetamodel can be initialized."""
    # Create some dummy models
    try:
        rf_model = RandomForestMetamodel()
        gam_model = GAMMetamodel()
        models = [rf_model, gam_model]
    except ImportError:
        pytest.skip("Required metamodels not available")

    # Test initialization with different methods
    ensemble_mean = EnsembleMetamodel(models, method='mean')
    assert ensemble_mean.models == models
    assert ensemble_mean.method == 'mean'

    ensemble_median = EnsembleMetamodel(models, method='median')
    assert ensemble_median.method == 'median'

    ensemble_weighted = EnsembleMetamodel(models, method='weighted')
    assert ensemble_weighted.method == 'weighted'


def test_ensemble_metamodel_fit_predict():
    """Test fitting and prediction with EnsembleMetamodel."""
    try:
        # Create test data
        parameter_set, y = create_test_data()

        # Create individual models
        rf_model = RandomForestMetamodel(n_estimators=10)
        gam_model = GAMMetamodel(n_splines=5)
        models = [rf_model, gam_model]

        # Create and fit ensemble
        ensemble = EnsembleMetamodel(models, method='mean')
        ensemble.fit(parameter_set, y)

        # Test prediction
        predictions = ensemble.predict(parameter_set)
        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))

        # Test scoring
        score = ensemble.score(parameter_set, y)
        assert isinstance(score, float)
        assert score > -1.0  # Even a poor model should have R2 > -1

        # Test RMSE
        rmse = ensemble.rmse(parameter_set, y)
        assert isinstance(rmse, float)
        assert rmse >= 0

    except ImportError:
        pytest.skip("Required metamodels not available")


def test_ensemble_metamodel_methods():
    """Test different ensemble methods."""
    try:
        # Create test data
        parameter_set, y = create_test_data()

        # Create individual models
        rf_model = RandomForestMetamodel(n_estimators=10)
        gam_model = GAMMetamodel(n_splines=5)
        models = [rf_model, gam_model]

        # Test mean ensemble
        ensemble_mean = EnsembleMetamodel(models, method='mean')
        ensemble_mean.fit(parameter_set, y)
        pred_mean = ensemble_mean.predict(parameter_set)

        # Test median ensemble
        ensemble_median = EnsembleMetamodel(models, method='median')
        ensemble_median.fit(parameter_set, y)
        pred_median = ensemble_median.predict(parameter_set)

        # Test weighted ensemble
        ensemble_weighted = EnsembleMetamodel(models, method='weighted')
        ensemble_weighted.fit(parameter_set, y)
        pred_weighted = ensemble_weighted.predict(parameter_set)

        # All predictions should have the same shape
        assert pred_mean.shape == pred_median.shape == pred_weighted.shape

        # Predictions should be finite
        assert np.all(np.isfinite(pred_mean))
        assert np.all(np.isfinite(pred_median))
        assert np.all(np.isfinite(pred_weighted))

    except ImportError:
        pytest.skip("Required metamodels not available")


def test_ensemble_metamodel_weighted():
    """Test weighted ensemble method."""
    try:
        # Create test data
        parameter_set, y = create_test_data()

        # Create individual models
        rf_model = RandomForestMetamodel(n_estimators=10)
        gam_model = GAMMetamodel(n_splines=5)
        models = [rf_model, gam_model]

        # Create and fit weighted ensemble
        ensemble = EnsembleMetamodel(models, method='weighted')
        ensemble.fit(parameter_set, y)

        # Check that weights were computed
        assert ensemble.weights is not None
        assert len(ensemble.weights) == len(models)
        assert abs(sum(ensemble.weights) - 1.0) < 1e-10  # Weights should sum to 1

        # Test prediction
        predictions = ensemble.predict(parameter_set)
        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))

    except ImportError:
        pytest.skip("Required metamodels not available")


def test_ensemble_metamodel_single_model():
    """Test ensemble with a single model."""
    try:
        # Create test data
        parameter_set, y = create_test_data()

        # Create single model
        rf_model = RandomForestMetamodel(n_estimators=10)
        models = [rf_model]

        # Create and fit ensemble
        ensemble = EnsembleMetamodel(models, method='mean')
        ensemble.fit(parameter_set, y)

        # Test prediction
        predictions = ensemble.predict(parameter_set)
        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))

    except ImportError:
        pytest.skip("Required metamodels not available")


def test_ensemble_metamodel_empty():
    """Test ensemble with no models."""
    ensemble = EnsembleMetamodel([])

    # Create dummy ParameterSet
    n_samples = 10
    param_dict = {"param1": np.random.randn(n_samples)}
    dataset = xr.Dataset(
        {k: ("n_samples", v) for k, v in param_dict.items()},
        coords={"n_samples": np.arange(n_samples)}
    )
    parameter_set = ParameterSet(dataset=dataset)

    # Try to predict without any models
    with pytest.raises(RuntimeError, match="No models in the ensemble"):
        ensemble.predict(parameter_set)


def test_ensemble_metamodel_invalid_method():
    """Test ensemble with invalid method."""
    try:
        rf_model = RandomForestMetamodel()
        gam_model = GAMMetamodel()
        models = [rf_model, gam_model]
    except ImportError:
        pytest.skip("Required metamodels not available")

    ensemble = EnsembleMetamodel(models, method='invalid')

    # Create test data
    parameter_set, y = create_test_data()
    ensemble.fit(parameter_set, y)

    # Try to predict with invalid method
    with pytest.raises(ValueError, match="Unknown ensemble method"):
        ensemble.predict(parameter_set)


if __name__ == "__main__":
    test_ensemble_metamodel_initialization()
    test_ensemble_metamodel_fit_predict()
    test_ensemble_metamodel_methods()
    test_ensemble_metamodel_weighted()
    test_ensemble_metamodel_single_model()
    test_ensemble_metamodel_empty()
    test_ensemble_metamodel_invalid_method()
    print("All ensemble metamodel tests passed!")
