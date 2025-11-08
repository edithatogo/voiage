"""Tests for the active learning metamodel."""

import numpy as np
import pytest
import xarray as xr

from voiage.metamodels import ActiveLearningMetamodel, RandomForestMetamodel
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


def test_active_learning_metamodel_initialization():
    """Test that ActiveLearningMetamodel can be initialized."""
    try:
        # Create base model
        base_model = RandomForestMetamodel()
    except ImportError:
        pytest.skip("Required metamodels not available")

    # Test initialization with different parameters
    active_model = ActiveLearningMetamodel(base_model)
    assert active_model.base_model == base_model
    assert active_model.n_initial_samples == 10
    assert active_model.n_query_samples == 5
    assert active_model.acquisition_function == 'uncertainty'

    # Test with custom parameters
    active_model_custom = ActiveLearningMetamodel(
        base_model,
        n_initial_samples=20,
        n_query_samples=3,
        acquisition_function='random'
    )
    assert active_model_custom.n_initial_samples == 20
    assert active_model_custom.n_query_samples == 3
    assert active_model_custom.acquisition_function == 'random'


def test_active_learning_metamodel_fit_predict():
    """Test fitting and prediction with ActiveLearningMetamodel."""
    try:
        # Create base model
        base_model = RandomForestMetamodel(n_estimators=10)
    except ImportError:
        pytest.skip("Required metamodels not available")

    # Create active learning model
    active_model = ActiveLearningMetamodel(base_model)

    # Create initial training data
    x_train, y_train = create_test_data(50, 3)

    # Create pool of data for active learning
    x_pool, y_pool = create_test_data(200, 3)
    x_pool_array = np.array(list(x_pool.parameters.values())).T

    # Fit the model
    active_model.fit(x_train, y_train, x_pool_array, y_pool, n_iterations=2)

    # Test prediction
    predictions = active_model.predict(x_train)
    assert len(predictions) == len(y_train)
    assert np.all(np.isfinite(predictions))

    # Test scoring
    score = active_model.score(x_train, y_train)
    assert isinstance(score, float)
    assert score > -1.0  # Even a poor model should have R2 > -1

    # Test RMSE
    rmse = active_model.rmse(x_train, y_train)
    assert isinstance(rmse, float)
    assert rmse >= 0


def test_active_learning_metamodel_predict_before_fit():
    """Test that predict raises an error if called before fit."""
    try:
        # Create base model
        base_model = RandomForestMetamodel()
    except ImportError:
        pytest.skip("Required metamodels not available")

    # Create active learning model
    active_model = ActiveLearningMetamodel(base_model)

    # Create test data
    x_test, y_test = create_test_data(10, 3)

    # Try to predict without fitting
    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        active_model.predict(x_test)

    # Try to score without fitting
    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        active_model.score(x_test, y_test)

    # Try to calculate RMSE without fitting
    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        active_model.rmse(x_test, y_test)


def test_active_learning_metamodel_acquisition_functions():
    """Test different acquisition functions."""
    try:
        # Create base model
        base_model = RandomForestMetamodel(n_estimators=10)
    except ImportError:
        pytest.skip("Required metamodels not available")

    # Test different acquisition functions
    acquisition_functions = ['uncertainty', 'random', 'margin']

    for acquisition_func in acquisition_functions:
        active_model = ActiveLearningMetamodel(
            base_model,
            acquisition_function=acquisition_func
        )

        # Create initial training data
        x_train, y_train = create_test_data(20, 3)

        # Create pool of data
        x_pool, y_pool = create_test_data(100, 3)
        x_pool_array = np.array(list(x_pool.parameters.values())).T

        # Fit the model
        active_model.fit(x_train, y_train, x_pool_array, y_pool, n_iterations=1)

        # Test prediction
        predictions = active_model.predict(x_train)
        assert len(predictions) == len(y_train)
        assert np.all(np.isfinite(predictions))


def test_active_learning_metamodel_invalid_acquisition_function():
    """Test active learning with invalid acquisition function."""
    try:
        # Create base model
        base_model = RandomForestMetamodel()
    except ImportError:
        pytest.skip("Required metamodels not available")

    # Create active learning model with invalid acquisition function
    active_model = ActiveLearningMetamodel(
        base_model,
        acquisition_function='invalid'
    )

    # Create initial training data
    x_train, y_train = create_test_data(20, 3)

    # Create pool of data
    x_pool, y_pool = create_test_data(100, 3)
    x_pool_array = np.array(list(x_pool.parameters.values())).T

    # Try to fit the model - should raise ValueError
    with pytest.raises(ValueError, match="Unknown acquisition function"):
        active_model.fit(x_train, y_train, x_pool_array, y_pool, n_iterations=1)


def test_active_learning_metamodel_synthetic_pool():
    """Test active learning with synthetic pool (no true labels)."""
    try:
        # Create base model
        base_model = RandomForestMetamodel(n_estimators=10)
    except ImportError:
        pytest.skip("Required metamodels not available")

    # Create active learning model
    active_model = ActiveLearningMetamodel(base_model)

    # Create initial training data
    x_train, y_train = create_test_data(20, 3)

    # Create pool data (without labels)
    x_pool, _ = create_test_data(100, 3)
    x_pool_array = np.array(list(x_pool.parameters.values())).T

    # Fit the model without providing pool labels
    active_model.fit(x_train, y_train, x_pool_array, n_iterations=1)

    # Test prediction
    predictions = active_model.predict(x_train)
    assert len(predictions) == len(y_train)
    assert np.all(np.isfinite(predictions))


if __name__ == "__main__":
    test_active_learning_metamodel_initialization()
    test_active_learning_metamodel_fit_predict()
    test_active_learning_metamodel_predict_before_fit()
    test_active_learning_metamodel_acquisition_functions()
    test_active_learning_metamodel_invalid_acquisition_function()
    test_active_learning_metamodel_synthetic_pool()
    print("All active learning metamodel tests passed!")
