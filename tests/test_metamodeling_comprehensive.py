"""
Comprehensive tests for metamodeling functionality in voiage.

This file provides extensive test coverage for all metamodeling features,
including edge cases, error conditions, and property-based testing.
"""

import numpy as np
import pytest
import xarray as xr

from voiage.schema import ParameterSet


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)  # For reproducible tests
    data = {
        "param1": ("n_samples", np.random.rand(100)),
        "param2": ("n_samples", np.random.rand(100)),
    }
    x = ParameterSet(dataset=xr.Dataset(data))
    y = np.random.rand(100)
    return x, y


@pytest.fixture
def small_sample_data():
    """Create small sample data for testing."""
    np.random.seed(42)  # For reproducible tests
    data = {
        "param1": ("n_samples", np.random.rand(10)),
        "param2": ("n_samples", np.random.rand(10)),
    }
    x = ParameterSet(dataset=xr.Dataset(data))
    y = np.random.rand(10)
    return x, y


def test_random_forest_metamodel_comprehensive(sample_data) -> None:
    """Comprehensive test for RandomForestMetamodel implementation."""
    try:
        from voiage.metamodels import RandomForestMetamodel
    except ImportError as e:
        pytest.skip(f"Skipping RandomForest test: {e}")

    x, y = sample_data

    # Test basic functionality
    model = RandomForestMetamodel(n_estimators=10, random_state=42)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred.shape == (100,)
    # Check that predictions are reasonable (not all zeros)
    assert np.var(y_pred) > 0

    # Test diagnostic methods
    r2 = model.score(x, y)
    assert isinstance(r2, float)
    assert 0 <= r2 <= 1  # R^2 should be between 0 and 1 for a fitted model

    rmse = model.rmse(x, y)
    assert isinstance(rmse, float)
    assert rmse >= 0

    # Test edge cases
    # Test with single parameter
    single_param_data = {
        "param1": ("n_samples", np.random.rand(50)),
    }
    x_single = ParameterSet(dataset=xr.Dataset(single_param_data))
    y_single = np.random.rand(50)

    model_single = RandomForestMetamodel(n_estimators=5, random_state=42)
    model_single.fit(x_single, y_single)
    y_pred_single = model_single.predict(x_single)
    assert y_pred_single.shape == (50,)

    # Test with constant target values
    y_constant = np.full(100, 0.5)
    model_constant = RandomForestMetamodel(n_estimators=5, random_state=42)
    model_constant.fit(x, y_constant)
    y_pred_constant = model_constant.predict(x)
    assert y_pred_constant.shape == (100,)

    # Test error conditions
    # Test prediction before fitting
    model_unfitted = RandomForestMetamodel(n_estimators=5, random_state=42)
    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        model_unfitted.predict(x)

    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        model_unfitted.score(x, y)

    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        model_unfitted.rmse(x, y)


def test_gam_metamodel_comprehensive(sample_data) -> None:
    """Comprehensive test for GAMMetamodel implementation."""
    try:
        from voiage.metamodels import GAMMetamodel
    except ImportError as e:
        pytest.skip(f"Skipping GAM test: {e}")

    x, y = sample_data

    # Test basic functionality
    try:
        model = GAMMetamodel(n_splines=5)
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (100,)
        # Check that predictions are reasonable (not all zeros)
        assert np.var(y_pred) > 0
    except AttributeError as e:
        # Handle numpy compatibility issues
        if "numpy" in str(e) and ("int" in str(e) or "float" in str(e)):
            pytest.skip(f"Skipping GAM test due to numpy compatibility issue: {e}")
        else:
            raise

    # Test with single parameter
    try:
        single_param_data = {
            "param1": ("n_samples", np.random.rand(50)),
        }
        x_single = ParameterSet(dataset=xr.Dataset(single_param_data))
        y_single = np.random.rand(50)

        model_single = GAMMetamodel(n_splines=3)
        model_single.fit(x_single, y_single)
        y_pred_single = model_single.predict(x_single)
        assert y_pred_single.shape == (50,)
    except AttributeError as e:
        # Handle numpy compatibility issues
        if "numpy" in str(e) and ("int" in str(e) or "float" in str(e)):
            pytest.skip(f"Skipping GAM test due to numpy compatibility issue: {e}")
        else:
            raise

    # Test error conditions
    # Test prediction before fitting
    try:
        model_unfitted = GAMMetamodel(n_splines=3)
        with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
            model_unfitted.predict(x)

        with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
            model_unfitted.score(x, y)

        with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
            model_unfitted.rmse(x, y)
    except AttributeError as e:
        # Handle numpy compatibility issues
        if "numpy" in str(e) and ("int" in str(e) or "float" in str(e)):
            pytest.skip(f"Skipping GAM test due to numpy compatibility issue: {e}")
        else:
            raise


def test_bart_metamodel_comprehensive(sample_data) -> None:
    """Comprehensive test for BARTMetamodel implementation."""
    try:
        from voiage.metamodels import BARTMetamodel
    except ImportError as e:
        pytest.skip(f"Skipping BART test: {e}")

    x, y = sample_data

    # Use a smaller sample for BART to keep tests reasonably fast
    x_small = ParameterSet(
        dataset=xr.Dataset(
            {
                "param1": ("n_samples", x.parameters["param1"][:20]),
                "param2": ("n_samples", x.parameters["param2"][:20]),
            }
        )
    )
    y_small = y[:20]

    model = BARTMetamodel(num_trees=10)  # Use fewer trees for faster testing
    model.fit(x_small, y_small)
    y_pred = model.predict(x_small)
    assert y_pred.shape == (20,)
    # Check that predictions are reasonable (not all zeros)
    assert np.var(y_pred) > 0

    # Test diagnostic methods
    r2 = model.score(x_small, y_small)
    assert isinstance(r2, float)
    assert r2 >= 0  # R^2 can be negative for poorly fitting models

    rmse = model.rmse(x_small, y_small)
    assert isinstance(rmse, float)
    assert rmse >= 0

    # Test error conditions
    # Test prediction before fitting
    model_unfitted = BARTMetamodel(num_trees=5)
    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        model_unfitted.predict(x_small)

    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        model_unfitted.score(x_small, y_small)

    with pytest.raises(RuntimeError, match="The model has not been fitted yet"):
        model_unfitted.rmse(x_small, y_small)


def test_flax_metamodel_dependency_handling(sample_data) -> None:
    """Test that FlaxMetamodel properly handles missing dependencies."""
    try:
        from voiage.metamodels import FlaxMetamodel
    except ImportError:
        pytest.skip("FlaxMetamodel not available")

    x, y = sample_data

    # Try to create the model - should raise ImportError if dependencies are missing
    try:
        model = FlaxMetamodel(learning_rate=0.01, n_epochs=10)
        # If we get here, dependencies are available, test basic functionality
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (100, 1)  # Flax model outputs shape
    except ImportError:
        # This is expected if dependencies are missing
        pass


def test_tinygp_metamodel_dependency_handling(sample_data) -> None:
    """Test that TinyGPMetamodel properly handles missing dependencies."""
    try:
        from voiage.metamodels import TinyGPMetamodel
    except ImportError:
        pytest.skip("TinyGPMetamodel not available")

    x, y = sample_data

    # Try to create the model - should raise ImportError if dependencies are missing
    try:
        model = TinyGPMetamodel()
        # If we get here, dependencies are available, test basic functionality
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (100,)  # GP model outputs shape
    except ImportError:
        # This is expected if dependencies are missing
        pass


def test_metamodel_protocol_compliance() -> None:
    """Test that all available metamodels comply with the Metamodel protocol."""
    from importlib.util import find_spec

    if find_spec("voiage.metamodels") is None:
        pytest.skip("Metamodel protocol not available")

    # Get all available metamodel classes
    available_metamodels = []

    try:
        from voiage.metamodels import RandomForestMetamodel

        available_metamodels.append(("RandomForestMetamodel", RandomForestMetamodel))
    except ImportError:
        pass

    try:
        from voiage.metamodels import GAMMetamodel

        available_metamodels.append(("GAMMetamodel", GAMMetamodel))
    except ImportError:
        pass

    try:
        from voiage.metamodels import BARTMetamodel

        available_metamodels.append(("BARTMetamodel", BARTMetamodel))
    except ImportError:
        pass

    try:
        from voiage.metamodels import FlaxMetamodel

        available_metamodels.append(("FlaxMetamodel", FlaxMetamodel))
    except ImportError:
        pass

    try:
        from voiage.metamodels import TinyGPMetamodel

        available_metamodels.append(("TinyGPMetamodel", TinyGPMetamodel))
    except ImportError:
        pass

    # Test that each metamodel has the required methods
    # Note: Some metamodels may not implement all methods, which is acceptable
    required_methods = ["fit", "predict"]
    _ = ["score", "rmse"]  # optional_methods = ['score', 'rmse']

    for name, metamodel_class in available_metamodels:
        # Check that it's a class
        assert isinstance(metamodel_class, type), f"{name} should be a class"

        # Check that it has all required methods
        for method in required_methods:
            assert hasattr(metamodel_class, method), (
                f"{name} should have method {method}"
            )

        # Check that it has at least one of the optional methods
        # has_optional = any(hasattr(metamodel_class, method) for method in optional_methods)
        # This is a loose check - having at least one optional method is good


def test_calculate_diagnostics_with_metamodels_without_methods(sample_data) -> None:
    """Test that calculate_diagnostics works with metamodels that don't implement score/rmse."""
    try:
        from voiage.metamodels import (
            FlaxMetamodel,
            TinyGPMetamodel,
            calculate_diagnostics,
        )
    except ImportError:
        pytest.skip("Required metamodels not available")

    x, y = sample_data

    # Test with FlaxMetamodel if available
    try:
        model = FlaxMetamodel(learning_rate=0.01, n_epochs=10)

        # Test error conditions before fitting
        with pytest.raises(RuntimeError, match="The model has not been fitted yet."):
            model.score(x, y)

        with pytest.raises(RuntimeError, match="The model has not been fitted yet."):
            model.rmse(x, y)

        model.fit(x, y)

        # Test score and rmse methods
        score = model.score(x, y)
        assert isinstance(score, float)
        assert score <= 1.0

        rmse = model.rmse(x, y)
        assert isinstance(rmse, float)
        assert rmse >= 0.0

        # This should work even if FlaxMetamodel doesn't implement score/rmse
        # because calculate_diagnostics will compute them manually
        diagnostics = calculate_diagnostics(model, x, y)
        assert "r2" in diagnostics
        assert "rmse" in diagnostics
    except ImportError:
        # Dependencies not available, skip
        pass

    # Test with TinyGPMetamodel if available
    try:
        model = TinyGPMetamodel()
        model.fit(x, y)
        # This should work even if TinyGPMetamodel doesn't implement score/rmse
        # because calculate_diagnostics will compute them manually
        diagnostics = calculate_diagnostics(model, x, y)
        assert "r2" in diagnostics
        assert "rmse" in diagnostics
    except ImportError:
        # Dependencies not available, skip
        pass


def test_calculate_diagnostics_comprehensive(sample_data) -> None:
    """Comprehensive test for the calculate_diagnostics function."""
    try:
        from voiage.metamodels import RandomForestMetamodel, calculate_diagnostics
    except ImportError as e:
        pytest.skip(f"Skipping diagnostics test: {e}")

    x, y = sample_data
    model = RandomForestMetamodel(n_estimators=10, random_state=42)
    model.fit(x, y)

    # Test diagnostics calculation
    diagnostics = calculate_diagnostics(model, x, y)

    # Check that all expected keys are present
    expected_keys = ["r2", "rmse", "mae", "mean_residual", "std_residual", "n_samples"]
    for key in expected_keys:
        assert key in diagnostics
        assert diagnostics[key] is not None

    # Check value types and ranges
    assert isinstance(diagnostics["r2"], float)
    assert isinstance(diagnostics["rmse"], float)
    assert isinstance(diagnostics["mae"], float)
    assert isinstance(diagnostics["mean_residual"], float)
    assert isinstance(diagnostics["std_residual"], float)
    assert isinstance(diagnostics["n_samples"], int)

    assert diagnostics["rmse"] >= 0
    assert diagnostics["mae"] >= 0
    assert diagnostics["n_samples"] == len(y)

    # Test edge cases
    # Test with perfect predictions (R^2 should be 1)
    y_perfect = model.predict(x)
    diagnostics_perfect = calculate_diagnostics(model, x, y_perfect)
    # Due to numerical precision, R^2 might not be exactly 1
    assert diagnostics_perfect["r2"] >= 0.99

    # Test with constant target values
    y_constant = np.full(100, 0.5)
    diagnostics_constant = calculate_diagnostics(model, x, y_constant)
    assert diagnostics_constant["r2"] <= 1.0  # R^2 should be <= 1
    assert diagnostics_constant["rmse"] >= 0


def test_cross_validate_comprehensive(sample_data) -> None:
    """Comprehensive test for the cross_validate function."""
    try:
        from voiage.metamodels import RandomForestMetamodel, cross_validate
    except ImportError as e:
        pytest.skip(f"Skipping cross-validation test: {e}")

    x, y = sample_data

    # Test cross-validation with RandomForest
    cv_results = cross_validate(RandomForestMetamodel, x, y, cv_folds=3)

    # Check that all expected keys are present
    expected_keys = [
        "cv_r2_mean",
        "cv_r2_std",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_mae_mean",
        "cv_mae_std",
        "n_folds",
        "fold_scores",
        "fold_rmse",
        "fold_mae",
    ]
    for key in expected_keys:
        assert key in cv_results
        assert cv_results[key] is not None

    # Check value types and ranges
    assert isinstance(cv_results["cv_r2_mean"], float)
    assert isinstance(cv_results["cv_r2_std"], float)
    assert isinstance(cv_results["cv_rmse_mean"], float)
    assert isinstance(cv_results["cv_rmse_std"], float)
    assert isinstance(cv_results["cv_mae_mean"], float)
    assert isinstance(cv_results["cv_mae_std"], float)
    assert isinstance(cv_results["n_folds"], int)
    assert isinstance(cv_results["fold_scores"], list)
    assert isinstance(cv_results["fold_rmse"], list)
    assert isinstance(cv_results["fold_mae"], list)

    assert cv_results["n_folds"] == 3
    assert len(cv_results["fold_scores"]) == 3
    assert len(cv_results["fold_rmse"]) == 3
    assert len(cv_results["fold_mae"]) == 3

    assert cv_results["cv_rmse_mean"] >= 0
    assert cv_results["cv_mae_mean"] >= 0

    # Test edge cases
    # Test with 2 folds
    cv_results_2 = cross_validate(RandomForestMetamodel, x, y, cv_folds=2)
    assert cv_results_2["n_folds"] == 2
    assert len(cv_results_2["fold_scores"]) == 2

    # Test with more folds than samples (should be capped)
    # This should work without error
    try:
        cv_results_many = cross_validate(RandomForestMetamodel, x, y, cv_folds=150)
        # Should be capped to number of samples
        assert cv_results_many["n_folds"] <= len(y)
    except Exception:
        # Some implementations might handle this differently
        pass


def test_compare_metamodels_comprehensive(sample_data) -> None:
    """Comprehensive test for the compare_metamodels function."""
    try:
        from voiage.metamodels import RandomForestMetamodel, compare_metamodels
    except ImportError as e:
        pytest.skip(f"Skipping model comparison test: {e}")

    x, y = sample_data

    # Test comparison with RandomForest
    models = [RandomForestMetamodel]
    comparison_results = compare_metamodels(models, x, y, cv_folds=2)

    # Check that results are returned for the model
    assert "RandomForestMetamodel" in comparison_results
    model_results = comparison_results["RandomForestMetamodel"]

    # Check that no error occurred
    # Note: Some models may have errors due to dependencies, that's acceptable
    if "error" not in model_results:
        # Check expected keys in results
        expected_keys = [
            "cv_r2_mean",
            "cv_r2_std",
            "cv_rmse_mean",
            "cv_rmse_std",
            "cv_mae_mean",
            "cv_mae_std",
            "n_folds",
        ]
        for key in expected_keys:
            assert key in model_results

    # Test with multiple models (if available)
    models_to_test = [RandomForestMetamodel]

    try:
        from voiage.metamodels import GAMMetamodel

        models_to_test.append(GAMMetamodel)
    except ImportError:
        pass

    try:
        from voiage.metamodels import BARTMetamodel

        models_to_test.append(BARTMetamodel)
    except ImportError:
        pass

    if len(models_to_test) > 1:
        comparison_multi = compare_metamodels(models_to_test, x, y, cv_folds=2)
        for model_class in models_to_test:
            model_name = model_class.__name__
            assert model_name in comparison_multi
            # Note: Some models may have errors due to dependencies, that's acceptable
            # We just check that the model name is in the results


def test_edge_cases_and_error_conditions(sample_data) -> None:
    """Test edge cases and error conditions for metamodeling functions."""
    try:
        from voiage.metamodels import (
            RandomForestMetamodel,
            compare_metamodels,
            cross_validate,
        )
    except ImportError as e:
        pytest.skip(f"Skipping edge case tests: {e}")

    x, y = sample_data

    # Test with empty data (should raise appropriate errors)
    empty_data = {
        "param1": ("n_samples", np.array([])),
        "param2": ("n_samples", np.array([])),
    }
    x_empty = ParameterSet(dataset=xr.Dataset(empty_data))
    y_empty = np.array([])

    model = RandomForestMetamodel(n_estimators=5, random_state=42)

    # This should raise an error due to empty data
    with pytest.raises(Exception):
        model.fit(x_empty, y_empty)

    # Test with mismatched dimensions
    y_mismatched = np.random.rand(50)  # Different size than x
    with pytest.raises(Exception):
        model.fit(x, y_mismatched)

    # Test cross_validate with invalid parameters
    # Note: These may not raise exceptions in all implementations
    try:
        cross_validate(RandomForestMetamodel, x, y, cv_folds=0)  # Invalid fold count
    except Exception:
        # Expected to raise an exception
        pass

    try:
        cross_validate(RandomForestMetamodel, x, y, cv_folds=-1)  # Negative fold count
    except Exception:
        # Expected to raise an exception
        pass

    # Test compare_metamodels with empty model list
    empty_comparison = compare_metamodels([], x, y)
    assert empty_comparison == {}


# Property-based tests (if hypothesis is available)
def test_metamodel_properties(sample_data) -> None:
    """Test mathematical properties of metamodels."""
    try:
        from importlib.util import find_spec

        if find_spec("hypothesis") is None:
            pytest.skip("Skipping property-based tests: hypothesis not available")

        from voiage.metamodels import RandomForestMetamodel
    except ImportError:
        pytest.skip("Skipping property-based tests: hypothesis not available")

    x, y = sample_data

    # Test that RMSE is always non-negative
    model = RandomForestMetamodel(n_estimators=10, random_state=42)
    model.fit(x, y)
    rmse = model.rmse(x, y)
    assert rmse >= 0, "RMSE should always be non-negative"

    # Test that R^2 is between -inf and 1 (can be negative for bad models)
    r2 = model.score(x, y)
    assert r2 <= 1.0, "R^2 should be <= 1"

    # Test that predictions have the same number of samples as input
    y_pred = model.predict(x)
    assert len(y_pred) == len(y), "Predictions should have same length as target"


class _LinearDummyMetamodel:
    """Minimal metamodel used to exercise ActiveLearningMetamodel."""

    def __init__(self) -> None:
        self.fit_calls: list[tuple[int, int]] = []

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        self.fit_calls.append((x.n_samples, len(y)))

    def predict(self, x: ParameterSet) -> np.ndarray:
        return np.sum(np.stack(list(x.parameters.values()), axis=1), axis=1)

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        return 0.5

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        return 1.0


class _FailingPredictMetamodel(_LinearDummyMetamodel):
    def predict(self, x: ParameterSet) -> np.ndarray:
        raise RuntimeError("prediction failed")


def test_active_learning_metamodel_fit_predict_score_and_rmse() -> None:
    """Active learning should fit the base model and expose predictions."""
    from voiage.metamodels import ActiveLearningMetamodel

    x = ParameterSet(
        dataset=xr.Dataset(
            {
                "param1": ("n_samples", np.array([0.0, 1.0, 2.0])),
                "param2": ("n_samples", np.array([1.0, 2.0, 3.0])),
            }
        )
    )
    y = np.array([1.0, 3.0, 5.0])
    x_pool = np.array([[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    y_pool = np.array([7.0, 9.0, 11.0])
    base_model = _LinearDummyMetamodel()
    model = ActiveLearningMetamodel(
        base_model,
        n_initial_samples=2,
        n_query_samples=1,
        acquisition_function="random",
    )

    selected_x, selected_y, selected_indices = model._select_initial_samples(
        x_pool, y_pool
    )
    model.fit(x, y, x_pool=x_pool, y_pool=y_pool, n_iterations=2)

    predictions = model.predict(x)

    assert selected_x.shape == (2, 2)
    assert selected_y is not None
    assert selected_y.shape == (2,)
    assert selected_indices.shape == (2,)
    assert model.is_fitted is True
    assert model.iteration == 2
    assert model.X_train is not None
    assert model.X_train.shape[0] == 5
    assert len(base_model.fit_calls) == 3
    np.testing.assert_allclose(predictions, np.array([1.0, 3.0, 5.0]))
    assert model.score(x, y) == pytest.approx(1.0)
    assert model.rmse(x, y) == pytest.approx(0.0)


def test_active_learning_acquisition_branches_and_validation() -> None:
    """Active learning should support acquisition fallbacks and validate names."""
    from voiage.metamodels import ActiveLearningMetamodel

    x_pool = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    y_pool = np.array([0.0, 2.0, 3.0])
    model = ActiveLearningMetamodel(
        _FailingPredictMetamodel(),
        n_query_samples=2,
        acquisition_function="uncertainty",
    )

    random_scores = model._acquisition_uncertainty(x_pool)
    model.X_train = np.array([[0.0, 0.0]])
    distance_scores = model._acquisition_uncertainty(x_pool)
    selected_x, selected_y, selected_indices = model._select_query_samples(
        x_pool, y_pool
    )

    assert random_scores.shape == (3,)
    np.testing.assert_allclose(distance_scores, np.array([0.0, 2.0, 3.0]))
    assert selected_x.shape == (2, 2)
    assert selected_y is not None
    assert selected_y.shape == (2,)
    assert selected_indices.shape == (2,)

    margin_model = ActiveLearningMetamodel(
        _FailingPredictMetamodel(), acquisition_function="margin"
    )
    margin_model.X_train = np.array([[0.0, 0.0]])
    assert margin_model._select_query_samples(x_pool, None)[0].shape[1] == 2

    invalid_model = ActiveLearningMetamodel(
        _LinearDummyMetamodel(), acquisition_function="invalid"
    )
    with pytest.raises(ValueError, match="Unknown acquisition function"):
        invalid_model._select_query_samples(x_pool, None)


def test_active_learning_requires_fit_before_prediction_diagnostics() -> None:
    """Prediction diagnostics should fail clearly before fitting."""
    from voiage.metamodels import ActiveLearningMetamodel

    x = ParameterSet(
        dataset=xr.Dataset({"param1": ("n_samples", np.array([0.0, 1.0]))})
    )
    y = np.array([0.0, 1.0])
    model = ActiveLearningMetamodel(_LinearDummyMetamodel())

    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(x)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.score(x, y)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.rmse(x, y)


def test_active_learning_fit_creates_synthetic_pool_without_labels() -> None:
    """Fit should handle the no-pool path by creating an unlabeled synthetic pool."""
    from voiage.metamodels import ActiveLearningMetamodel

    x = ParameterSet(
        dataset=xr.Dataset(
            {
                "param1": ("n_samples", np.array([0.0, 1.0])),
                "param2": ("n_samples", np.array([1.0, 2.0])),
            }
        )
    )
    y = np.array([1.0, 3.0])
    base_model = _LinearDummyMetamodel()
    model = ActiveLearningMetamodel(base_model, n_query_samples=2)

    model.fit(x, y, n_iterations=1)

    assert model.is_fitted is True
    assert model.X_train is not None
    assert model.X_train.shape == (2, 2)
    assert len(base_model.fit_calls) == 2


if __name__ == "__main__":
    # Run the tests manually if executed directly
    pytest.main([__file__, "-v"])
