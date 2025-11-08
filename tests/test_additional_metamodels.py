# tests/test_additional_metamodels.py

import numpy as np
import pytest
import xarray as xr

from voiage.schema import ParameterSet


@pytest.fixture
def sample_data():
    np.random.seed(42)  # For reproducible tests
    data = {
        "param1": ("n_samples", np.random.rand(100)),
        "param2": ("n_samples", np.random.rand(100)),
    }
    x = ParameterSet(dataset=xr.Dataset(data))
    y = np.random.rand(100)
    return x, y


def test_random_forest_metamodel(sample_data):
    """Test RandomForestMetamodel implementation."""
    try:
        from voiage.metamodels import RandomForestMetamodel
    except ImportError as e:
        pytest.skip(f"Skipping RandomForest test: {e}")

    x, y = sample_data
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


def test_gam_metamodel(sample_data):
    """Test GAMMetamodel implementation."""
    try:
        from voiage.metamodels import GAMMetamodel
    except ImportError as e:
        pytest.skip(f"Skipping GAM test: {e}")

    x, y = sample_data
    try:
        model = GAMMetamodel(n_splines=5)
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (100,)
        # Check that predictions are reasonable (not all zeros)
        assert np.var(y_pred) > 0
    except AttributeError as e:
        # Handle numpy compatibility issues
        if "numpy" in str(e) and "int" in str(e):
            pytest.skip(f"Skipping GAM test due to numpy compatibility issue: {e}")
        else:
            raise


def test_bart_metamodel(sample_data):
    """Test BARTMetamodel implementation."""
    try:
        from voiage.metamodels import BARTMetamodel
    except ImportError as e:
        pytest.skip(f"Skipping BART test: {e}")

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

    # Test diagnostic methods
    r2 = model.score(x_small, y_small)
    assert isinstance(r2, float)
    assert r2 >= 0  # R^2 can be negative for poorly fitting models

    rmse = model.rmse(x_small, y_small)
    assert isinstance(rmse, float)
    assert rmse >= 0


def test_metamodel_protocol():
    """Test that available metamodels follow the Metamodel protocol."""
    try:
        from voiage.metamodels import Metamodel  # noqa: F401
    except ImportError:
        pytest.skip("Metamodel protocol not available")

    # Check that available metamodels implement the required methods
    available_metamodels = []

    try:
        from voiage.metamodels import RandomForestMetamodel
        available_metamodels.append(RandomForestMetamodel)
    except ImportError:
        pass

    try:
        from voiage.metamodels import GAMMetamodel
        available_metamodels.append(GAMMetamodel)
    except ImportError:
        pass

    try:
        from voiage.metamodels import BARTMetamodel
        available_metamodels.append(BARTMetamodel)
    except ImportError:
        pass

    for metamodel_class in available_metamodels:
        # Check that it's a class that can be instantiated
        assert isinstance(metamodel_class, type)

        # Check that it has the required methods
        assert hasattr(metamodel_class, 'fit')
        assert hasattr(metamodel_class, 'predict')


def test_calculate_diagnostics(sample_data):
    """Test the calculate_diagnostics function."""
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


def test_cross_validate(sample_data):
    """Test the cross_validate function."""
    try:
        from voiage.metamodels import RandomForestMetamodel, cross_validate
    except ImportError as e:
        pytest.skip(f"Skipping cross-validation test: {e}")

    x, y = sample_data

    # Test cross-validation with RandomForest
    cv_results = cross_validate(RandomForestMetamodel, x, y, cv_folds=3)

    # Check that all expected keys are present
    expected_keys = [
        "cv_r2_mean", "cv_r2_std",
        "cv_rmse_mean", "cv_rmse_std",
        "cv_mae_mean", "cv_mae_std",
        "n_folds", "fold_scores", "fold_rmse", "fold_mae"
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


def test_compare_metamodels(sample_data):
    """Test the compare_metamodels function."""
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
    assert "error" not in model_results

    # Check expected keys in results
    expected_keys = [
        "cv_r2_mean", "cv_r2_std",
        "cv_rmse_mean", "cv_rmse_std",
        "cv_mae_mean", "cv_mae_std",
        "n_folds"
    ]
    for key in expected_keys:
        assert key in model_results


if __name__ == "__main__":
    # Run the tests manually if executed directly
    pytest.main([__file__, "-v"])
