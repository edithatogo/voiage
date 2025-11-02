"""
Test script demonstrating metamodeling functionality.
"""

import os
import sys

import numpy as np
import xarray as xr

# Add the voiage package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voiage.metamodels import (
    RandomForestMetamodel,
    calculate_diagnostics,
    cross_validate,
)
from voiage.schema import ParameterSet


def test_metamodeling_functionality():
    """Test the metamodeling functionality."""
    print("Testing voiage Metamodeling Functionality")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n_samples = 50  # Use smaller sample for faster testing
    param1 = np.random.rand(n_samples)
    param2 = np.random.rand(n_samples)

    # Create a simple target function with some noise
    y = 2 * param1 + 3 * param2 + param1 * param2 + 0.1 * np.random.randn(n_samples)

    # Create ParameterSet
    data = {
        "param1": ("n_samples", param1),
        "param2": ("n_samples", param2),
    }
    x = ParameterSet(dataset=xr.Dataset(data))

    print(f"Created sample data with {len(y)} samples and {len(x.parameter_names)} parameters")

    # Test RandomForest
    print("\n1. Testing RandomForest Metamodel:")
    rf_model = RandomForestMetamodel(n_estimators=10, random_state=42)
    rf_model.fit(x, y)
    rf_pred = rf_model.predict(x)
    print(f"   Prediction shape: {rf_pred.shape}")
    print(f"   R² score: {rf_model.score(x, y):.4f}")
    print(f"   RMSE: {rf_model.rmse(x, y):.4f}")

    # Test diagnostics
    rf_diagnostics = calculate_diagnostics(rf_model, x, y)
    print(f"   Diagnostics: R²={rf_diagnostics['r2']:.4f}, RMSE={rf_diagnostics['rmse']:.4f}")

    # Test cross-validation
    print("\n2. Cross-validation example:")
    cv_results = cross_validate(RandomForestMetamodel, x, y, cv_folds=3)
    print(f"   Cross-validation R²: {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
    print(f"   Cross-validation RMSE: {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_metamodeling_functionality()
