"""
Example script demonstrating metamodeling functionality.

This script shows how to use the various metamodels implemented in voiage,
including fitting, prediction, diagnostics, and cross-validation.
"""

import numpy as np
import xarray as xr

from voiage.schema import ParameterSet
from voiage.metamodels import (
    RandomForestMetamodel,
    GAMMetamodel,
    BARTMetamodel,
    calculate_diagnostics,
    cross_validate,
    compare_metamodels
)


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create sample parameters
    n_samples = 100
    param1 = np.random.rand(n_samples)
    param2 = np.random.rand(n_samples)
    
    # Create a simple target function with some noise
    # y = 2*x1 + 3*x2 + x1*x2 + noise
    y = 2 * param1 + 3 * param2 + param1 * param2 + 0.1 * np.random.randn(n_samples)
    
    # Create ParameterSet
    data = {
        "param1": ("n_samples", param1),
        "param2": ("n_samples", param2),
    }
    x = ParameterSet(dataset=xr.Dataset(data))
    
    return x, y


def demonstrate_metamodels():
    """Demonstrate the metamodeling functionality."""
    print("voiage Metamodeling Example")
    print("=" * 40)
    
    # Create sample data
    x, y = create_sample_data()
    print(f"Created sample data with {len(y)} samples and {len(x.parameter_names)} parameters")
    
    # Test RandomForest
    print("\n1. Testing RandomForest Metamodel:")
    rf_model = RandomForestMetamodel(n_estimators=50, random_state=42)
    rf_model.fit(x, y)
    rf_pred = rf_model.predict(x)
    print(f"   Prediction shape: {rf_pred.shape}")
    print(f"   R² score: {rf_model.score(x, y):.4f}")
    print(f"   RMSE: {rf_model.rmse(x, y):.4f}")
    
    # Test diagnostics
    rf_diagnostics = calculate_diagnostics(rf_model, x, y)
    print(f"   Diagnostics: R²={rf_diagnostics['r2']:.4f}, RMSE={rf_diagnostics['rmse']:.4f}")
    
    # Test GAM (if available)
    print("\n2. Testing GAM Metamodel:")
    try:
        gam_model = GAMMetamodel(n_splines=10)
        gam_model.fit(x, y)
        gam_pred = gam_model.predict(x)
        print(f"   Prediction shape: {gam_pred.shape}")
        print(f"   R² score: {gam_model.score(x, y):.4f}")
        print(f"   RMSE: {gam_model.rmse(x, y):.4f}")
    except Exception as e:
        print(f"   GAM not available or failed: {e}")
    
    # Test BART (if available)
    print("\n3. Testing BART Metamodel:")
    try:
        # Use smaller sample for BART to keep it fast
        x_small = ParameterSet(
            dataset=xr.Dataset({
                "param1": ("n_samples", x.parameters["param1"][:50]),
                "param2": ("n_samples", x.parameters["param2"][:50]),
            })
        )
        y_small = y[:50]
        
        bart_model = BARTMetamodel(num_trees=20)
        bart_model.fit(x_small, y_small)
        bart_pred = bart_model.predict(x_small)
        print(f"   Prediction shape: {bart_pred.shape}")
        print(f"   R² score: {bart_model.score(x_small, y_small):.4f}")
        print(f"   RMSE: {bart_model.rmse(x_small, y_small):.4f}")
    except Exception as e:
        print(f"   BART not available or failed: {e}")
    
    # Test cross-validation
    print("\n4. Cross-validation example:")
    try:
        cv_results = cross_validate(RandomForestMetamodel, x, y, cv_folds=3)
        print(f"   Cross-validation R²: {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
        print(f"   Cross-validation RMSE: {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f}")
    except Exception as e:
        print(f"   Cross-validation failed: {e}")
    
    # Test model comparison
    print("\n5. Model comparison example:")
    try:
        models = [RandomForestMetamodel]
        comparison = compare_metamodels(models, x, y, cv_folds=2)
        for model_name, results in comparison.items():
            if "error" not in results:
                print(f"   {model_name}: R²={results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
            else:
                print(f"   {model_name}: Error - {results['error']}")
    except Exception as e:
        print(f"   Model comparison failed: {e}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    demonstrate_metamodels()