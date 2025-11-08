"""Comprehensive tests for metamodels module to increase coverage above 95%."""

import numpy as np
import pytest

from voiage.metamodels import (
    MLP,
    BARTMetamodel,
    EnsembleMetamodel,
    GAMMetamodel,
    RandomForestMetamodel,
)
from voiage.schema import ParameterSet


class TestRandomForestMetamodelComprehensive:
    """Comprehensive tests for RandomForestMetamodel to achieve >95% coverage."""

    def test_rf_metamodel_basic_functionality(self):
        """Test basic random forest metamodel functionality."""
        # Create training data as ParameterSet
        X_train_data = {
            "param1": np.random.rand(100).astype(np.float64),  # 100 samples for param1
            "param2": np.random.rand(100).astype(np.float64),  # 100 samples for param2
            "param3": np.random.rand(100).astype(np.float64)   # 100 samples for param3
        }
        X_train = ParameterSet.from_numpy_or_dict(X_train_data)
        y_train = np.random.rand(100).astype(np.float64)     # 100 outcomes

        # Test initialization
        rf_meta = RandomForestMetamodel(n_estimators=10)  # Use smaller number for faster testing

        # Test fitting
        rf_meta.fit(X_train, y_train)

        # Test prediction with test ParameterSet
        X_test_data = {
            "param1": np.random.rand(10).astype(np.float64),  # 10 test samples
            "param2": np.random.rand(10).astype(np.float64),  # 10 test samples
            "param3": np.random.rand(10).astype(np.float64)   # 10 test samples
        }
        X_test = ParameterSet.from_numpy_or_dict(X_test_data)
        predictions = rf_meta.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 10  # Same number of test samples
        assert all(np.isfinite(pred) for pred in predictions)  # All predictions should be finite

        # Test scoring
        score = rf_meta.score(X_test, predictions)
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ Random Forest metamodel basic functionality works")

    def test_rf_metamodel_validation_errors(self):
        """Test random forest metamodel with validation errors."""
        rf_meta = RandomForestMetamodel()

        # Test with mismatched dimensions
        X_train = np.random.rand(10, 3)  # 3 parameters
        y_train = np.random.rand(10)     # 10 outcomes

        X_test = np.random.rand(5, 2)    # Different number of parameters (should fail)

        rf_meta.fit(X_train, y_train)

        with pytest.raises(ValueError):
            rf_meta.predict(X_test)

        print("✅ Random Forest metamodel validation errors handled correctly")

    def test_rf_metamodel_edge_cases(self):
        """Test random forest metamodel with edge cases."""
        # Single sample, single parameter
        X_single = np.array([[0.5]], dtype=np.float64)
        y_single = np.array([1.0], dtype=np.float64)

        rf_meta = RandomForestMetamodel(n_estimators=5)
        rf_meta.fit(X_single, y_single)

        X_test_single = np.array([[0.6]], dtype=np.float64)
        pred_single = rf_meta.predict(X_test_single)

        assert isinstance(pred_single, np.ndarray)
        assert pred_single.shape == (1,)

        # Many samples, single parameter
        X_many = np.random.rand(1000, 1).astype(np.float64)
        y_many = np.random.rand(1000).astype(np.float64)

        rf_meta_many = RandomForestMetamodel(n_estimators=5)
        rf_meta_many.fit(X_many, y_many)

        X_test_many = np.random.rand(5, 1).astype(np.float64)
        pred_many = rf_meta_many.predict(X_test_many)

        assert pred_many.shape == (5,)

        # Single sample, many parameters
        X_many_params = np.random.rand(1, 50).astype(np.float64)
        y_one = np.array([1.0], dtype=np.float64)

        rf_meta_many_params = RandomForestMetamodel(n_estimators=5)
        rf_meta_many_params.fit(X_many_params, y_one)

        X_test_many_params = np.random.rand(1, 50).astype(np.float64)
        pred_many_params = rf_meta_many_params.predict(X_test_many_params)

        assert pred_many_params.shape == (1,)

        print("✅ Random Forest metamodel edge cases handled correctly")

    def test_rf_metamodel_hyperparameters(self):
        """Test random forest metamodel with different hyperparameters."""
        X_train = np.random.rand(50, 5).astype(np.float64)
        y_train = np.random.rand(50).astype(np.float64)

        # Test with different hyperparameters
        rf_meta = RandomForestMetamodel(
            n_estimators=20,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

        rf_meta.fit(X_train, y_train)

        # Test with different test data
        X_test = np.random.rand(10, 5).astype(np.float64)
        predictions = rf_meta.predict(X_test)

        assert predictions.shape == (10,)

        # Test scoring
        score = rf_meta.score(X_test, np.random.rand(10).astype(np.float64))
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ Random Forest metamodel hyperparameters work correctly")

    def test_rf_metamodel_rmse_calculation(self):
        """Test RMSE calculation for random forest metamodel."""
        X_train = np.random.rand(30, 2).astype(np.float64)
        y_train = np.random.rand(30).astype(np.float64)

        rf_meta = RandomForestMetamodel(n_estimators=10)
        rf_meta.fit(X_train, y_train)

        X_test = np.random.rand(10, 2).astype(np.float64)
        y_true = np.random.rand(10).astype(np.float64)

        # Calculate RMSE using the method
        rmse_score = rf_meta.rmse(X_test, y_true)
        assert isinstance(rmse_score, float)
        assert rmse_score >= 0.0  # RMSE should be non-negative

        print("✅ Random Forest metamodel RMSE calculation works correctly")


class TestGAMMetamodelComprehensive:
    """Comprehensive tests for GAMMetamodel to achieve >95% coverage."""

    def test_gam_metamodel_basic_functionality(self):
        """Test basic GAM metamodel functionality."""
        # Create training data
        X_train = np.random.rand(100, 3).astype(np.float64)
        y_train = np.random.rand(100).astype(np.float64)

        gam_meta = GAMMetamodel(max_iter=50)  # Lower for faster testing

        # Test fitting
        gam_meta.fit(X_train, y_train)

        # Test prediction
        X_test = np.random.rand(10, 3).astype(np.float64)
        predictions = gam_meta.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 10
        assert all(np.isfinite(pred) for pred in predictions)

        # Test scoring
        score = gam_meta.score(X_test, predictions)
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ GAM metamodel basic functionality works")

    def test_gam_metamodel_validation_errors(self):
        """Test GAM metamodel with validation errors."""
        gam_meta = GAMMetamodel()

        # Test with mismatched dimensions
        X_train = np.random.rand(10, 3).astype(np.float64)  # 3 parameters
        y_train = np.random.rand(10).astype(np.float64)     # 10 outcomes

        X_test = np.random.rand(5, 2).astype(np.float64)    # Different number of parameters (should fail)

        gam_meta.fit(X_train, y_train)

        with pytest.raises(ValueError):
            gam_meta.predict(X_test)

        print("✅ GAM metamodel validation errors handled correctly")

    def test_gam_metamodel_edge_cases(self):
        """Test GAM metamodel with edge cases."""
        # Single sample single parameter
        X_single = np.array([[0.5]], dtype=np.float64)
        y_single = np.array([1.0], dtype=np.float64)

        gam_meta = GAMMetamodel(max_iter=20)
        gam_meta.fit(X_single, y_single)

        X_test_single = np.array([[0.6]], dtype=np.float64)
        pred_single = gam_meta.predict(X_test_single)

        assert isinstance(pred_single, np.ndarray)
        assert pred_single.shape == (1,)

        # Test with many samples and parameters with different shapes
        X_many = np.random.rand(500, 2).astype(np.float64)
        y_many = np.random.rand(500).astype(np.float64)

        gam_meta_many = GAMMetamodel(max_iter=30)
        gam_meta_many.fit(X_many, y_many)

        X_test_many = np.random.rand(20, 2).astype(np.float64)
        pred_many = gam_meta_many.predict(X_test_many)

        assert pred_many.shape == (20,)

        print("✅ GAM metamodel edge cases handled correctly")

    def test_gam_metamodel_hyperparameters(self):
        """Test GAM metamodel with different hyperparameters."""
        X_train = np.random.rand(50, 4).astype(np.float64)
        y_train = np.random.rand(50).astype(np.float64)

        # Test with specific hyperparameters
        gam_meta = GAMMetamodel(
            max_iter=100,
            tol=1e-4,
        )

        gam_meta.fit(X_train, y_train)

        X_test = np.random.rand(15, 4).astype(np.float64)
        predictions = gam_meta.predict(X_test)

        assert predictions.shape == (15,)

        # Test score calculation
        score = gam_meta.score(X_test, np.random.rand(15).astype(np.float64))
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ GAM metamodel hyperparameters work correctly")

    def test_gam_metamodel_rmse_calculation(self):
        """Test RMSE calculation for GAM metamodel."""
        X_train = np.random.rand(40, 2).astype(np.float64)
        y_train = np.random.rand(40).astype(np.float64)

        gam_meta = GAMMetamodel(max_iter=50)
        gam_meta.fit(X_train, y_train)

        X_test = np.random.rand(12, 2).astype(np.float64)
        y_true = np.random.rand(12).astype(np.float64)

        rmse_score = gam_meta.rmse(X_test, y_true)
        assert isinstance(rmse_score, float)
        assert rmse_score >= 0.0

        print("✅ GAM metamodel RMSE calculation works correctly")


class TestBARTMetamodelComprehensive:
    """Comprehensive tests for BARTMetamodel to achieve >95% coverage."""

    def test_bart_metamodel_basic_functionality(self):
        """Test basic BART metamodel functionality."""
        # Create training data
        X_train = np.random.rand(80, 2).astype(np.float64)
        y_train = np.random.rand(80).astype(np.float64)

        bart_meta = BARTMetamodel(n_trees=10, n_chains=1)  # Smaller values for testing

        # Test fitting
        bart_meta.fit(X_train, y_train)

        # Test prediction
        X_test = np.random.rand(8, 2).astype(np.float64)
        predictions = bart_meta.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 8
        assert all(np.isfinite(pred) for pred in predictions)

        # Test scoring
        score = bart_meta.score(X_test, predictions)
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ BART metamodel basic functionality works")

    def test_bart_metamodel_validation_errors(self):
        """Test BART metamodel with validation errors."""
        bart_meta = BARTMetamodel()

        # Test with mismatched dimensions
        X_train = np.random.rand(10, 3).astype(np.float64)  # 3 parameters
        y_train = np.random.rand(10).astype(np.float64)     # 10 outcomes

        X_test = np.random.rand(5, 2).astype(np.float64)    # Different number of parameters (should fail)

        bart_meta.fit(X_train, y_train)

        with pytest.raises(ValueError):
            bart_meta.predict(X_test)

        print("✅ BART metamodel validation errors handled correctly")

    def test_bart_metamodel_edge_cases(self):
        """Test BART metamodel with edge cases."""
        # Single sample single parameter
        X_single = np.array([[0.5]], dtype=np.float64)
        y_single = np.array([1.0], dtype=np.float64)

        bart_meta = BARTMetamodel(n_trees=5, n_chains=1)
        bart_meta.fit(X_single, y_single)

        X_test_single = np.array([[0.6]], dtype=np.float64)
        pred_single = bart_meta.predict(X_test_single)

        assert isinstance(pred_single, np.ndarray)
        assert pred_single.shape == (1,)

        # Larger cases
        X_many = np.random.rand(200, 3).astype(np.float64)
        y_many = np.random.rand(200).astype(np.float64)

        bart_meta_many = BARTMetamodel(n_trees=20, n_chains=1)
        bart_meta_many.fit(X_many, y_many)

        X_test_many = np.random.rand(25, 3).astype(np.float64)
        pred_many = bart_meta_many.predict(X_test_many)

        assert pred_many.shape == (25,)

        print("✅ BART metamodel edge cases handled correctly")

    def test_bart_metamodel_hyperparameters(self):
        """Test BART metamodel with different hyperparameters."""
        X_train = np.random.rand(60, 3).astype(np.float64)
        y_train = np.random.rand(60).astype(np.float64)

        # Test with specific hyperparameters
        bart_meta = BARTMetamodel(
            n_trees=15,
            n_samples=100,
            n_burn=50,
            n_chains=1
        )

        bart_meta.fit(X_train, y_train)

        X_test = np.random.rand(15, 3).astype(np.float64)
        predictions = bart_meta.predict(X_test)

        assert predictions.shape == (15,)

        # Test score
        score = bart_meta.score(X_test, np.random.rand(15).astype(np.float64))
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ BART metamodel hyperparameters work correctly")

    def test_bart_metamodel_rmse_calculation(self):
        """Test RMSE calculation for BART metamodel."""
        X_train = np.random.rand(50, 2).astype(np.float64)
        y_train = np.random.rand(50).astype(np.float64)

        bart_meta = BARTMetamodel(n_trees=10, n_chains=1)
        bart_meta.fit(X_train, y_train)

        X_test = np.random.rand(10, 2).astype(np.float64)
        y_true = np.random.rand(10).astype(np.float64)

        rmse_score = bart_meta.rmse(X_test, y_true)
        assert isinstance(rmse_score, float)
        assert rmse_score >= 0.0

        print("✅ BART metamodel RMSE calculation works correctly")


class TestMLPComprehensive:
    """Comprehensive tests for MLP to achieve >95% coverage."""

    def test_mlp_metamodel_basic_functionality(self):
        """Test basic MLP metamodel functionality."""
        # Create training data
        X_train = np.random.rand(100, 3).astype(np.float64)
        y_train = np.random.rand(100).astype(np.float64)

        mlp_meta = MLP(hidden_layer_sizes=(10, 5), max_iter=50)

        # Test fitting
        mlp_meta.fit(X_train, y_train)

        # Test prediction
        X_test = np.random.rand(10, 3).astype(np.float64)
        predictions = mlp_meta.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 10
        assert all(np.isfinite(pred) for pred in predictions)

        # Test scoring
        score = mlp_meta.score(X_test, predictions)
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ MLP metamodel basic functionality works")

    def test_mlp_metamodel_validation_errors(self):
        """Test MLP metamodel with validation errors."""
        mlp_meta = MLP()

        # Test with mismatched dimensions
        X_train = np.random.rand(10, 3).astype(np.float64)  # 3 parameters
        y_train = np.random.rand(10).astype(np.float64)     # 10 outcomes

        X_test = np.random.rand(5, 2).astype(np.float64)    # Different number of parameters (should fail)

        mlp_meta.fit(X_train, y_train)

        with pytest.raises(ValueError):
            mlp_meta.predict(X_test)

        print("✅ MLP metamodel validation errors handled correctly")

    def test_mlp_metamodel_edge_cases(self):
        """Test MLP metamodel with edge cases."""
        # Single sample single parameter
        X_single = np.array([[0.5]], dtype=np.float64)
        y_single = np.array([1.0], dtype=np.float64)

        mlp_meta = MLP(hidden_layer_sizes=(5,), max_iter=20)
        mlp_meta.fit(X_single, y_single)

        X_test_single = np.array([[0.6]], dtype=np.float64)
        pred_single = mlp_meta.predict(X_test_single)

        assert isinstance(pred_single, np.ndarray)
        assert pred_single.shape == (1,)

        # Many samples, two parameters
        X_many = np.random.rand(300, 2).astype(np.float64)
        y_many = np.random.rand(300).astype(np.float64)

        mlp_meta_many = MLP(hidden_layer_sizes=(20, 10), max_iter=30)
        mlp_meta_many.fit(X_many, y_many)

        X_test_many = np.random.rand(15, 2).astype(np.float64)
        pred_many = mlp_meta_many.predict(X_test_many)

        assert pred_many.shape == (15,)

        print("✅ MLP metamodel edge cases handled correctly")

    def test_mlp_metamodel_hyperparameters(self):
        """Test MLP metamodel with different hyperparameters."""
        X_train = np.random.rand(75, 4).astype(np.float64)
        y_train = np.random.rand(75).astype(np.float64)

        # Test with specific hyperparameters
        mlp_meta = MLP(
            hidden_layer_sizes=(20, 10, 5),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            max_iter=100,
            random_state=42
        )

        mlp_meta.fit(X_train, y_train)

        X_test = np.random.rand(20, 4).astype(np.float64)
        predictions = mlp_meta.predict(X_test)

        assert predictions.shape == (20,)

        # Test score
        score = mlp_meta.score(X_test, np.random.rand(20).astype(np.float64))
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ MLP metamodel hyperparameters work correctly")

    def test_mlp_metamodel_rmse_calculation(self):
        """Test RMSE calculation for MLP metamodel."""
        X_train = np.random.rand(60, 2).astype(np.float64)
        y_train = np.random.rand(60).astype(np.float64)

        mlp_meta = MLP(hidden_layer_sizes=(10, 5), max_iter=50)
        mlp_meta.fit(X_train, y_train)

        X_test = np.random.rand(12, 2).astype(np.float64)
        y_true = np.random.rand(12).astype(np.float64)

        rmse_score = mlp_meta.rmse(X_test, y_true)
        assert isinstance(rmse_score, float)
        assert rmse_score >= 0.0

        print("✅ MLP metamodel RMSE calculation works correctly")


class TestEnsembleMetamodelComprehensive:
    """Comprehensive tests for EnsembleMetamodel to achieve >95% coverage."""

    def test_ensemble_metamodel_basic_functionality(self):
        """Test basic ensemble metamodel functionality."""
        # Create training data
        X_train = np.random.rand(100, 3).astype(np.float64)
        y_train = np.random.rand(100).astype(np.float64)

        # Create individual metamodels for ensemble
        rf_meta = RandomForestMetamodel(n_estimators=10)
        gam_meta = GAMMetamodel(max_iter=50)

        ensemble_meta = EnsembleMetamodel(metamodels=[rf_meta, gam_meta])

        # Test fitting
        ensemble_meta.fit(X_train, y_train)

        # Test prediction
        X_test = np.random.rand(10, 3).astype(np.float64)
        predictions = ensemble_meta.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 10
        assert all(np.isfinite(pred) for pred in predictions)

        # Test scoring
        score = ensemble_meta.score(X_test, predictions)
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ Ensemble metamodel basic functionality works")

    def test_ensemble_metamodel_validation_errors(self):
        """Test ensemble metamodel with validation errors."""
        # Try to create ensemble without any metamodels
        with pytest.raises(ValueError, match="At least one metamodel must be provided"):
            EnsembleMetamodel(metamodels=[])

        # Try to create ensemble with mismatched dimensions
        X_train = np.random.rand(10, 3).astype(np.float64)  # 3 parameters
        y_train = np.random.rand(10).astype(np.float64)     # 10 outcomes

        rf_meta = RandomForestMetamodel(n_estimators=5)
        gam_meta = GAMMetamodel(max_iter=20)

        ensemble_meta = EnsembleMetamodel(metamodels=[rf_meta, gam_meta])
        ensemble_meta.fit(X_train, y_train)

        X_test = np.random.rand(5, 2).astype(np.float64)  # Different number of parameters

        with pytest.raises(ValueError):
            ensemble_meta.predict(X_test)

        print("✅ Ensemble metamodel validation errors handled correctly")

    def test_ensemble_metamodel_edge_cases(self):
        """Test ensemble metamodel with edge cases."""
        # Single metamodel in ensemble
        X_single = np.random.rand(50, 2).astype(np.float64)
        y_single = np.random.rand(50).astype(np.float64)

        rf_only = RandomForestMetamodel(n_estimators=5)
        ensemble_single = EnsembleMetamodel(metamodels=[rf_only])

        ensemble_single.fit(X_single, y_single)

        X_test_single = np.random.rand(5, 2).astype(np.float64)
        pred_single = ensemble_single.predict(X_test_single)

        assert isinstance(pred_single, np.ndarray)
        assert pred_single.shape == (5,)

        # Many metamodels in ensemble
        X_many = np.random.rand(120, 2).astype(np.float64)
        y_many = np.random.rand(120).astype(np.float64)

        rf_meta = RandomForestMetamodel(n_estimators=5)
        gam_meta = GAMMetamodel(max_iter=20)
        mlp_meta = MLP(hidden_layer_sizes=(5,), max_iter=30)

        ensemble_many = EnsembleMetamodel(metamodels=[rf_meta, gam_meta, mlp_meta])
        ensemble_many.fit(X_many, y_many)

        X_test_many = np.random.rand(15, 2).astype(np.float64)
        pred_many = ensemble_many.predict(X_test_many)

        assert pred_many.shape == (15,)

        print("✅ Ensemble metamodel edge cases handled correctly")

    def test_ensemble_metamodel_prediction_weights(self):
        """Test ensemble metamodel with specific prediction weights."""
        X_train = np.random.rand(80, 3).astype(np.float64)
        y_train = np.random.rand(80).astype(np.float64)

        rf_meta = RandomForestMetamodel(n_estimators=10)
        gam_meta = GAMMetamodel(max_iter=50)

        # Test ensemble with custom weights
        ensemble_weighted = EnsembleMetamodel(
            metamodels=[rf_meta, gam_meta],
            weights=[0.7, 0.3]  # Weight RF more heavily
        )

        ensemble_weighted.fit(X_train, y_train)

        X_test = np.random.rand(10, 3).astype(np.float64)
        predictions = ensemble_weighted.predict(X_test)

        assert predictions.shape == (10,)

        # Test score
        score = ensemble_weighted.score(X_test, np.random.rand(10).astype(np.float64))
        assert isinstance(score, float)
        assert np.isfinite(score)

        print("✅ Ensemble metamodel prediction with weights works correctly")

    def test_ensemble_metamodel_rmse_calculation(self):
        """Test RMSE calculation for ensemble metamodel."""
        X_train = np.random.rand(70, 2).astype(np.float64)
        y_train = np.random.rand(70).astype(np.float64)

        rf_meta = RandomForestMetamodel(n_estimators=10)
        gam_meta = GAMMetamodel(max_iter=50)

        ensemble_meta = EnsembleMetamodel(metamodels=[rf_meta, gam_meta])
        ensemble_meta.fit(X_train, y_train)

        X_test = np.random.rand(12, 2).astype(np.float64)
        y_true = np.random.rand(12).astype(np.float64)

        rmse_score = ensemble_meta.rmse(X_test, y_true)
        assert isinstance(rmse_score, float)
        assert rmse_score >= 0.0

        print("✅ Ensemble metamodel RMSE calculation works correctly")

    def test_metamodel_protocol_compliance(self):
        """Test that all metamodels implement required interface methods."""
        X_train = np.random.rand(50, 2).astype(np.float64)
        y_train = np.random.rand(50).astype(np.float64)

        X_test = np.random.rand(10, 2).astype(np.float64)
        y_test = np.random.rand(10).astype(np.float64)

        # Test RandomForestMetamodel
        rf_meta = RandomForestMetamodel(n_estimators=5)
        rf_meta.fit(X_train, y_train)
        rf_pred = rf_meta.predict(X_test)
        rf_score = rf_meta.score(X_test, y_test)
        rf_rmse = rf_meta.rmse(X_test, y_test)

        assert isinstance(rf_pred, np.ndarray)
        assert isinstance(rf_score, float)
        assert isinstance(rf_rmse, float)

        # Test GAMMetamodel
        gam_meta = GAMMetamodel(max_iter=30)
        gam_meta.fit(X_train, y_train)
        gam_pred = gam_meta.predict(X_test)
        gam_score = gam_meta.score(X_test, y_test)
        gam_rmse = gam_meta.rmse(X_test, y_test)

        assert isinstance(gam_pred, np.ndarray)
        assert isinstance(gam_score, float)
        assert isinstance(gam_rmse, float)

        # Test MLP
        mlp_meta = MLP(hidden_layer_sizes=(5,), max_iter=30)
        mlp_meta.fit(X_train, y_train)
        mlp_pred = mlp_meta.predict(X_test)
        mlp_score = mlp_meta.score(X_test, y_test)
        mlp_rmse = mlp_meta.rmse(X_test, y_test)

        assert isinstance(mlp_pred, np.ndarray)
        assert isinstance(mlp_score, float)
        assert isinstance(mlp_rmse, float)

        # Test EnsembleMetamodel
        ensemble_meta = EnsembleMetamodel(metamodels=[rf_meta, gam_meta])
        ensemble_meta.fit(X_train, y_train)
        ensemble_pred = ensemble_meta.predict(X_test)
        ensemble_score = ensemble_meta.score(X_test, y_test)
        ensemble_rmse = ensemble_meta.rmse(X_test, y_test)

        assert isinstance(ensemble_pred, np.ndarray)
        assert isinstance(ensemble_score, float)
        assert isinstance(ensemble_rmse, float)

        print("✅ All metamodel protocols comply with interface requirements")
