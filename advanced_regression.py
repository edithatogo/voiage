#!/usr/bin/env python3
"""
Phase 1.5: Advanced Regression Techniques for VOI
Comprehensive implementation of state-of-the-art regression methods for
Expected Value of Perfect Information (EVPPI) and Expected Value of Sample Information (EVSI)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
from functools import partial
import warnings
from abc import ABC, abstractmethod

class AdvancedRegressionModel(ABC):
    """Abstract base class for advanced regression models."""
    
    @abstractmethod
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> 'AdvancedRegressionModel':
        """Fit the regression model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make predictions with uncertainty estimates."""
        pass
    
    @abstractmethod
    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Calculate model performance score (R²)."""
        pass


class GaussianProcessRegression(AdvancedRegressionModel):
    """Gaussian Process Regression with JAX optimization."""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K = None
        self.L = None
        self.alpha = None
        
    def rbf_kernel(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        """Radial Basis Function kernel."""
        # Calculate squared distances
        X1_expanded = jnp.expand_dims(X1, 1)  # (n1, 1, d)
        X2_expanded = jnp.expand_dims(X2, 0)  # (1, n2, d)
        squared_dist = jnp.sum((X1_expanded - X2_expanded) ** 2, axis=2)
        
        # RBF kernel
        return self.variance * jnp.exp(-0.5 * squared_dist / (self.length_scale ** 2))
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> 'GaussianProcessRegression':
        """Fit Gaussian Process regression model."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix
        self.K = self.rbf_kernel(X, X) + self.noise * jnp.eye(len(X))
        
        # Cholesky decomposition for numerical stability
        self.L = jnp.linalg.cholesky(self.K)
        
        # Solve for alpha (K^-1 * y)
        alpha = jax.scipy.linalg.solve_triangular(self.L, y, lower=True)
        self.alpha = jax.scipy.linalg.solve_triangular(self.L.T, alpha, lower=False)
        
        return self
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions without uncertainty estimates."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        K_s = self.rbf_kernel(self.X_train, X)
        K_ss = self.rbf_kernel(X, X)
        
        # Mean prediction
        mu = jnp.dot(K_s.T, self.alpha)
        return mu
    
    def predict_with_uncertainty(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make predictions with uncertainty estimates."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        K_s = self.rbf_kernel(self.X_train, X)
        K_ss = self.rbf_kernel(X, X)
        
        # Mean prediction
        mu = jnp.dot(K_s.T, self.alpha)
        
        # Variance prediction
        v = jax.scipy.linalg.solve_triangular(self.L, K_s, lower=True)
        var = jnp.diag(K_ss) - jnp.sum(v ** 2, axis=0)
        
        # Ensure positive variances
        var = jnp.maximum(var, self.noise)
        
        return mu, var
    
    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Calculate negative log likelihood as performance metric."""
        y_pred = self.predict(X)
        mse = jnp.mean((y - y_pred) ** 2)
        return -mse  # Higher is better


class NeuralNetworkRegression(AdvancedRegressionModel):
    """Neural Network Regression with JAX."""
    
    def __init__(self, 
                 hidden_sizes: List[int] = [50, 30], 
                 activation: str = 'tanh',
                 learning_rate: float = 0.01,
                 epochs: int = 1000,
                 batch_size: int = 32):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.params = None
        
    def _activation(self, x):
        """Activation function."""
        if self.activation == 'tanh':
            return jnp.tanh(x)
        elif self.activation == 'relu':
            return jax.nn.relu(x)
        elif self.activation == 'sigmoid':
            return jax.nn.sigmoid(x)
        else:
            return jnp.tanh(x)  # Default to tanh
    
    def _init_params(self, key: jnp.ndarray, input_dim: int) -> Dict[str, jnp.ndarray]:
        """Initialize network parameters."""
        key = jax.random.split(key, len(self.hidden_sizes) + 2)
        
        params = {}
        layer_sizes = [input_dim] + self.hidden_sizes + [1]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w_scale = jnp.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            params[f'W{i}'] = jax.random.normal(key[i], (layer_sizes[i], layer_sizes[i + 1])) * w_scale
            params[f'b{i}'] = jnp.zeros(layer_sizes[i + 1])
        
        return params
    
    def _forward(self, params: Dict[str, jnp.ndarray], X: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        h = X
        
        for i in range(len(self.hidden_sizes) + 1):
            h = jnp.dot(h, params[f'W{i}']) + params[f'b{i}']
            if i < len(self.hidden_sizes):  # No activation on output layer
                h = self._activation(h)
        
        return h.flatten()
    
    def _loss(self, params: Dict[str, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Mean squared error loss."""
        y_pred = self._forward(params, X)
        return jnp.mean((y - y_pred) ** 2)
    
    def _update_step(self, params: Dict[str, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray, key: jnp.ndarray):
        """Single gradient descent update step."""
        # Gradient computation
        grads = jax.grad(self._loss)(params, X, y)
        
        # Update parameters
        updated_params = {}
        for key_name in params:
            updated_params[key_name] = params[key_name] - self.learning_rate * grads[key_name]
        
        return updated_params
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> 'NeuralNetworkRegression':
        """Train the neural network."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        self.params = self._init_params(key, X.shape[1])
        
        # Training loop
        for epoch in range(self.epochs):
            # Mini-batch training
            n_batches = len(X) // self.batch_size
            key = jax.random.split(key, n_batches + 1)[0]
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                self.params = self._update_step(self.params, X_batch, y_batch, key)
        
        return self
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._forward(self.params, X)
    
    def predict_with_uncertainty(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make predictions with uncertainty (approximate)."""
        y_pred = self.predict(X)
        # Simple uncertainty estimation based on prediction variance
        uncertainty = jnp.full_like(y_pred, jnp.var(y_pred) * 0.1)
        return y_pred, uncertainty
    
    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return 1.0 - (ss_res / ss_tot)


class EnsembleRegression(AdvancedRegressionModel):
    """Ensemble of multiple regression models."""
    
    def __init__(self, models: Optional[List[AdvancedRegressionModel]] = None, 
                 weights: Optional[jnp.ndarray] = None):
        if models is None:
            # Default ensemble: GPR + Neural Network + Polynomial
            self.models = [
                GaussianProcessRegression(),
                NeuralNetworkRegression(hidden_sizes=[20]),
                PolynomialRegression(degree=2)
            ]
        else:
            self.models = models
        
        if weights is None:
            # Equal weights by default
            self.weights = jnp.ones(len(self.models)) / len(self.models)
        else:
            self.weights = jnp.array(weights)
            # Normalize weights
            self.weights = self.weights / jnp.sum(self.weights)
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> 'EnsembleRegression':
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions by averaging ensemble predictions."""
        predictions = jnp.array([model.predict(X) for model in self.models])
        # Broadcast weights to match prediction shape
        weights_broadcast = jnp.broadcast_to(self.weights[:, None], predictions.shape)
        return jnp.sum(predictions * weights_broadcast, axis=0)
    
    def predict_with_uncertainty(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make predictions with uncertainty based on model disagreement."""
        predictions = jnp.array([model.predict(X) for model in self.models])
        mean_pred = jnp.mean(predictions, axis=0)
        pred_var = jnp.var(predictions, axis=0)
        
        return mean_pred, pred_var
    
    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Calculate ensemble R² score."""
        y_pred = self.predict(X)
        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return 1.0 - (ss_res / ss_tot)


class PolynomialRegression(AdvancedRegressionModel):
    """Polynomial regression with regularization."""
    
    def __init__(self, degree: int = 2, regularization: float = 1e-3):
        self.degree = degree
        self.regularization = regularization
        self.fitted_params = None
        
    def _create_features(self, X: jnp.ndarray) -> jnp.ndarray:
        """Create polynomial features."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # Start with bias term
        features = [jnp.ones(n_samples)]
        
        # Add polynomial terms
        for d in range(1, self.degree + 1):
            for i in range(n_features):
                features.append(X[:, i] ** d)
        
        # Add interaction terms for degree >= 2
        if self.degree >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    features.append(X[:, i] * X[:, j])
        
        return jnp.column_stack(features)
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> 'PolynomialRegression':
        """Fit polynomial regression model."""
        X_poly = self._create_features(X)
        n_features = X_poly.shape[1]
        
        # Solve with regularization: (X^T X + λI)^(-1) X^T y
        XtX = jnp.dot(X_poly.T, X_poly)
        reg_matrix = self.regularization * jnp.eye(n_features)
        XtX_reg = XtX + reg_matrix
        Xty = jnp.dot(X_poly.T, y)
        
        self.fitted_params = jnp.linalg.solve(XtX_reg, Xty)
        return self
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions."""
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_poly = self._create_features(X)
        return jnp.dot(X_poly, self.fitted_params)
    
    def predict_with_uncertainty(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make predictions with uncertainty (based on residual variance)."""
        y_pred = self.predict(X)
        
        # Estimate uncertainty from model complexity and data fit
        n_samples = X.shape[0]
        degrees_of_freedom = max(1, n_samples - self.fitted_params.shape[0])
        
        # Simple uncertainty estimation
        uncertainty = jnp.full_like(y_pred, 1.0 / jnp.sqrt(degrees_of_freedom))
        return y_pred, uncertainty
    
    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return 1.0 - (ss_res / ss_tot)


class AdvancedRegressionPipeline:
    """Complete pipeline for advanced regression in VOI analysis."""
    
    def __init__(self, 
                 model_type: str = "ensemble",
                 cv_folds: int = 5,
                 feature_selection: bool = True):
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.feature_selection = feature_selection
        self.model = None
        self.selected_features = None
        
    def _create_model(self) -> AdvancedRegressionModel:
        """Create the regression model based on type."""
        if self.model_type == "gpr":
            return GaussianProcessRegression()
        elif self.model_type == "neural_net":
            return NeuralNetworkRegression()
        elif self.model_type == "polynomial":
            return PolynomialRegression()
        elif self.model_type == "ensemble":
            return EnsembleRegression()
        else:
            warnings.warn(f"Unknown model type: {self.model_type}. Using ensemble.")
            return EnsembleRegression()
    
    def _feature_selection(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Simple feature selection based on correlation."""
        if X.ndim == 1:
            return X
        
        correlations = jnp.abs(jnp.array([jnp.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]))
        
        # Select features with correlation > 0.1
        selected_indices = jnp.where(correlations > 0.1)[0]
        
        if len(selected_indices) == 0:
            # If no features pass threshold, select the top 2
            selected_indices = jnp.argsort(correlations)[-2:]
        
        self.selected_features = selected_indices
        return X[:, selected_indices]
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> 'AdvancedRegressionPipeline':
        """Fit the regression pipeline."""
        # Feature selection
        if self.feature_selection and X.ndim > 1:
            X_processed = self._feature_selection(X, y)
        else:
            X_processed = X
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_processed, y)
        
        return self
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions."""
        if self.selected_features is not None and X.ndim > 1:
            X_processed = X[:, self.selected_features]
        else:
            X_processed = X
        
        return self.model.predict(X_processed)
    
    def predict_with_uncertainty(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make predictions with uncertainty."""
        if self.selected_features is not None and X.ndim > 1:
            X_processed = X[:, self.selected_features]
        else:
            X_processed = X
        
        return self.model.predict_with_uncertainty(X_processed)
    
    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Calculate model performance score."""
        if self.selected_features is not None and X.ndim > 1:
            X_processed = X[:, self.selected_features]
        else:
            X_processed = X
        
        return self.model.score(X_processed, y)
    
    def cross_validate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """Perform cross-validation."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        fold_size = n_samples // self.cv_folds
        scores = []
        
        for i in range(self.cv_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.cv_folds - 1 else n_samples
            
            # Create train/validation split
            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            X_train = jnp.concatenate([X[:start_idx], X[end_idx:]], axis=0)
            y_train = jnp.concatenate([y[:start_idx], y[end_idx:]], axis=0)
            
            # Fit and evaluate
            temp_pipeline = AdvancedRegressionPipeline(
                model_type=self.model_type,
                cv_folds=1,  # No nested CV
                feature_selection=self.feature_selection
            )
            temp_pipeline.fit(X_train, y_train)
            score = temp_pipeline.score(X_val, y_val)
            scores.append(score)
        
        return {
            'mean_score': float(jnp.mean(jnp.array(scores))),
            'std_score': float(jnp.std(jnp.array(scores))),
            'scores': scores
        }


# Example usage and testing
if __name__ == "__main__":
    print("🧮 Phase 1.5: Advanced Regression Techniques")
    print("=" * 55)
    
    # Generate test data
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 3))
    y = X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 100)
    
    # Convert to JAX arrays
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)
    
    # Test different models
    models = {
        "GPR": GaussianProcessRegression(),
        "Neural Network": NeuralNetworkRegression(hidden_sizes=[20, 10]),
        "Polynomial": PolynomialRegression(degree=2),
        "Ensemble": EnsembleRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📊 Testing {name}:")
        
        # Fit model
        model.fit(X_jax, y_jax)
        
        # Make predictions
        y_pred = model.predict(X_jax)
        
        # Calculate score
        score = model.score(X_jax, y_jax)
        results[name] = score
        
        print(f"   R² Score: {score:.4f}")
        
        # Test uncertainty estimation if available
        try:
            mean_pred, var_pred = model.predict_with_uncertainty(X_jax[:5])
            print(f"   Uncertainty estimation: Available")
            print(f"   Mean variance: {jnp.mean(var_pred):.4f}")
        except:
            print(f"   Uncertainty estimation: Not available")
    
    # Test pipeline
    print(f"\n🔧 Testing Advanced Pipeline:")
    pipeline = AdvancedRegressionPipeline(model_type="ensemble", feature_selection=True)
    pipeline.fit(X_jax, y_jax)
    pipeline_score = pipeline.score(X_jax, y_jax)
    cv_results = pipeline.cross_validate(X_jax, y_jax)
    
    print(f"   Pipeline R² Score: {pipeline_score:.4f}")
    print(f"   CV Mean Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    print(f"\n✅ Advanced Regression Techniques Ready!")
    print(f"💡 Models: {list(results.keys())}")
    print(f"📈 Best Model: {max(results, key=results.get)} (R² = {max(results.values()):.4f})")