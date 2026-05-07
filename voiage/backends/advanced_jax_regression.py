"""Advanced regression helpers for JAX-backed EVPPI calculations."""

from __future__ import annotations

import numpy as np

from voiage.exceptions import raise_value_error

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency fallback
    import numpy as jnp


class JaxAdvancedRegression:
    """Advanced JAX-optimized regression models for EVPPI calculations."""

    def __init__(self, model_type: str = "polynomial") -> None:
        self.model_type = model_type
        self.fitted_params: np.ndarray | None = None
        self.degree = 2

    def polynomial_features(self, x: np.ndarray, degree: int = 2) -> np.ndarray:
        """Generate polynomial features for regression."""
        x = jnp.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_samples, n_features = x.shape

        # Create polynomial features up to specified degree
        features = [jnp.ones(n_samples)]  # Bias term

        for d in range(1, degree + 1):
            features.extend(x[:, i] ** d for i in range(n_features))

        # Add interaction terms for degree >= 2
        if degree >= 2:
            features.extend(
                x[:, i] * x[:, j]
                for i in range(n_features)
                for j in range(i + 1, n_features)
            )

        return np.asarray(jnp.column_stack(features))

    def fit_polynomial(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 2,
        regularization: float = 1e-6,
    ) -> JaxAdvancedRegression:
        """Fit polynomial regression using JAX optimization."""
        # Generate polynomial features
        x_poly = self.polynomial_features(x, degree)
        _n_samples, n_features = x_poly.shape

        # Solve normal equations with regularization: (X^T X + λI)^(-1) X^T y
        xtx = jnp.dot(x_poly.T, x_poly)
        reg_matrix = regularization * jnp.eye(n_features)
        xtx_reg = xtx + reg_matrix
        xty = jnp.dot(x_poly.T, y)

        # Solve for parameters
        beta = jnp.linalg.solve(xtx_reg, xty)
        self.fitted_params = beta
        self.degree = degree

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using fitted model."""
        if self.fitted_params is None:
            raise_value_error("Model must be fitted before prediction")

        x_poly = self.polynomial_features(x, self.degree)
        return np.asarray(jnp.dot(x_poly, self.fitted_params))

    def r_squared(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared score."""
        y_pred = self.predict(x)
        y = jnp.asarray(y)
        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return float(1.0 - (ss_res / ss_tot))

    def cross_validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 2,
        n_folds: int = 5,
        regularization: float = 1e-6,
    ) -> float:
        """Perform cross-validation to find optimal degree."""
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        n_samples = x.shape[0]
        fold_size = n_samples // n_folds
        scores = []

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else n_samples

            # Create train/validation split
            x_val = x[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            x_train = jnp.concatenate([x[:start_idx], x[end_idx:]], axis=0)
            y_train = jnp.concatenate([y[:start_idx], y[end_idx:]], axis=0)

            # Fit model
            self.fit_polynomial(x_train, y_train, degree, regularization)

            # Calculate validation score
            y_pred = self.predict(x_val)
            ss_res = jnp.sum((y_val - y_pred) ** 2)
            ss_tot = jnp.sum((y_val - jnp.mean(y_val)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot)
            scores.append(float(r2))

        return float(np.mean(np.asarray(scores, dtype=float)))
