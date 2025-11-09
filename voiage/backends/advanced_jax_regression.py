
import jax.numpy as jnp

class JaxAdvancedRegression:
    """Advanced JAX-optimized regression models for EVPPI calculations."""
    
    def __init__(self, model_type="polynomial"):
        self.model_type = model_type
        self.fitted_params = None
        
    def polynomial_features(self, x, degree=2):
        """Generate polynomial features for regression."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_samples, n_features = x.shape
        
        # Create polynomial features up to specified degree
        features = [jnp.ones(n_samples)]  # Bias term
        
        for d in range(1, degree + 1):
            for i in range(n_features):
                features.append(x[:, i] ** d)
        
        # Add interaction terms for degree >= 2
        if degree >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    features.append(x[:, i] * x[:, j])
        
        return jnp.column_stack(features)
    
    def fit_polynomial(self, X, y, degree=2, regularization=1e-6):
        """Fit polynomial regression using JAX optimization."""
        # Generate polynomial features
        X_poly = self.polynomial_features(X, degree)
        n_samples, n_features = X_poly.shape
        
        # Solve normal equations with regularization: (X^T X + λI)^(-1) X^T y
        XtX = jnp.dot(X_poly.T, X_poly)
        reg_matrix = regularization * jnp.eye(n_features)
        XtX_reg = XtX + reg_matrix
        Xty = jnp.dot(X_poly.T, y)
        
        # Solve for parameters
        beta = jnp.linalg.solve(XtX_reg, Xty)
        self.fitted_params = beta
        self.degree = degree
        
        return self
    
    def predict(self, X):
        """Make predictions using fitted model."""
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before prediction")
            
        X_poly = self.polynomial_features(X, self.degree)
        return jnp.dot(X_poly, self.fitted_params)
    
    def r_squared(self, X, y):
        """Calculate R-squared score."""
        y_pred = self.predict(X)
        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return 1.0 - (ss_res / ss_tot)
    
    def cross_validate(self, X, y, degree=2, n_folds=5, regularization=1e-6):
        """Perform cross-validation to find optimal degree."""
        n_samples = X.shape[0]
        fold_size = n_samples // n_folds
        scores = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else n_samples
            
            # Create train/validation split
            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            X_train = jnp.concatenate([X[:start_idx], X[end_idx:]], axis=0)
            y_train = jnp.concatenate([y[:start_idx], y[end_idx:]], axis=0)
            
            # Fit model
            self.fit_polynomial(X_train, y_train, degree, regularization)
            
            # Calculate validation score
            y_pred = self.predict(X_val)
            ss_res = jnp.sum((y_val - y_pred) ** 2)
            ss_tot = jnp.sum((y_val - jnp.mean(y_val)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot)
            scores.append(float(r2))
            
        return jnp.mean(jnp.array(scores))
