# voiage/metamodels.py

"""Metamodels for Value of Information analysis."""

from typing import Protocol

import numpy as np
import xarray as xr

# Optional dependencies
try:
    import flax.linen as nn
    from flax.training import train_state
    import jax
    import jax.numpy as jnp
    import optax
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = None
    jax = None
    jnp = None
    optax = None
    train_state = None

try:
    import tinygp
    TINYGP_AVAILABLE = True
except ImportError:
    TINYGP_AVAILABLE = False
    tinygp = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    LinearRegression = None

try:
    from pygam import LinearGAM, s as gam_spline
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
    LinearGAM = None
    gam_spline = None

try:
    import pymc as pm
    import arviz as az
    import pymc_bart as pmb
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None
    pmb = None

from voiage.schema import ParameterSet


class Metamodel(Protocol):
    """A protocol for metamodels used in VOI analysis."""

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data.
        Parameters
        ----------
        x : ParameterSet
            The input parameters.
        y : np.ndarray
            The target values.
        """
        ...

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters.
        Parameters
        ----------
        x : ParameterSet
            The input parameters.
        Returns
        -------
        np.ndarray
            The predicted target values.
        """
        ...

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        x : ParameterSet
            The input parameters.
        y : np.ndarray
            The true target values.
            
        Returns
        -------
        float
            R^2 score.
        """
        ...

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction.
        
        Parameters
        ----------
        x : ParameterSet
            The input parameters.
        y : np.ndarray
            The true target values.
            
        Returns
        -------
        float
            RMSE score.
        """
        ...


def calculate_diagnostics(model: Metamodel, x: ParameterSet, y: np.ndarray) -> dict:
    """Calculate comprehensive diagnostics for a fitted metamodel.
    
    Parameters
    ----------
    model : Metamodel
        A fitted metamodel instance.
    x : ParameterSet
        The input parameters.
    y : np.ndarray
        The true target values.
        
    Returns
    -------
    dict
        A dictionary containing various diagnostic metrics.
    """
    # Calculate basic metrics
    try:
        r2 = model.score(x, y)
    except (AttributeError, NotImplementedError):
        # If score method is not implemented, calculate it manually
        y_pred = model.predict(x)
        # Convert to numpy array if it's an xarray DataArray
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    
    try:
        rmse = model.rmse(x, y)
    except (AttributeError, NotImplementedError):
        # If rmse method is not implemented, calculate it manually
        y_pred = model.predict(x)
        # Convert to numpy array if it's an xarray DataArray
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    y_pred = model.predict(x)
    # Convert to numpy array if it's an xarray DataArray
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values
    mae = np.mean(np.abs(y - y_pred))
    
    # Calculate additional metrics
    residuals = y - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Return all diagnostics
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mean_residual": mean_residual,
        "std_residual": std_residual,
        "n_samples": len(y)
    }


def cross_validate(model: Metamodel, x: ParameterSet, y: np.ndarray, cv_folds: int = 5, random_state: int = 42) -> dict:
    """Perform cross-validation for a metamodel.
    
    Parameters
    ----------
    model : Metamodel
        A metamodel class (not instance) to be cross-validated.
    x : ParameterSet
        The input parameters.
    y : np.ndarray
        The true target values.
    cv_folds : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    dict
        A dictionary containing cross-validation results.
    """
    # Convert ParameterSet to numpy array
    x_np = np.array(list(x.parameters.values())).T
    n_samples = len(y)
    
    # Create shuffled indices
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    
    # Calculate fold sizes
    fold_size = n_samples // cv_folds
    remainder = n_samples % cv_folds
    
    # Store results for each fold
    fold_scores = []
    fold_rmse = []
    fold_mae = []
    
    # Perform cross-validation
    for fold in range(cv_folds):
        # Calculate start and end indices for test set
        start_idx = fold * fold_size + min(fold, remainder)
        end_idx = start_idx + fold_size + (1 if fold < remainder else 0)
        
        # Split data into train and test sets
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        # Create train and test ParameterSets
        train_params = {}
        test_params = {}
        for param_name, param_values in x.parameters.items():
            train_params[param_name] = ("n_samples", param_values[train_indices])
            test_params[param_name] = ("n_samples", param_values[test_indices])
        
        train_x = ParameterSet(dataset=xr.Dataset(train_params))
        test_x = ParameterSet(dataset=xr.Dataset(test_params))
        train_y = y[train_indices]
        test_y = y[test_indices]
        
        # Create and fit model instance
        model_instance = model()
        model_instance.fit(train_x, train_y)
        
        # Calculate diagnostics on test set
        diagnostics = calculate_diagnostics(model_instance, test_x, test_y)
        fold_scores.append(diagnostics["r2"])
        fold_rmse.append(diagnostics["rmse"])
        fold_mae.append(diagnostics["mae"])
    
    # Return cross-validation results
    return {
        "cv_r2_mean": np.mean(fold_scores),
        "cv_r2_std": np.std(fold_scores),
        "cv_rmse_mean": np.mean(fold_rmse),
        "cv_rmse_std": np.std(fold_rmse),
        "cv_mae_mean": np.mean(fold_mae),
        "cv_mae_std": np.std(fold_mae),
        "n_folds": cv_folds,
        "fold_scores": fold_scores,
        "fold_rmse": fold_rmse,
        "fold_mae": fold_mae
    }


def compare_metamodels(models: list, x: ParameterSet, y: np.ndarray, cv_folds: int = 5) -> dict:
    """Compare multiple metamodels using cross-validation.
    
    Parameters
    ----------
    models : list
        A list of metamodel classes to compare.
    x : ParameterSet
        The input parameters.
    y : np.ndarray
        The true target values.
    cv_folds : int, default=5
        Number of cross-validation folds.
        
    Returns
    -------
    dict
        A dictionary containing comparison results for all models.
    """
    results = {}
    
    for model in models:
        try:
            cv_results = cross_validate(model, x, y, cv_folds)
            model_name = model.__name__
            results[model_name] = cv_results
        except Exception as e:
            # If a model fails, record the error
            model_name = model.__name__
            results[model_name] = {"error": str(e)}
    
    return results


class MLP:
    """A simple MLP model."""
    
    def __init__(self, features: int):
        if not FLAX_AVAILABLE:
            raise ImportError("Flax is required for MLP. Please install it with `pip install flax`.")
        
        self.module = nn.Dense(features=features)
    
    def __call__(self, x):
        # This is a placeholder - actual implementation would be in the Flax module
        pass


class FlaxMetamodel:
    """A metamodel that uses a Flax MLP to predict the target values."""

    def __init__(self, learning_rate=0.01, n_epochs=100):
        if not FLAX_AVAILABLE:
            raise ImportError("Flax is required for FlaxMetamodel. Please install it with `pip install flax`.")
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.state = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        x_np = np.array(list(x.parameters.values())).T
        y_np = y.reshape(-(-1, 1))

        model = nn.Dense(features=128)
        params = model.init(jax.random.PRNGKey(0), x_np)["params"]
        tx = optax.adam(self.learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )

        @jax.jit
        def train_step(state, batch_x, batch_y):
            def loss_fn(params):
                logits = state.apply_fn({"params": params}, batch_x)
                loss = jnp.mean((logits - batch_y) ** 2)
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        for _ in range(self.n_epochs):
            self.state, _ = train_step(self.state, x_np, y_np)

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.state is None:
            raise RuntimeError("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.state.apply_fn({"params": self.state.params}, x_np)
        return np.array(y_pred)


class TinyGPMetamodel:
    """A metamodel that uses a tinygp GP to predict the target values."""

    def __init__(self):
        if not TINYGP_AVAILABLE:
            raise ImportError("tinygp is required for TinyGPMetamodel. Please install it with `pip install tinygp`.")
        
        self.gp = None
        self.x_train = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        self.x_train = np.array(list(x.parameters.values())).T
        self.y_train = y

        kernel = 1.0 * tinygp.kernels.Matern32(1.0)

        def loss(params):
            gp = tinygp.GaussianProcess(
                params["kernel"], self.x_train, diag=jnp.exp(params["log_diag"])
            )
            return -gp.log_probability(self.y_train)

        opt = optax.adam(0.01)
        params = {
            "kernel": kernel,
            "log_diag": jnp.log(jnp.full(self.x_train.shape[0], 1e-5)),
        }
        state = opt.init(params)

        for _ in range(100):
            loss_val, grads = jax.value_and_grad(loss)(params)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)

        self.gp = tinygp.GaussianProcess(
            params["kernel"], self.x_train, diag=jnp.exp(params["log_diag"])
        )

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.gp is None:
            raise RuntimeError("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        _, cond = self.gp.condition(self.y_train, x_np)
        return cond.loc


class RandomForestMetamodel:
    """A metamodel that uses a Random Forest to predict the target values."""

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RandomForestMetamodel. "
                            "Please install it with `pip install scikit-learn`.")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        x_np = np.array(list(x.parameters.values())).T
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(x_np, y)

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        x_np = np.array(list(x.parameters.values())).T
        return self.model.predict(x_np)

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.model.predict(x_np)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.model.predict(x_np)
        return np.sqrt(np.mean((y - y_pred) ** 2))


class GAMMetamodel:
    """A metamodel that uses a Generalized Additive Model to predict the target values."""

    def __init__(self, n_splines=10, lam=0.1):
        if not PYGAM_AVAILABLE:
            raise ImportError("pygam is required for GAMMetamodel. "
                            "Please install it with `pip install pygam`.")
        
        self.n_splines = n_splines
        self.lam = lam
        self.model = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        x_np = np.array(list(x.parameters.values())).T
        
        # Create spline terms for each feature
        # For n features, we create terms like s(0) + s(1) + ... + s(n-1)
        n_features = x_np.shape[1]
        if n_features == 1:
            terms = gam_spline(0, n_splines=self.n_splines)
        else:
            # Build the terms dynamically
            terms = gam_spline(0, n_splines=self.n_splines)
            for i in range(1, n_features):
                terms += gam_spline(i, n_splines=self.n_splines)
        
        self.model = LinearGAM(terms, lam=self.lam)
        self.model.fit(x_np, y)

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        x_np = np.array(list(x.parameters.values())).T
        return self.model.predict(x_np)

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.model.predict(x_np)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.model.predict(x_np)
        return np.sqrt(np.mean((y - y_pred) ** 2))


class BARTMetamodel:
    """A metamodel that uses a BART (Bayesian Additive Regression Trees) to predict the target values."""

    def __init__(self, num_trees=50, alpha=0.95, beta=2.0):
        if not PYMC_AVAILABLE:
            raise ImportError("pymc and pymc-bart are required for BARTMetamodel. "
                            "Please install them with `pip install pymc pymc-bart`.")
        
        self.num_trees = num_trees
        self.alpha = alpha
        self.beta = beta
        self.model = None
        self.trace = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        x_np = np.array(list(x.parameters.values())).T
        
        with pm.Model() as model:
            # BART prior using pymc-bart
            mu = pmb.BART('mu', X=x_np, Y=y, m=self.num_trees)
            # Likelihood
            sigma = pm.HalfNormal('sigma', 1)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Sample from the posterior
            self.trace = pm.sample(500, tune=500, chains=2, cores=1, random_seed=42, return_inferencedata=True)
        
        self.model = model

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.model is None or self.trace is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        x_np = np.array(list(x.parameters.values())).T
        
        # Use the posterior predictive to make predictions
        with self.model:
            post_pred = pm.sample_posterior_predictive(self.trace, var_names=['mu'], random_seed=42)
        
        # Return the mean prediction
        return np.mean(post_pred.posterior_predictive['mu'], axis=(0, 1))

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.model is None or self.trace is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        y_pred = self.predict(x)
        # Convert to numpy array if it's an xarray DataArray
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.model is None or self.trace is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        y_pred = self.predict(x)
        # Convert to numpy array if it's an xarray DataArray
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        return np.sqrt(np.mean((y - y_pred) ** 2))
