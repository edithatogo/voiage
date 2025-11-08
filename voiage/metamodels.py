# voiage/metamodels.py

"""Metamodels for Value of Information analysis."""

from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr

# Optional dependencies
try:
    import flax.linen as flax_nn
    from flax.training import train_state
    import jax
    import jax.numpy as jnp
    import optax
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    flax_nn = None
    jax = None
    jnp = None
    optax = None
    train_state = None

try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    torch_nn = None
    optim = None

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

# Fix for numpy 2.0 compatibility with pygam
# pygam uses deprecated numpy aliases that were removed in numpy 2.0
try:
    import numpy as np

    # Add back deprecated aliases for pygam compatibility
    # These aliases were removed in numpy 2.0
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'complex'):
        np.complex = complex

    # Also patch scipy sparse matrix attributes if needed
    try:
        import scipy.sparse
        if hasattr(scipy.sparse, 'csr_matrix') and not hasattr(scipy.sparse.csr_matrix, 'A'):
            # Add the A property as an alias to toarray()
            def _get_a(self):
                return self.toarray()
            scipy.sparse.csr_matrix.A = property(_get_a)
    except ImportError:
        pass

    from pygam import LinearGAM
    from pygam import s as gam_spline
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
    LinearGAM = None
    gam_spline = None

try:
    import arviz as az
    import pymc as pm
    import pymc_bart as pmb
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None
    pmb = None

from voiage.schema import ParameterSet


@runtime_checkable
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

        self.module = flax_nn.Dense(features=features)

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
        y_np = y.reshape((-1, 1))

        model = flax_nn.Dense(features=1)
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

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.state is None:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.state is None:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return np.sqrt(np.mean((y - y_pred) ** 2))


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

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.gp is None:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.gp is None:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return np.sqrt(np.mean((y - y_pred) ** 2))


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

        # Ensure n_splines is greater than the default spline_order (3)
        if n_splines <= 3:
            n_splines = 4

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
            pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            # Sample from the posterior
            self.trace = pm.sample(500, tune=500, chains=2, cores=1, random_seed=42, return_inferencedata=True)

        self.model = model

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.model is None or self.trace is None:
            raise RuntimeError("The model has not been fitted yet.")

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


class ActiveLearningMetamodel:
    """A metamodel that uses active learning to iteratively improve its predictions."""

    def __init__(self, base_model, n_initial_samples=10, n_query_samples=5, acquisition_function='uncertainty'):
        """
        Initialize the active learning metamodel.

        Parameters
        ----------
        base_model : Metamodel
            The base metamodel to use for predictions.
        n_initial_samples : int, default=10
            Number of initial samples to start with.
        n_query_samples : int, default=5
            Number of samples to query at each iteration.
        acquisition_function : str, default='uncertainty'
            The acquisition function to use for selecting samples.
            Options are 'uncertainty', 'random', 'margin'.
        """
        self.base_model = base_model
        self.n_initial_samples = n_initial_samples
        self.n_query_samples = n_query_samples
        self.acquisition_function = acquisition_function
        self.is_fitted = False
        self.X_train = None
        self.y_train = None
        self.iteration = 0

    def _select_initial_samples(self, x_pool, y_pool):
        """Select initial samples randomly from the pool."""
        n_samples = min(self.n_initial_samples, len(x_pool))
        indices = np.random.choice(len(x_pool), n_samples, replace=False)
        x_selected = x_pool[indices]
        if y_pool is not None:
            y_selected = y_pool[indices]
        else:
            y_selected = None
        return x_selected, y_selected, indices

    def _acquisition_uncertainty(self, x_pool):
        """Select samples with highest prediction uncertainty."""
        # For models that support uncertainty estimation, use that
        # For now, we'll use a simple approach based on model variance across bootstrap samples
        # or use model-specific uncertainty if available

        # Simple approach: use prediction variance across multiple predictions
        # (this is a placeholder - real implementation would depend on the base model)
        try:
            predictions = []
            # Try to get multiple predictions if the model supports it
            for _ in range(10):
                pred = self.base_model.predict(self._array_to_parameterset(x_pool))
                predictions.append(pred)

            # Calculate variance across predictions
            pred_var = np.var(predictions, axis=0)
            return pred_var
        except Exception:
            # Fallback: use a simple distance-based uncertainty
            if self.X_train is not None:
                # Calculate distances to existing training points
                distances = np.min([np.linalg.norm(x_pool - x_train, axis=1)
                                  for x_train in self.X_train], axis=0)
                return distances
            else:
                # If no training data, select randomly
                return np.random.rand(len(x_pool))

    def _acquisition_random(self, x_pool):
        """Select samples randomly."""
        return np.random.rand(len(x_pool))

    def _acquisition_margin(self, x_pool):
        """Select samples with smallest margin (closest to decision boundary)."""
        # This is more relevant for classification, but we can adapt it for regression
        # by looking at predictions close to some threshold or with high variance
        return self._acquisition_uncertainty(x_pool)

    def _select_query_samples(self, x_pool, y_pool):
        """Select samples to query based on the acquisition function."""
        if self.acquisition_function == 'uncertainty':
            scores = self._acquisition_uncertainty(x_pool)
        elif self.acquisition_function == 'random':
            scores = self._acquisition_random(x_pool)
        elif self.acquisition_function == 'margin':
            scores = self._acquisition_margin(x_pool)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")

        # Select samples with highest scores
        n_query = min(self.n_query_samples, len(x_pool))
        indices = np.argpartition(scores, -n_query)[-n_query:]
        x_selected = x_pool[indices]
        if y_pool is not None:
            y_selected = y_pool[indices]
        else:
            y_selected = None
        return x_selected, y_selected, indices

    def _array_to_parameterset(self, x_array):
        """Convert numpy array to ParameterSet."""
        n_samples = x_array.shape[0]
        param_dict = {}
        for i in range(x_array.shape[1]):
            param_dict[f"param_{i}"] = x_array[:, i]

        dataset = xr.Dataset(
            {k: ("n_samples", v) for k, v in param_dict.items()},
            coords={"n_samples": np.arange(n_samples)}
        )
        return ParameterSet(dataset=dataset)

    def _parameterset_to_array(self, param_set):
        """Convert ParameterSet to numpy array."""
        return np.array(list(param_set.parameters.values())).T

    def fit(self, x: ParameterSet, y: np.ndarray, x_pool=None, y_pool=None, n_iterations=5) -> None:
        """
        Fit the metamodel using active learning.

        Parameters
        ----------
        x : ParameterSet
            Initial training parameters.
        y : np.ndarray
            Initial training targets.
        x_pool : np.ndarray, optional
            Pool of unlabeled samples to query from.
        y_pool : np.ndarray, optional
            True labels for the pool (for simulation purposes).
        n_iterations : int, default=5
            Number of active learning iterations.
        """
        # Convert initial training data to arrays
        x_initial = self._parameterset_to_array(x)

        # If no pool is provided, create a synthetic one for demonstration
        if x_pool is None:
            # Create a pool of random samples around the initial data
            n_pool = 1000
            x_pool = np.random.randn(n_pool, x_initial.shape[1]) * 2
            y_pool = None  # No true labels for synthetic pool

        # Start with the initial training data
        self.X_train = x_initial
        self.y_train = y

        # Fit the base model on initial training data
        train_param_set = self._array_to_parameterset(self.X_train)
        if self.y_train is not None and len(self.y_train) > 0:
            self.base_model.fit(train_param_set, self.y_train)

        # Active learning loop
        for iteration in range(n_iterations):
            self.iteration = iteration + 1

            # Select new samples to query
            if y_pool is not None:
                # If we have true labels, we can evaluate the selection
                x_new, y_new, new_indices = self._select_query_samples(x_pool, y_pool)
            else:
                # Otherwise, just select based on acquisition function
                x_new, _, new_indices = self._select_query_samples(x_pool, None)
                y_new = None  # We don't have true labels

            # Add new samples to training set only if we have labels for them
            if y_new is not None:
                self.X_train = np.vstack([self.X_train, x_new])
                self.y_train = np.hstack([self.y_train, y_new])

            # Fit the base model on updated training data
            train_param_set = self._array_to_parameterset(self.X_train)
            if self.y_train is not None and len(self.y_train) > 0:
                self.base_model.fit(train_param_set, self.y_train)

        self.is_fitted = True

    def predict(self, x: ParameterSet) -> np.ndarray:
        """
        Predict using the actively learned metamodel.

        Parameters
        ----------
        x : ParameterSet
            The input parameters.

        Returns
        -------
        np.ndarray
            The predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("The model has not been fitted yet.")

        return self.base_model.predict(x)

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

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
        if not self.is_fitted:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """
        Return the root mean squared error of the prediction.

        Parameters
        ----------
        x : ParameterSet
            The input parameters.
        y : np.ndarray
            The true target values.

        Returns
        -------
        float
            RMSE.
        """
        if not self.is_fitted:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return np.sqrt(np.mean((y - y_pred) ** 2))


class EnsembleMetamodel:
    """A metamodel that combines predictions from multiple metamodels."""

    def __init__(self, models, method='mean'):
        """
        Initialize the ensemble metamodel.

        Parameters
        ----------
        models : list
            A list of metamodel instances to ensemble.
        method : str, default='mean'
            The ensemble method to use. Options are 'mean', 'median', 'weighted'.
        """
        self.models = models
        self.method = method
        self.weights = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """
        Fit all metamodels in the ensemble.

        Parameters
        ----------
        x : ParameterSet
            The input parameters.
        y : np.ndarray
            The target values.
        """
        # Fit all models
        for model in self.models:
            model.fit(x, y)

        # If using weighted ensemble, compute weights based on model performance
        if self.method == 'weighted':
            scores = []
            for model in self.models:
                try:
                    score = model.score(x, y)
                    scores.append(max(score, 0))  # Ensure non-negative weights
                except Exception:
                    scores.append(0.0)  # If scoring fails, give zero weight

            # Normalize scores to get weights
            total_score = sum(scores)
            if total_score > 0:
                self.weights = [score / total_score for score in scores]
            else:
                # If all scores are zero, use equal weights
                self.weights = [1.0 / len(self.models)] * len(self.models)

    def predict(self, x: ParameterSet) -> np.ndarray:
        """
        Predict using the ensemble of metamodels.

        Parameters
        ----------
        x : ParameterSet
            The input parameters.

        Returns
        -------
        np.ndarray
            The ensemble predictions.
        """
        if not self.models:
            raise RuntimeError("No models in the ensemble.")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)

        # Combine predictions based on the ensemble method
        if self.method == 'mean':
            return np.mean(predictions, axis=0)
        elif self.method == 'median':
            return np.median(predictions, axis=0)
        elif self.method == 'weighted':
            if self.weights is None:
                # If weights haven't been computed, use equal weights
                weights = [1.0 / len(self.models)] * len(self.models)
            else:
                weights = self.weights
            return np.average(predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the ensemble prediction.

        Parameters
        ----------
        x : ParameterSet
            The input parameters.
        y : np.ndarray
            The true target values.

        Returns
        -------
        float
            R^2 score of the ensemble.
        """
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """
        Return the root mean squared error of the ensemble prediction.

        Parameters
        ----------
        x : ParameterSet
            The input parameters.
        y : np.ndarray
            The true target values.

        Returns
        -------
        float
            RMSE of the ensemble.
        """
        y_pred = self.predict(x)
        return np.sqrt(np.mean((y - y_pred) ** 2))


class PyTorchNNMetamodel:
    """A metamodel that uses a PyTorch neural network to predict the target values."""

    def __init__(self, hidden_layers=None, learning_rate=0.001, n_epochs=1000, batch_size=32):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchNNMetamodel. "
                            "Please install it with `pip install torch`.")

        if hidden_layers is None:
            hidden_layers = [64, 32]
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim, output_dim):
        """Build the neural network architecture."""
        layers = []
        # Input layer
        layers.append(torch_nn.Linear(input_dim, self.hidden_layers[0]))
        layers.append(torch_nn.ReLU())

        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            layers.append(torch_nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
            layers.append(torch_nn.ReLU())

        # Output layer
        layers.append(torch_nn.Linear(self.hidden_layers[-1], output_dim))

        return torch_nn.Sequential(*layers)

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        # Convert data to torch tensors
        x_np = np.array(list(x.parameters.values())).T
        x_tensor = torch.FloatTensor(x_np).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        # Get input and output dimensions
        input_dim = x_tensor.shape[1]
        output_dim = y_tensor.shape[1]

        # Build model
        self.model = self._build_model(input_dim, output_dim).to(self.device)

        # Define loss function and optimizer
        criterion = torch_nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            # Forward pass
            outputs = self.model(x_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {loss.item():.4f}')

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")

        # Convert data to torch tensor
        x_np = np.array(list(x.parameters.values())).T
        x_tensor = torch.FloatTensor(x_np).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_tensor)

        # Convert back to numpy array
        return predictions.cpu().numpy().flatten()

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return np.sqrt(np.mean((y - y_pred) ** 2))
