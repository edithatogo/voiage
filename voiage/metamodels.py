# voiage/metamodels.py

"""Metamodels for Value of Information analysis."""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

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
    from torch import optim
    import torch.nn as torch_nn

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
    # Add back deprecated aliases for pygam compatibility without triggering
    # NumPy 2.x compatibility warnings during module import.
    if "int" not in np.__dict__:
        np.__dict__["int"] = int
    if "float" not in np.__dict__:
        np.__dict__["float"] = float
    if "bool" not in np.__dict__:
        np.__dict__["bool"] = bool
    if "complex" not in np.__dict__:
        np.__dict__["complex"] = complex

    # Also patch scipy sparse matrix attributes if needed
    try:
        import scipy.sparse

        if hasattr(scipy.sparse, "csr_matrix") and not hasattr(
            scipy.sparse.csr_matrix, "A"
        ):
            # Add the A property as an alias to toarray()
            def _get_a(self: _SparseMatrixProtocol) -> np.ndarray:
                return np.asarray(self.toarray())

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

from voiage.exceptions import (
    raise_import_error,
    raise_runtime_error,
    raise_value_error,
)
from voiage.schema import ParameterSet

DiagnosticsDict = dict[str, float | int]


@runtime_checkable
class _PredictorProtocol(Protocol):
    """Protocol for wrapped regressors used by metamodel adapters."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> object:
        """Fit the wrapped model."""
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the wrapped model."""
        ...


@runtime_checkable
class _TinyGPConditionProtocol(Protocol):
    """Protocol for tinygp conditional predictions."""

    loc: np.ndarray


@runtime_checkable
class _TinyGPProtocol(Protocol):
    """Protocol for the tinygp GaussianProcess object used here."""

    def condition(
        self, y: np.ndarray, x: np.ndarray
    ) -> tuple[object, _TinyGPConditionProtocol]:
        """Condition the GP on observed targets."""
        ...


@runtime_checkable
class _SparseMatrixProtocol(Protocol):
    """Protocol for sparse matrices exposing ``toarray``."""

    def toarray(self) -> np.ndarray:
        """Convert the sparse matrix to a dense array."""
        ...


def _as_numpy(values: np.ndarray | xr.DataArray) -> np.ndarray:
    """Return a NumPy view of supported metamodel arrays."""
    return values.values if hasattr(values, "values") else np.asarray(values)


def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute an R^2 score without emitting constant-target warnings."""
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    if y_true_np.size == 0:
        raise_value_error("Cannot compute R^2 for empty targets.")

    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)

    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0
    return float(1 - (ss_res / ss_tot))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE while rejecting empty inputs explicitly."""
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    if y_true_np.size == 0:
        raise_value_error("Cannot compute RMSE for empty targets.")
    return float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)))


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


def calculate_diagnostics(
    model: Metamodel,
    x: ParameterSet,
    y: np.ndarray,
) -> DiagnosticsDict:
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
        y_pred = _as_numpy(model.predict(x))
        r2 = _safe_r2_score(y, y_pred)

    try:
        rmse = model.rmse(x, y)
    except (AttributeError, NotImplementedError):
        # If rmse method is not implemented, calculate it manually
        y_pred = _as_numpy(model.predict(x))
        rmse = _safe_rmse(y, y_pred)

    y_pred = _as_numpy(model.predict(x))
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
        "n_samples": len(y),
    }


def cross_validate(
    model: type[Metamodel],
    x: ParameterSet,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
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
    if cv_folds <= 0:
        raise_value_error("cv_folds must be a positive integer.")
    cv_folds = min(cv_folds, n_samples)

    # Create shuffled indices
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)

    # Calculate fold sizes
    fold_size = n_samples // cv_folds
    remainder = n_samples % cv_folds

    # Store results for each fold
    fold_scores: list[float] = []
    fold_rmse: list[float] = []
    fold_mae: list[float] = []

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
        fold_scores.append(float(diagnostics["r2"]))
        fold_rmse.append(float(diagnostics["rmse"]))
        fold_mae.append(float(diagnostics["mae"]))

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
        "fold_mae": fold_mae,
    }


def _safe_cross_validate_model(
    model: type[Metamodel],
    x: ParameterSet,
    y: np.ndarray,
    cv_folds: int,
) -> tuple[str, dict[str, object]]:
    """Run cross-validation for one model and capture failures."""
    model_name = model.__name__
    try:
        return model_name, cross_validate(model, x, y, cv_folds)
    except Exception as e:
        return model_name, {"error": str(e)}


def _safe_weighted_score(model: Metamodel, x: ParameterSet, y: np.ndarray) -> float:
    """Score a model for ensemble weighting, falling back to zero."""
    try:
        return max(model.score(x, y), 0)
    except Exception:
        return 0.0


def compare_metamodels(
    models: list[type[Metamodel]],
    x: ParameterSet,
    y: np.ndarray,
    cv_folds: int = 5,
) -> dict[str, dict[str, object]]:
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
    results: dict[str, dict[str, object]] = {}

    for model in models:
        model_name, cv_results = _safe_cross_validate_model(model, x, y, cv_folds)
        results[model_name] = cv_results

    return results


class MLP:  # pragma: no cover
    """A simple MLP model."""

    def __init__(self, features: int):
        if not FLAX_AVAILABLE:
            raise_import_error(
                "Flax is required for MLP. Please install it with `pip install flax`."
            )

        self.module = flax_nn.Dense(features=features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return a placeholder Flax module output."""
        return np.asarray(self.module(x))


class FlaxMetamodel:  # pragma: no cover
    """A metamodel that uses a Flax MLP to predict the target values."""

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 100) -> None:
        if not FLAX_AVAILABLE:
            raise_import_error(
                "Flax is required for FlaxMetamodel. Please install it with `pip install flax`."
            )

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

        def train_step(
            state: train_state.TrainState,
            batch_x: np.ndarray,
            batch_y: np.ndarray,
        ) -> tuple[train_state.TrainState, float]:
            def loss_fn(params: object) -> jnp.ndarray:
                logits = state.apply_fn({"params": params}, batch_x)
                return jnp.mean((logits - batch_y) ** 2)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        train_step_jit = jax.jit(train_step)
        for _ in range(self.n_epochs):
            self.state, _ = train_step_jit(self.state, x_np, y_np)

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.state is None:
            raise_runtime_error("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.state.apply_fn({"params": self.state.params}, x_np)
        return np.array(y_pred)

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.state is None:
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return _safe_r2_score(y, y_pred)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.state is None:
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return _safe_rmse(y, y_pred)


class TinyGPMetamodel:  # pragma: no cover
    """A metamodel that uses a tinygp GP to predict the target values."""

    def __init__(self) -> None:
        if not TINYGP_AVAILABLE:
            raise_import_error(
                "tinygp is required for TinyGPMetamodel. Please install it with `pip install tinygp`."
            )

        self.gp: _TinyGPProtocol | None = None
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        self.x_train = np.array(list(x.parameters.values())).T
        self.y_train = y

        kernel = 1.0 * tinygp.kernels.Matern32(1.0)

        def loss(params: dict[str, object]) -> jnp.ndarray:
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
            _loss_val, grads = jax.value_and_grad(loss)(params)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)

        self.gp = tinygp.GaussianProcess(
            params["kernel"], self.x_train, diag=jnp.exp(params["log_diag"])
        )

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.gp is None:
            raise_runtime_error("The model has not been fitted yet.")
        assert self.y_train is not None

        x_np = np.array(list(x.parameters.values())).T
        _, cond = self.gp.condition(self.y_train, x_np)
        return np.asarray(cond.loc)

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.gp is None:
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return _safe_r2_score(y, y_pred)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.gp is None:
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return _safe_rmse(y, y_pred)


class RandomForestMetamodel:
    """A metamodel that uses a Random Forest to predict the target values."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        if not SKLEARN_AVAILABLE:
            raise_import_error(
                "scikit-learn is required for RandomForestMetamodel. "
                "Please install it with `pip install scikit-learn`."
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: _PredictorProtocol | None = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        x_np = np.array(list(x.parameters.values())).T

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(x_np, y)

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.model is None:
            raise_runtime_error("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        return np.asarray(self.model.predict(x_np))

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.model is None:
            raise_runtime_error("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.model.predict(x_np)
        return _safe_r2_score(y, y_pred)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.model is None:
            raise_runtime_error("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.model.predict(x_np)
        return _safe_rmse(y, y_pred)


class GAMMetamodel:
    """A metamodel that uses a Generalized Additive Model to predict the target values."""

    def __init__(self, n_splines: int = 10, lam: float = 0.1) -> None:
        if not PYGAM_AVAILABLE and not SKLEARN_AVAILABLE:
            raise_import_error("pygam or scikit-learn is required for GAMMetamodel.")

        # Ensure n_splines is greater than the default spline_order (3)
        if n_splines <= 3:
            n_splines = 4

        self.n_splines = n_splines
        self.lam = lam
        self.model: _PredictorProtocol | None = None
        self._use_fallback = not PYGAM_AVAILABLE

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        x_np = np.array(list(x.parameters.values())).T

        if self._use_fallback:
            self.model = LinearRegression()
            self.model.fit(x_np, y)
            return

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
            raise_runtime_error("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        return np.asarray(self.model.predict(x_np))

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.model is None:
            raise_runtime_error("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.model.predict(x_np)
        return _safe_r2_score(y, y_pred)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.model is None:
            raise_runtime_error("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        y_pred = self.model.predict(x_np)
        return _safe_rmse(y, y_pred)


class BARTMetamodel:
    """A metamodel that uses a BART (Bayesian Additive Regression Trees) to predict the target values."""

    def __init__(
        self, num_trees: int = 50, alpha: float = 0.95, beta: float = 2.0
    ) -> None:
        if not PYMC_AVAILABLE and not SKLEARN_AVAILABLE:
            raise_import_error(
                "pymc/pymc-bart or scikit-learn is required for BARTMetamodel."
            )

        self.num_trees = num_trees
        self.alpha = alpha
        self.beta = beta
        self.model: object | None = None
        self.trace: object | None = None
        self._use_fallback = not PYMC_AVAILABLE

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        x_np = np.array(list(x.parameters.values())).T

        if self._use_fallback:
            model = RandomForestRegressor(n_estimators=self.num_trees, random_state=42)
            model.fit(x_np, y)
            self.model = model
            self.trace = "fallback"
            return

        with pm.Model() as model:
            # BART prior using pymc-bart
            mu = pmb.BART("mu", X=x_np, Y=y, m=self.num_trees)
            # Likelihood
            sigma = pm.HalfNormal("sigma", 1)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Sample from the posterior
            self.trace = pm.sample(
                500,
                tune=500,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
            )

        self.model = model

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.model is None or self.trace is None:
            raise_runtime_error("The model has not been fitted yet.")

        if self._use_fallback:
            x_np = np.array(list(x.parameters.values())).T
            model = cast("_PredictorProtocol", self.model)
            return np.asarray(model.predict(x_np))

        # Use the posterior predictive to make predictions
        with cast("pm.Model", self.model):
            post_pred = pm.sample_posterior_predictive(
                self.trace, var_names=["mu"], random_seed=42
            )

        # Return the mean prediction
        return np.asarray(np.mean(post_pred.posterior_predictive["mu"], axis=(0, 1)))

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.model is None or self.trace is None:
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        # Convert to numpy array if it's an xarray DataArray
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        return _safe_r2_score(y, y_pred)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.model is None or self.trace is None:
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        # Convert to numpy array if it's an xarray DataArray
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values
        return _safe_rmse(y, y_pred)


class ActiveLearningMetamodel:
    """A metamodel that uses active learning to iteratively improve its predictions."""

    def __init__(
        self,
        base_model: Metamodel,
        n_initial_samples: int = 10,
        n_query_samples: int = 5,
        acquisition_function: str = "uncertainty",
    ) -> None:
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
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.iteration = 0

    def _select_initial_samples(
        self,
        x_pool: np.ndarray,
        y_pool: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Select initial samples randomly from the pool."""
        n_samples = min(self.n_initial_samples, len(x_pool))
        indices = np.random.choice(len(x_pool), n_samples, replace=False)
        x_selected = x_pool[indices]
        y_selected = y_pool[indices] if y_pool is not None else None
        return x_selected, y_selected, indices

    def _acquisition_uncertainty(self, x_pool: np.ndarray) -> np.ndarray:
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
            return np.asarray(np.var(np.stack(predictions, axis=0), axis=0))
        except Exception:
            # Fallback: use a simple distance-based uncertainty
            if self.X_train is not None:
                # Calculate distances to existing training points
                distances = [
                    np.linalg.norm(x_pool - x_train, axis=1) for x_train in self.X_train
                ]
                min_distances = np.min(np.stack(distances, axis=0), axis=0)
                return np.array([float(value) for value in min_distances], dtype=float)
            # If no training data, select randomly
            return np.array(
                [float(value) for value in np.random.rand(len(x_pool))], dtype=float
            )

    def _acquisition_random(self, x_pool: np.ndarray) -> np.ndarray:
        """Select samples randomly."""
        return np.random.rand(len(x_pool))

    def _acquisition_margin(self, x_pool: np.ndarray) -> np.ndarray:
        """Select samples with smallest margin (closest to decision boundary)."""
        # This is more relevant for classification, but we can adapt it for regression
        # by looking at predictions close to some threshold or with high variance
        return self._acquisition_uncertainty(x_pool)

    def _select_query_samples(
        self,
        x_pool: np.ndarray,
        y_pool: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Select samples to query based on the acquisition function."""
        if self.acquisition_function == "uncertainty":
            scores = self._acquisition_uncertainty(x_pool)
        elif self.acquisition_function == "random":
            scores = self._acquisition_random(x_pool)
        elif self.acquisition_function == "margin":
            scores = self._acquisition_margin(x_pool)
        else:
            raise_value_error(
                f"Unknown acquisition function: {self.acquisition_function}"
            )

        # Select samples with highest scores
        n_query = min(self.n_query_samples, len(x_pool))
        indices = np.argpartition(scores, -n_query)[-n_query:]
        x_selected = x_pool[indices]
        y_selected = y_pool[indices] if y_pool is not None else None
        return x_selected, y_selected, indices

    def _array_to_parameterset(self, x_array: np.ndarray) -> ParameterSet:
        """Convert numpy array to ParameterSet."""
        n_samples = x_array.shape[0]
        param_dict = {}
        for i in range(x_array.shape[1]):
            param_dict[f"param_{i}"] = x_array[:, i]

        dataset = xr.Dataset(
            {k: ("n_samples", v) for k, v in param_dict.items()},
            coords={"n_samples": np.arange(n_samples)},
        )
        return ParameterSet(dataset=dataset)

    def _parameterset_to_array(self, param_set: ParameterSet) -> np.ndarray:
        """Convert ParameterSet to numpy array."""
        return np.stack(list(param_set.parameters.values()), axis=1)

    def fit(
        self,
        x: ParameterSet,
        y: np.ndarray,
        x_pool: np.ndarray | None = None,
        y_pool: np.ndarray | None = None,
        n_iterations: int = 5,
    ) -> None:
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
                x_new, y_new, _new_indices = self._select_query_samples(x_pool, y_pool)
            else:
                # Otherwise, just select based on acquisition function
                x_new, _, _new_indices = self._select_query_samples(x_pool, None)
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
            raise_runtime_error("The model has not been fitted yet.")

        return np.asarray(self.base_model.predict(x))

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
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return _safe_r2_score(y, y_pred)

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
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return _safe_rmse(y, y_pred)


class EnsembleMetamodel:
    """A metamodel that combines predictions from multiple metamodels."""

    def __init__(self, models: list[Metamodel], method: str = "mean") -> None:
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
        self.weights: list[float] | None = None

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
        if self.method == "weighted":
            scores = [_safe_weighted_score(model, x, y) for model in self.models]

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
            raise_runtime_error("No models in the ensemble.")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)

        # Combine predictions based on the ensemble method
        if self.method == "mean":
            return np.asarray(np.mean(predictions, axis=0))
        if self.method == "median":
            return np.asarray(np.median(predictions, axis=0))
        if self.method == "weighted":
            if self.weights is None:
                # If weights haven't been computed, use equal weights
                weights = [1.0 / len(self.models)] * len(self.models)
            else:
                weights = self.weights
            return np.average(predictions, axis=0, weights=weights)
        return raise_value_error(f"Unknown ensemble method: {self.method}")

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
        return _safe_r2_score(y, y_pred)

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
        return _safe_rmse(y, y_pred)


class PyTorchNNMetamodel:  # pragma: no cover
    """A metamodel that uses a PyTorch neural network to predict the target values."""

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        learning_rate: float = 0.001,
        n_epochs: int = 1000,
        batch_size: int = 32,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise_import_error(
                "PyTorch is required for PyTorchNNMetamodel. "
                "Please install it with `pip install torch`."
            )

        if hidden_layers is None:
            hidden_layers = [64, 32]
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model: torch_nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim: int, output_dim: int) -> torch_nn.Module:
        """Build the neural network architecture."""
        layers = []
        # Input layer
        layers.append(torch_nn.Linear(input_dim, self.hidden_layers[0]))
        layers.append(torch_nn.ReLU())

        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            layers.append(
                torch_nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1])
            )
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
                print(f"Epoch [{epoch + 1}/{self.n_epochs}], Loss: {loss.item():.4f}")

    def predict(self, x: ParameterSet) -> np.ndarray:
        """Predict the target values for the given input parameters."""
        if self.model is None:
            raise_runtime_error("The model has not been fitted yet.")

        # Convert data to torch tensor
        x_np = np.array(list(x.parameters.values())).T
        x_tensor = torch.FloatTensor(x_np).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_tensor)

        # Convert back to numpy array
        return np.asarray(predictions.cpu().numpy().flatten())

    def score(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if self.model is None:
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return _safe_r2_score(y, y_pred)

    def rmse(self, x: ParameterSet, y: np.ndarray) -> float:
        """Return the root mean squared error of the prediction."""
        if self.model is None:
            raise_runtime_error("The model has not been fitted yet.")

        y_pred = self.predict(x)
        return _safe_rmse(y, y_pred)
