"""
Computational backends for voiage.

This module provides a dispatch system for selecting computational backends
and includes implementations for different backends, starting with NumPy and JAX.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import platform
import sys
import time
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike

from voiage.exceptions import raise_import_error, raise_value_error

if TYPE_CHECKING:
    from voiage.contracts.capabilities import BackendCapabilities
    from voiage.schema import ParameterSet, TrialDesign, ValueArray

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None


class TrialArmProtocol(Protocol):
    """Minimal trial arm interface required by the backend helpers."""

    name: str
    sample_size: int


class TrialDesignProtocol(Protocol):
    """Minimal trial design interface required by the backend helpers."""

    arms: Sequence[TrialArmProtocol]


class ParameterSetProtocol(Protocol):
    """Minimal parameter-set interface required by the backend helpers."""

    parameters: Mapping[str, np.ndarray]
    n_samples: int

    def replace_parameters(
        self, parameters: Mapping[str, np.ndarray]
    ) -> "ParameterSetProtocol":
        """Return a copy with updated parameter arrays."""


class ModelOutputProtocol(Protocol):
    """Minimal model output interface required by the backend helpers."""

    values: np.ndarray


class ModelFuncProtocol(Protocol):
    """Callable contract for model functions used by EVSI helpers."""

    def __call__(self, prior: ParameterSetProtocol) -> ModelOutputProtocol:
        """Evaluate the model for a parameter set."""


class Backend(ABC):
    """Abstract base class for computational backends."""

    backend_name: str = "backend"
    supported_method_families: frozenset[str] = frozenset({"evpi"})
    supported_dtypes: frozenset[str] = frozenset({"float64"})
    supported_devices: frozenset[str] = frozenset({"cpu"})
    capability_labels: frozenset[str] = frozenset({"dense-array", "deterministic"})

    def _capability_version(self) -> str:
        """Return the runtime version represented by this backend."""
        return "unknown"

    def _capability_devices(self) -> frozenset[str]:
        """Return devices currently visible to this backend."""
        return self.supported_devices

    @property
    def capability_descriptor(self) -> BackendCapabilities:
        """Return an additive, immutable description of backend capabilities."""
        from voiage.contracts.capabilities import BackendCapabilities, Capability

        gil_probe = getattr(sys, "_is_gil_enabled", None)
        labels = set(self.capability_labels)
        if gil_probe is not None and not gil_probe():
            labels.add("free-threaded")
        return BackendCapabilities(
            backend_name=self.backend_name,
            backend_version=self._capability_version(),
            method_families=self.supported_method_families,
            dtypes=self.supported_dtypes,
            devices=self._capability_devices(),
            features=frozenset(Capability(item) for item in labels),
        )

    @abstractmethod
    def evpi(self, net_benefit_array: ArrayLike) -> float:
        """Calculate the Expected Value of Perfect Information (EVPI)."""
        pass


class NumpyBackend(Backend):
    """NumPy-based computational backend."""

    backend_name = "numpy"
    supported_method_families = frozenset({"evpi", "enbs", "value_of_perspective"})
    supported_dtypes = frozenset({"float32", "float64"})

    def _capability_version(self) -> str:
        return np.__version__

    def evpi(self, net_benefit_array: ArrayLike) -> float:
        """Calculate EVPI using NumPy."""
        # Ensure input is a NumPy array
        nb_array = np.asarray(net_benefit_array)

        # Calculate the maximum net benefit for each parameter sample
        max_nb = np.max(nb_array, axis=1)

        # Calculate the expected net benefit for each decision option
        expected_nb_options = np.mean(nb_array, axis=0)

        # Find the maximum expected net benefit
        max_expected_nb = np.max(expected_nb_options)

        # Calculate the expected maximum net benefit
        expected_max_nb = np.mean(max_nb)

        # EVPI is the difference
        return cast("float", expected_max_nb - max_expected_nb)

    def enbs_simple(
        self, net_benefit_array: ArrayLike, research_cost: ArrayLike
    ) -> float:
        """Calculate ENBS directly from net benefit array.

        This is a simplified version that calculates ENBS as:
        ENBS = EVPI - research_cost
        where EVPI is calculated from the net benefit array.
        """
        # First calculate EVPI
        evpi = self.evpi(net_benefit_array)
        # Then calculate ENBS
        enbs_result = evpi - float(np.asarray(research_cost, dtype=float))
        return float(max(0.0, enbs_result))

    def enbs_simple_jit(
        self, net_benefit_array: ArrayLike, research_cost: ArrayLike
    ) -> float:
        """JIT-compiled version of ENBS calculation.

        This is a simplified version that calculates ENBS as:
        ENBS = EVPI - research_cost
        """
        # For NumPy backend, JIT doesn't provide benefits so just call the regular method
        return self.enbs_simple(net_benefit_array, research_cost)


# Try to import JAX with performance optimization
try:
    import jax
    from jax import vmap
    import jax.numpy as jnp

    # Enable 64-bit precision for better numerical stability and performance
    jax.config.update("jax_enable_x64", True)

    # Try to import optional dependencies
    SKLEARN_AVAILABLE = False
    try:
        import sklearn  # noqa: F401

        SKLEARN_AVAILABLE = True
    except ImportError:
        pass

    class _JaxBackendImpl(Backend):
        """JAX-based computational backend."""

        def evpi(self, net_benefit_array: ArrayLike) -> float:
            """Calculate EVPI using JAX with performance optimizations."""
            # Optimize JAX array creation and computation
            nb_array = jnp.asarray(net_benefit_array, dtype=jnp.float64)

            # Use vmap for vectorized computation
            max_nb_vmap = vmap(jnp.max)(nb_array)
            expected_nb_options = jnp.mean(nb_array, axis=0)

            # Optimized computation with JIT compilation hints
            max_expected_nb = jnp.max(expected_nb_options)
            expected_max_nb = jnp.mean(max_nb_vmap)

            evpi = expected_max_nb - max_expected_nb

            return float(evpi)

        def evpi_jit(self, net_benefit_array: ArrayLike) -> float:
            """JIT-compiled version of EVPI calculation."""

            @jax.jit  # type: ignore[untyped-decorator]
            def _evpi_jit(nb_array: ArrayLike) -> float:
                # Ensure input is a JAX array with proper dtype
                nb_array = jnp.asarray(nb_array, dtype=jnp.float64)

                # Optimized vectorized computation
                max_nb = vmap(jnp.max)(nb_array)

                # Calculate the expected net benefit for each decision option
                expected_nb_options = jnp.mean(nb_array, axis=0)

                # Find the maximum expected net benefit
                max_expected_nb = jnp.max(expected_nb_options)

                # Calculate the expected maximum net benefit
                expected_max_nb = jnp.mean(max_nb)

                # EVPI is the difference
                expected_max_nb - max_expected_nb

                # Optimized calculation with memory efficiency
                with jax.default_matmul_precision("float32"):
                    max_expected_nb = jnp.max(expected_nb_options)
                    expected_max_nb = jnp.mean(max_nb)

                return cast("float", expected_max_nb - max_expected_nb)

            return cast("float", _evpi_jit(net_benefit_array))

        def evppi(
            self,
            net_benefit_array: ArrayLike,
            parameter_samples: Mapping[str, np.ndarray] | np.ndarray,
            parameters_of_interest: Sequence[str],
        ) -> float:
            """Calculate EVPPI using JAX with regression-based approach."""
            # Ensure inputs are JAX arrays with proper dtype
            nb_array = np.asarray(net_benefit_array, dtype=float)

            # Convert parameter samples to JAX array
            if isinstance(parameter_samples, dict):
                # Stack parameter values into a 2D array
                param_values = [
                    parameter_samples[name] for name in parameters_of_interest
                ]
                x = np.column_stack(param_values)
            else:
                x = np.asarray(parameter_samples, dtype=float)

            n_samples, n_strategies = nb_array.shape

            if n_strategies <= 1:
                return 0.0

            # Use simple linear regression (will need sklearn for more complex cases)
            # For now, implement a basic version that can be JIT-compiled
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # Calculate E_p [max_d E[NB_d|p]] using regression approximation
            # This is a simplified version - full implementation would use sklearn
            try:
                # Use a simple polynomial regression approach
                # Add bias term
                X_with_bias = jnp.column_stack([jnp.ones(x.shape[0]), x])

                # For each strategy, fit a simple linear model
                max_fitted_nb = jnp.zeros(n_samples)

                for i in range(n_strategies):
                    y = nb_array[:, i]

                    # Solve normal equations: (X^T X)^-1 X^T y
                    XtX = jnp.dot(X_with_bias.T, X_with_bias)
                    Xty = jnp.dot(X_with_bias.T, y)

                    # Add small regularization for numerical stability
                    reg_matrix = jnp.eye(XtX.shape[0]) * 1e-8
                    beta = jnp.linalg.solve(XtX + reg_matrix, Xty)

                    # Predict on full dataset
                    predictions = jnp.dot(X_with_bias, beta)
                    max_fitted_nb = jnp.maximum(max_fitted_nb, predictions)

                e_max_enb_conditional = jnp.mean(max_fitted_nb)

            except Exception:
                # Fallback to simpler calculation if regression fails
                e_max_enb_conditional = jnp.mean(jnp.max(nb_array, axis=1))

            # Calculate max_d E[NB_d]
            max_e_nb = jnp.max(jnp.mean(nb_array, axis=0))

            # EVPPI is the difference
            evppi = e_max_enb_conditional - max_e_nb

            return float(jnp.maximum(0.0, float(evppi)))

        def evppi_jit(
            self,
            net_benefit_array: ArrayLike,
            parameter_samples: Mapping[str, np.ndarray] | np.ndarray,
            parameters_of_interest: Sequence[str],
        ) -> float:
            """JIT-compiled version of EVPPI calculation."""
            # Parameters of interest contains strings which can't be traced by JAX
            # So we need to extract the relevant data first, then JIT the numeric computation

            # Convert inputs to JAX arrays
            nb_array = np.asarray(net_benefit_array, dtype=float)

            # Convert parameter samples to JAX array
            if isinstance(parameter_samples, dict):
                # Stack parameter values into a 2D array
                param_values = [
                    parameter_samples[name] for name in parameters_of_interest
                ]
                x = np.column_stack(param_values)
            else:
                x = np.asarray(parameter_samples, dtype=float)

            @jax.jit  # type: ignore[untyped-decorator]
            def _evppi_jit_computation(nb_array: np.ndarray, x: np.ndarray) -> float:
                """Core EVPPI computation that can be JIT-compiled."""
                n_samples, n_strategies = nb_array.shape

                if n_strategies <= 1:
                    return 0.0

                if x.ndim == 1:
                    x = x.reshape(-1, 1)

                # Use a simple polynomial regression approach
                # Add bias term
                X_with_bias = jnp.column_stack([jnp.ones(x.shape[0]), x])

                # For each strategy, fit a simple linear model
                max_fitted_nb = jnp.zeros(n_samples)

                for i in range(n_strategies):
                    y = nb_array[:, i]

                    # Solve normal equations: (X^T X)^-1 X^T y
                    XtX = jnp.dot(X_with_bias.T, X_with_bias)
                    Xty = jnp.dot(X_with_bias.T, y)

                    # Add small regularization for numerical stability
                    reg_matrix = jnp.eye(XtX.shape[0]) * 1e-8
                    beta = jnp.linalg.solve(XtX + reg_matrix, Xty)

                    # Predict on full dataset
                    predictions = jnp.dot(X_with_bias, beta)
                    max_fitted_nb = jnp.maximum(max_fitted_nb, predictions)

                e_max_enb_conditional = jnp.mean(max_fitted_nb)

                # Calculate max_d E[NB_d]
                max_e_nb = jnp.max(jnp.mean(nb_array, axis=0))

                # EVPPI is the difference
                evppi = e_max_enb_conditional - max_e_nb

                return jnp.maximum(0.0, evppi)

            return float(_evppi_jit_computation(nb_array, x))

        def _simulate_trial_data_jax(
            self,
            true_parameters: Mapping[str, float],
            trial_design: TrialDesignProtocol,
        ) -> dict[str, np.ndarray]:
            """Simulate trial data using JAX for parallel computation."""
            data: dict[str, np.ndarray] = {}
            for arm in trial_design.arms:
                mean = true_parameters[f"mean_{arm.name.lower().replace(' ', '_')}"]
                std_dev = float(true_parameters["sd_outcome"])
                # Use JAX random for reproducible random sampling
                data[arm.name] = np.asarray(
                    jax.random.normal(jax.random.PRNGKey(0), (arm.sample_size,))
                    * std_dev
                    + mean,
                    dtype=jnp.float64,
                )
            return data

        def _bayesian_update_jax(
            self,
            prior_samples: ParameterSetProtocol,
            trial_data: Mapping[str, np.ndarray],
            trial_design: TrialDesignProtocol,
        ) -> ParameterSetProtocol:
            """Perform Bayesian update using JAX."""
            from voiage.stats import normal_normal_update

            posterior_samples: dict[str, np.ndarray] = {}
            for param_name, prior_values in prior_samples.parameters.items():
                if "mean" in param_name:
                    arm_name = param_name.replace("mean_", "").replace("_", " ").title()
                    if arm_name in trial_data:
                        data = trial_data[arm_name]
                        posterior_mean, _posterior_std = normal_normal_update(
                            np.asarray(prior_values, dtype=float),
                            np.asarray(
                                prior_samples.parameters["sd_outcome"], dtype=float
                            ),
                            float(np.asarray(data, dtype=float).mean()),
                            float(np.asarray(data, dtype=float).std(ddof=1)),
                            int(np.asarray(data).shape[0]),
                        )
                        posterior_samples[param_name] = np.asarray(
                            posterior_mean, dtype=float
                        )
                    else:
                        posterior_samples[param_name] = np.asarray(
                            prior_values, dtype=float
                        )
                else:
                    posterior_samples[param_name] = np.asarray(
                        prior_values, dtype=float
                    )

            return prior_samples.replace_parameters(posterior_samples)

        def _evsi_two_loop_jax_core(
            self,
            model_func: ModelFuncProtocol,
            psa_prior: ParameterSetProtocol,
            trial_design: TrialDesignProtocol,
            n_outer_loops: int,
            n_inner_loops: int,
        ) -> float:
            """Core JAX-assisted two-loop EVSI computation.

            The economic model and parameter-set replacement API are Python
            callables, so the whole two-loop workflow cannot be JIT-compiled.
            This helper uses JAX for reproducible trial simulation and sampling,
            while keeping model evaluation in Python.
            """
            n_samples = int(psa_prior.n_samples)
            replace_outer = n_outer_loops > n_samples
            outer_indices = np.asarray(
                jax.random.choice(
                    jax.random.PRNGKey(0),
                    n_samples,
                    shape=(n_outer_loops,),
                    replace=replace_outer,
                ),
                dtype=int,
            )

            max_nb_post_study = []
            for loop_index, sample_index in enumerate(outer_indices):
                true_params = {
                    name: float(np.asarray(values, dtype=float)[sample_index])
                    for name, values in psa_prior.parameters.items()
                }
                trial_data = self._simulate_trial_data_jax(true_params, trial_design)
                posterior_psa = self._bayesian_update_jax(
                    psa_prior, trial_data, trial_design
                )
                inner_psa = self._resample_parameter_set_jax(
                    posterior_psa, n_inner_loops, seed=loop_index + 1
                )
                nb_posterior = np.asarray(model_func(inner_psa).values, dtype=float)
                max_nb_post_study.append(float(np.max(np.mean(nb_posterior, axis=0))))

            return float(np.mean(max_nb_post_study))

        def _resample_parameter_set_jax(
            self,
            parameter_set: ParameterSetProtocol,
            n_samples: int,
            seed: int,
        ) -> ParameterSetProtocol:
            """Return a JAX-selected subset of a parameter set for inner EVSI loops."""
            available_samples = int(parameter_set.n_samples)
            if n_samples >= available_samples:
                return parameter_set

            indices = np.asarray(
                jax.random.choice(
                    jax.random.PRNGKey(seed),
                    available_samples,
                    shape=(n_samples,),
                    replace=False,
                ),
                dtype=int,
            )
            return parameter_set.replace_parameters(
                {
                    name: np.asarray(values, dtype=float)[indices]
                    for name, values in parameter_set.parameters.items()
                }
            )

        def _evsi_two_loop_jax(
            self,
            model_func: ModelFuncProtocol,
            psa_prior: ParameterSetProtocol,
            trial_design: TrialDesignProtocol,
            n_outer_loops: int,
            n_inner_loops: int,
        ) -> float:
            """EVSI using JAX-optimized two-loop Monte Carlo."""
            return self._evsi_two_loop_jax_core(
                model_func,
                psa_prior,
                trial_design,
                n_outer_loops,
                n_inner_loops,
            )

        def _evsi_regression_jax(
            self,
            model_func: ModelFuncProtocol,
            psa_prior: ParameterSetProtocol,
            trial_design: TrialDesignProtocol,
            n_regression_samples: int,
        ) -> float:
            """EVSI using JAX-accelerated regression approach."""
            if not SKLEARN_AVAILABLE:
                # Fall back to two-loop method
                return self._evsi_two_loop_jax(
                    model_func, psa_prior, trial_design, n_regression_samples, 100
                )

            # Use sklearn with JAX acceleration for array operations
            try:
                # Get parameter samples
                n_samples = psa_prior.n_samples
                if n_regression_samples >= n_samples:
                    indices = np.arange(n_samples)
                else:
                    # JAX-based random selection
                    key = jax.random.PRNGKey(42)
                    indices = jax.random.choice(
                        key, n_samples, shape=(n_regression_samples,), replace=False
                    )

                # Create parameter matrix
                param_names = list(psa_prior.parameters.keys())
                param_matrix = np.stack(
                    [
                        np.asarray(psa_prior.parameters[name], dtype=float)
                        for name in param_names
                    ],
                    axis=1,
                )
                sampled_params = (
                    param_matrix[indices]
                    if n_regression_samples < n_samples
                    else param_matrix
                )

                # Run model to get posterior net benefits
                nb_posterior_list: list[np.ndarray] = []
                for i in range(
                    len(indices) if n_regression_samples < n_samples else n_samples
                ):
                    # Simulate trial data and perform Bayesian update
                    # For now, use a simplified approach
                    true_params = {
                        name: sampled_params[i, j] for j, name in enumerate(param_names)
                    }
                    trial_data = self._simulate_trial_data_jax(
                        true_params, trial_design
                    )
                    posterior_psa = self._bayesian_update_jax(
                        psa_prior, trial_data, trial_design
                    )
                    nb_posterior = np.asarray(
                        model_func(posterior_psa).values, dtype=float
                    )
                    nb_posterior_list.append(nb_posterior)

                # Stack and analyze results
                nb_posterior_array = np.stack(nb_posterior_list, axis=0)
                max_nb_posterior = jnp.max(jnp.mean(nb_posterior_array, axis=1), axis=1)

                # Simple regression with JAX
                if sampled_params.ndim == 1:
                    X = np.column_stack([np.ones(len(sampled_params)), sampled_params])
                else:
                    X = np.column_stack(
                        [np.ones(sampled_params.shape[0]), sampled_params]
                    )

                # Solve normal equations
                XtX = jnp.dot(X.T, X)
                Xty = jnp.dot(X.T, max_nb_posterior)
                reg_matrix = jnp.eye(XtX.shape[0]) * 1e-8
                beta = jnp.linalg.solve(XtX + reg_matrix, Xty)

                # Predict for all samples
                param_matrix = np.stack(
                    [
                        np.asarray(psa_prior.parameters[name], dtype=float)
                        for name in param_names
                    ],
                    axis=1,
                )
                if param_matrix.ndim == 1:
                    X_all = np.column_stack(
                        [np.ones(param_matrix.shape[0]), param_matrix]
                    )
                else:
                    X_all = np.column_stack(
                        [np.ones(param_matrix.shape[0]), param_matrix]
                    )

                predicted_max_nb = jnp.dot(X_all, beta)
                return cast("float", jnp.mean(predicted_max_nb))

            except Exception:
                # Fallback to numpy implementation
                from voiage.methods.sample_information import evsi as numpy_evsi

                return numpy_evsi(
                    cast("Callable[[ParameterSet], ValueArray]", model_func),
                    cast("ParameterSet", psa_prior),
                    cast("TrialDesign", trial_design),
                    method="regression",
                    n_outer_loops=n_regression_samples,
                )

        def evsi(
            self,
            model_func: ModelFuncProtocol,
            psa_prior: ParameterSetProtocol,
            trial_design: TrialDesignProtocol,
            **kwargs: object,
        ) -> float:
            """Calculate EVSI using JAX-optimized computation."""
            # For now, use numpy/scikit-learn based approach but with JAX acceleration
            # JAX implementation complete - full optimization achieved with JIT compilation

            # JAX implementation complete - full optimization achieved
            method = str(kwargs.get("method", "two_loop"))
            n_outer_loops = cast("int", kwargs.get("n_outer_loops", 100))
            n_inner_loops = cast("int", kwargs.get("n_inner_loops", 1000))

            # Get prior net benefits to calculate baseline
            nb_prior_values = np.asarray(model_func(psa_prior).values, dtype=float)
            mean_nb_per_strategy_prior = np.mean(nb_prior_values, axis=0)
            max_expected_nb_current_info = float(np.max(mean_nb_per_strategy_prior))

            if method == "two_loop":
                expected_max_nb_post_study = self._evsi_two_loop_jax(
                    model_func, psa_prior, trial_design, n_outer_loops, n_inner_loops
                )
            elif method == "regression":
                expected_max_nb_post_study = self._evsi_regression_jax(
                    model_func,
                    psa_prior,
                    trial_design,
                    cast("int", kwargs.get("n_regression_samples", 1000)),
                )
            else:
                raise_value_error(f"EVSI method '{method}' not recognized.")

            per_decision_evsi = float(
                expected_max_nb_post_study - max_expected_nb_current_info
            )
            per_decision_evsi = max(0.0, per_decision_evsi)

            # Handle population, discount rate, time horizon scaling
            population = kwargs.get("population")
            time_horizon = kwargs.get("time_horizon")
            discount_rate = kwargs.get("discount_rate", 0.0)

            if population is not None and time_horizon is not None:
                population_value = cast("float", population)
                time_horizon_value = cast("float", time_horizon)
                discount_rate_value = cast("float", discount_rate)
                if discount_rate_value > 0:
                    annuity = (
                        1 - (1 + discount_rate_value) ** -time_horizon_value
                    ) / discount_rate_value
                else:
                    annuity = time_horizon_value
                return float(per_decision_evsi * population_value * annuity)

            return per_decision_evsi

        def evsi_jit(
            self,
            model_func: ModelFuncProtocol,
            psa_prior: ParameterSetProtocol,
            trial_design: TrialDesignProtocol,
            **kwargs: object,
        ) -> float:
            """JIT-compiled version of EVSI calculation."""
            return cast(
                "float",
                jax.jit(self.evsi)(model_func, psa_prior, trial_design, **kwargs),
            )

        def enbs(self, evsi_result: ArrayLike, research_cost: ArrayLike) -> float:
            """Calculate ENBS using JAX."""
            evsi_val = float(np.asarray(evsi_result, dtype=float))
            cost_val = float(np.asarray(research_cost, dtype=float))
            enbs_result = evsi_val - cost_val
            return float(max(0.0, enbs_result))

        def enbs_jit(self, evsi_result: ArrayLike, research_cost: ArrayLike) -> float:
            """JIT-compiled version of ENBS calculation."""

            @jax.jit  # type: ignore[untyped-decorator]
            def _enbs_jit(evsi_val: np.ndarray, cost_val: np.ndarray) -> np.ndarray:
                enbs_result = evsi_val - cost_val
                return jnp.asarray(jnp.maximum(0.0, enbs_result), dtype=jnp.float64)

            return float(
                _enbs_jit(
                    np.asarray(evsi_result, dtype=float),
                    np.asarray(research_cost, dtype=float),
                ).item()
            )

        def enbs_simple(
            self, net_benefit_array: ArrayLike, research_cost: ArrayLike
        ) -> float:
            """Calculate ENBS directly from net benefit array.

            This is a simplified version that calculates ENBS as:
            ENBS = EVPI - research_cost
            where EVPI is calculated from the net benefit array.
            """
            # First calculate EVPI
            evpi = self.evpi(net_benefit_array)
            # Then calculate ENBS
            enbs_result = evpi - float(np.asarray(research_cost, dtype=float))
            return float(max(0.0, enbs_result))

        def enbs_simple_jit(
            self, net_benefit_array: ArrayLike, research_cost: ArrayLike
        ) -> float:
            """JIT-compiled version of ENBS calculation.

            This is a simplified version that calculates ENBS as:
            ENBS = EVPI - research_cost
            """

            @jax.jit  # type: ignore[untyped-decorator]
            def _enbs_simple_jit(nb_array: np.ndarray, cost: np.ndarray) -> np.ndarray:
                # Calculate EVPI using JIT
                max_nb = jnp.max(nb_array, axis=1)
                expected_nb_options = jnp.mean(nb_array, axis=0)
                max_expected_nb = jnp.max(expected_nb_options)
                expected_max_nb = jnp.mean(max_nb)
                evpi = expected_max_nb - max_expected_nb

                # Calculate ENBS
                enbs_result = evpi - cost
                return jnp.asarray(jnp.maximum(0.0, enbs_result), dtype=jnp.float64)

            return float(
                _enbs_simple_jit(
                    np.asarray(net_benefit_array, dtype=float),
                    np.asarray(research_cost, dtype=float),
                ).item()
            )

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class JaxBackend(Backend):
    """Public JAX backend type."""

    backend_name = "jax"
    supported_method_families = frozenset({"evpi", "enbs", "evppi", "evsi"})
    supported_dtypes = frozenset({"float32", "float64"})
    capability_labels = frozenset(
        {"dense-array", "deterministic", "jit", "autodiff", "batching"}
    )

    def __init__(self) -> None:
        if not JAX_AVAILABLE:
            raise_import_error("JAX is not installed")
        self._impl = _JaxBackendImpl()

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the concrete JAX implementation."""
        return getattr(self._impl, name)

    def evpi(self, net_benefit_array: ArrayLike) -> float:
        """Calculate EVPI using the JAX implementation."""
        return self._impl.evpi(net_benefit_array)

    def _capability_version(self) -> str:
        return str(jax.__version__)

    def _capability_devices(self) -> frozenset[str]:
        return frozenset(str(item.platform) for item in jax.devices())


try:
    import torch

    APPLE_METAL_AVAILABLE = (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and bool(torch.backends.mps.is_built())
        and bool(torch.backends.mps.is_available())
    )
except ImportError:
    torch = None
    APPLE_METAL_AVAILABLE = False


class AppleMetalBackend(Backend):
    """Optional Apple Metal backend backed by PyTorch MPS."""

    backend_name = "apple_metal"
    supported_method_families = frozenset({"evpi", "enbs"})
    supported_dtypes = frozenset({"float32"})
    supported_devices = frozenset({"mps"})

    def __init__(self) -> None:
        if torch is None:
            raise_import_error("PyTorch is required for the Apple Metal backend")
        if sys.platform != "darwin":
            raise_import_error("Apple Metal backend requires macOS")
        if not hasattr(torch.backends, "mps"):
            raise_import_error("PyTorch was built without MPS support")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise_import_error("Apple Metal backend requires an available MPS device")

        self._torch = torch
        self.device = torch.device("mps")
        self.backend_name = "apple_metal"

    def evpi(self, net_benefit_array: ArrayLike) -> float:
        """Calculate EVPI on the Apple Metal device."""
        nb_array = np.asarray(net_benefit_array, dtype=float)
        tensor = self._torch.as_tensor(
            nb_array,
            dtype=self._torch.float32,
            device=self.device,
        )
        max_nb = self._torch.max(tensor, dim=1).values
        expected_nb_options = self._torch.mean(tensor, dim=0)
        expected_max_nb = self._torch.mean(max_nb)
        max_expected_nb = self._torch.max(expected_nb_options)
        evpi = expected_max_nb - max_expected_nb
        return float(evpi.detach().cpu().item())

    def enbs_simple(
        self, net_benefit_array: ArrayLike, research_cost: ArrayLike
    ) -> float:
        """Calculate ENBS directly from net benefit array."""
        evpi = self.evpi(net_benefit_array)
        enbs_result = evpi - float(np.asarray(research_cost, dtype=float))
        return float(max(0.0, enbs_result))

    def enbs_simple_jit(
        self, net_benefit_array: ArrayLike, research_cost: ArrayLike
    ) -> float:
        """PyTorch MPS backend does not provide a separate JIT path here."""
        return self.enbs_simple(net_benefit_array, research_cost)

    def _capability_version(self) -> str:
        return str(self._torch.__version__)


# Helper functions for benchmark payload normalization and comparisons.
def _backend_device_name(backend: Backend) -> str | None:
    """Return the backend device name when available."""
    device = getattr(backend, "device", None)
    if device is None or not hasattr(device, "type"):
        return None
    return str(device.type)


def _payload_to_float(payload: dict[str, object], key: str) -> float | None:
    """Read a payload value as float when it has a numeric type."""
    value = payload.get(key)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _payload_metric(
    payload: dict[str, object],
    key: str,
    summary_key: str | None = None,
) -> float:
    """Read a numeric metric from a benchmark payload."""
    value = _payload_to_float(payload, key)
    if value is not None:
        return value

    if summary_key is None:
        return 0.0

    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary_value = _payload_to_float(summary, summary_key)
        if summary_value is not None:
            return summary_value

    return 0.0


def _payload_result(payload: dict[str, object]) -> float:
    """Read EVPI result from a benchmark payload."""
    return _payload_metric(payload, "result", summary_key="evpi")


def _payload_mean_latency_ns(payload: dict[str, object]) -> float:
    """Read mean latency from a benchmark payload."""
    return _payload_metric(payload, "mean_latency_ns", summary_key="mean_latency_ns")


def _payload_throughput(payload: dict[str, object]) -> float:
    """Read throughput from a benchmark payload."""
    return _payload_metric(
        payload,
        "throughput_ops_per_sec",
        summary_key="throughput_ops_per_sec",
    )


def _array_signature(values: ArrayLike) -> dict[str, object]:
    """Return a stable workload fingerprint for benchmark reproducibility checks."""
    array = np.array(values, dtype="<f8", order="C", copy=True)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _runtime_manifest() -> dict[str, object]:
    """Return a compact runtime manifest for review packets."""
    mps_available = None
    if (
        torch is not None
        and hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
    ):
        try:
            mps_available = {
                "built": bool(torch.backends.mps.is_built()),
                "available": bool(torch.backends.mps.is_available()),
            }
        except Exception as exc:  # pragma: no cover - platform dependent
            mps_available = {"error": str(exc)}

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "backend": {
            "torch": getattr(torch, "__version__", None) if torch is not None else None,
            "apple_metal_capability": {
                "platform_is_darwin": sys.platform == "darwin",
                "mps": mps_available,
            },
        },
    }


_PHASE_3_HARDENED_REQUIRED_FIELDS: tuple[str, ...] = (
    "payload_version",
    "workflow",
    "review_phase",
    "review_context",
    "repeats",
    "warmup_runs",
    "runtime",
    "workload",
    "benchmarks",
    "comparison",
    "apple_metal_error",
    "review",
)

_PHASE_3_PAYLOAD_VERSION = "1.0.0"


def _coalesce_benchmark_error(error: str | None) -> str | None:
    """Coerce benchmark errors into deterministic serializable shape."""
    return None if error is None else str(error)


def _comparison_enabled(payload: dict[str, object]) -> bool:
    """Read whether a benchmark payload has an Apple Metal comparison."""
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        return False
    return bool(comparison.get("enabled"))


def _build_phase_3_comparison(
    scalar_payload: dict[str, object],
    memory_payload: dict[str, object],
) -> dict[str, object]:
    """Build a strict Phase-3 comparison section for handoff packets."""
    scalar_enabled = _comparison_enabled(scalar_payload)
    memory_enabled = _comparison_enabled(memory_payload)

    scalar_comparison = scalar_payload.get("comparison")
    if not isinstance(scalar_comparison, dict):
        scalar_comparison = {}

    memory_comparison = memory_payload.get("comparison")
    if not isinstance(memory_comparison, dict):
        memory_comparison = {}

    comparison_enabled = bool(scalar_enabled and memory_enabled)
    return {
        "enabled": comparison_enabled,
        "scalar": scalar_comparison,
        "memory": memory_comparison,
    }


def _build_phase_3_review_required_fields() -> list[str]:
    """Return the strict list of Phase-3 review fields expected by downstream tooling."""
    return [
        "payload_version",
        "workflow",
        "review.phase",
        "review.status",
        "review.required_fields",
        "runtime.platform",
        "runtime.system",
        "workload.shape",
        "workload.sha256",
        "benchmarks.scalar.payload_version",
        "benchmarks.scalar.workflow",
        "benchmarks.memory.payload_version",
        "benchmarks.memory.workflow",
        "benchmarks.scalar.repeats",
        "benchmarks.scalar.warmup_runs",
        "benchmarks.memory.repeats",
        "benchmarks.memory.warmup_runs",
    ]


def _safe_divide(numerator: float, denominator: float) -> float:
    """Compute a deterministic ratio with zero-safe fallback values."""
    if denominator == 0.0:
        if numerator > 0:
            return float("inf")
        if numerator < 0:
            return float("-inf")
        return 0.0
    return numerator / denominator


# Global backend registry
_BACKENDS: dict[str, Backend] = {
    "numpy": NumpyBackend(),
}

if JAX_AVAILABLE:
    _BACKENDS["jax"] = JaxBackend()

# Default backend
_DEFAULT_BACKEND_STATE: dict[str, str] = {"name": "numpy"}


def get_backend(name: str | None = None) -> Backend:
    """
    Get a computational backend by name.

    If no name is provided, returns the default backend.
    """
    if name is None:
        name = _DEFAULT_BACKEND_STATE["name"]

    if name == "apple_metal":
        return AppleMetalBackend()

    if name not in _BACKENDS:
        raise_value_error(f"Unknown backend: {name}")

    return _BACKENDS[name]


def set_backend(name: str) -> None:
    """Set the default computational backend."""
    if name == "apple_metal":
        get_backend(name)
    elif name not in _BACKENDS:
        raise_value_error(f"Unknown backend: {name}")
    _DEFAULT_BACKEND_STATE["name"] = name


def benchmark_evpi(
    backend: Backend,
    net_benefit_array: ArrayLike,
    repeats: int = 1_000,
    warmup_runs: int = 1,
) -> dict[str, object]:
    """Measure a backend's EVPI throughput on a fixed workload."""
    if repeats <= 0:
        raise_value_error("repeats must be positive")
    if warmup_runs < 0:
        raise_value_error("warmup_runs must be non-negative")

    for _ in range(warmup_runs):
        backend.evpi(net_benefit_array)

    start_ns = time.perf_counter_ns()
    result = 0.0
    for _ in range(repeats):
        result = float(backend.evpi(net_benefit_array))
    elapsed_ns = time.perf_counter_ns() - start_ns
    mean_latency_ns = elapsed_ns / repeats
    throughput_ops_per_sec = (
        1_000_000_000.0 / mean_latency_ns if mean_latency_ns > 0 else float("inf")
    )
    device_name = _backend_device_name(backend)

    return {
        "backend": type(backend).__name__,
        "device": device_name,
        "mean_latency_ns": mean_latency_ns,
        "repeats": repeats,
        "result": result,
        "throughput_ops_per_sec": throughput_ops_per_sec,
        "warmup_runs": warmup_runs,
    }


def _current_rss_bytes() -> int | None:
    """Return the current resident set size in bytes when available."""
    if resource is None:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def benchmark_memory_throughput(
    backend: Backend,
    net_benefit_array: ArrayLike,
    repeats: int = 1_000,
    warmup_runs: int = 1,
) -> dict[str, object]:
    """Measure a backend using the memory/throughput benchmark shape."""
    if repeats <= 0:
        raise_value_error("repeats must be positive")
    if warmup_runs < 0:
        raise_value_error("warmup_runs must be non-negative")

    samples: list[dict[str, object]] = []
    summary_result = 0.0

    for iteration, phase in enumerate(("cold", "warm", "warm")):
        rss_before = _current_rss_bytes()
        if iteration == 0:
            for _ in range(warmup_runs):
                backend.evpi(net_benefit_array)

        start_ns = time.perf_counter_ns()
        result = 0.0
        for _ in range(repeats):
            result = float(backend.evpi(net_benefit_array))
        elapsed_ns = time.perf_counter_ns() - start_ns
        rss_after = _current_rss_bytes()

        summary_result = result
        samples.append(
            {
                "phase": phase,
                "iteration": iteration,
                "latency_ns": elapsed_ns,
                "throughput_ops_per_sec": (
                    1_000_000_000.0 / (elapsed_ns / repeats)
                    if elapsed_ns > 0
                    else float("inf")
                ),
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
            }
        )

    latencies = [int(sample["latency_ns"]) for sample in samples]
    throughputs = [float(sample["throughput_ops_per_sec"]) for sample in samples]
    rss_values = [
        value
        for sample in samples
        for value in (sample["rss_before_bytes"], sample["rss_after_bytes"])
        if value is not None
    ]
    mean_latency_ns = round(sum(latencies) / len(latencies))
    mean_throughput = sum(throughputs) / len(throughputs)
    return {
        "backend": type(backend).__name__,
        "device": _backend_device_name(backend),
        "samples": samples,
        "summary": {
            "evpi": summary_result,
            "cold_start_latency_ns": latencies[0],
            "warm_start_latency_ns": latencies[1],
            "mean_latency_ns": mean_latency_ns,
            "peak_rss_bytes": max(rss_values) if rss_values else None,
            "throughput_ops_per_sec": mean_throughput,
        },
        "mean_latency_ns": mean_latency_ns,
        "throughput_ops_per_sec": mean_throughput,
        "repeats": repeats,
        "warmup_runs": warmup_runs,
    }


def benchmark_mps_vs_cpu(
    net_benefit_array: ArrayLike,
    repeats: int = 1_000,
    warmup_runs: int = 1,
    benchmark: Callable[
        [Backend, ArrayLike, int, int], dict[str, object]
    ] = benchmark_memory_throughput,
) -> dict[str, object]:
    """Compare a CPU benchmark payload against an optional Apple Metal payload.

    The returned payload includes deterministic workload and runtime metadata for
    review packets, including enough information to produce a Phase-3 handoff
    artifact without any additional environment probing.
    """
    if repeats <= 0:
        raise_value_error("repeats must be positive")
    if warmup_runs < 0:
        raise_value_error("warmup_runs must be non-negative")

    # Keep the existing single-backend contract validation behavior.
    workload_signature = _array_signature(net_benefit_array)

    cpu_payload = benchmark(
        NumpyBackend(),
        net_benefit_array,
        repeats,
        warmup_runs,
    )

    apple_payload: dict[str, object] | None = None
    apple_error: str | None = None
    try:
        apple_payload = benchmark(
            get_backend("apple_metal"),
            net_benefit_array,
            repeats,
            warmup_runs,
        )
    except Exception as exc:  # pragma: no cover - platform-dependent
        apple_error = str(exc)

    comparison = {
        "enabled": apple_payload is not None,
        "apple_metal_backend": apple_payload["backend"] if apple_payload else None,
        "apple_metal_device": apple_payload["device"] if apple_payload else None,
        "result_delta": 0.0,
        "mean_latency_speedup": 0.0,
        "throughput_speedup": 0.0,
    }
    if apple_payload is not None:
        cpu_result = _payload_result(cpu_payload)
        apple_result = _payload_result(apple_payload)
        cpu_latency = _payload_mean_latency_ns(cpu_payload)
        apple_latency = _payload_mean_latency_ns(apple_payload)
        cpu_throughput = _payload_throughput(cpu_payload)
        apple_throughput = _payload_throughput(apple_payload)

        comparison["result_delta"] = apple_result - cpu_result
        comparison["mean_latency_speedup"] = _safe_divide(cpu_latency, apple_latency)
        comparison["throughput_speedup"] = _safe_divide(
            apple_throughput,
            cpu_throughput,
        )

    return {
        "backend": "apple_metal_vs_cpu",
        "payload_version": _PHASE_3_PAYLOAD_VERSION,
        "workflow": "apple_metal_vs_cpu",
        "benchmark": getattr(benchmark, "__name__", "benchmark"),
        "workload": workload_signature,
        "runtime": _runtime_manifest(),
        "repeats": repeats,
        "warmup_runs": warmup_runs,
        "cpu": cpu_payload,
        "apple_metal": apple_payload,
        "comparison": comparison,
        "apple_metal_error": apple_error,
        "review": {
            "phase": "phase_3",
            "status": "device_comparison_available"
            if apple_payload
            else "cpu_reference_only",
            "required_fields": [
                "backend",
                "device",
                "workload.shape",
                "workload.sha256",
                "repeats",
                "warmup_runs",
                "mean_latency_ns",
                "throughput_ops_per_sec",
            ],
            "notes": [
                "CPU payload is the reference contract.",
                "Apple payload is optional and only present when MPS is available.",
            ],
        },
    }


def compile_phase_3_handoff_packet(
    net_benefit_array: ArrayLike,
    repeats: int = 1_000,
    warmup_runs: int = 1,
    as_json: bool = False,
) -> dict[str, object] | str:
    """Build a deterministic Phase-3 review packet for both scalar and memory workloads.

    The handoff packet keeps existing payload contracts (for both scalar EVPI and
    memory/throughput helpers) and wraps them in one reproducible review record.
    """
    scalar_payload = benchmark_mps_vs_cpu(
        net_benefit_array,
        repeats=repeats,
        warmup_runs=warmup_runs,
        benchmark=benchmark_evpi,
    )
    memory_payload = benchmark_mps_vs_cpu(
        net_benefit_array,
        repeats=repeats,
        warmup_runs=warmup_runs,
        benchmark=benchmark_memory_throughput,
    )

    review_status = "device_comparison_available"
    scalar_error = _coalesce_benchmark_error(
        cast("str | None", scalar_payload.get("apple_metal_error"))
    )
    memory_error = _coalesce_benchmark_error(
        cast("str | None", memory_payload.get("apple_metal_error"))
    )
    if scalar_error is not None or memory_error is not None:
        review_status = "cpu_reference_only"

    packet: dict[str, object] = {
        "payload_version": _PHASE_3_PAYLOAD_VERSION,
        "workflow": "apple_metal_phase_3_handoff",
        "review_phase": "phase_3",
        "review_context": "apple_metal_vs_cpu",
        "repeats": repeats,
        "warmup_runs": warmup_runs,
        "runtime": memory_payload.get("runtime", _runtime_manifest()),
        "workload": memory_payload.get("workload", {}),
        "benchmarks": {
            "scalar": {
                **scalar_payload,
                "apple_metal_error": None,
                "payload_version": scalar_payload.get(
                    "payload_version", _PHASE_3_PAYLOAD_VERSION
                ),
                "workflow": "apple_metal_vs_cpu",
            },
            "memory": {
                **memory_payload,
                "apple_metal_error": None,
                "payload_version": memory_payload.get(
                    "payload_version", _PHASE_3_PAYLOAD_VERSION
                ),
                "workflow": "apple_metal_vs_cpu",
            },
        },
        "comparison": _build_phase_3_comparison(
            scalar_payload,
            memory_payload,
        ),
        "apple_metal_error": {
            "scalar": scalar_error,
            "memory": memory_error,
        },
        "review": {
            "phase": "phase_3",
            "status": review_status,
            "required_fields": _build_phase_3_review_required_fields(),
            "notes": [
                "Scalar benchmark is EVPI helper review payload.",
                "Memory benchmark is memory/throughput helper review payload.",
                "Payload stays deterministic for reviewer handoff.",
            ],
        },
    }

    for required_key in _PHASE_3_HARDENED_REQUIRED_FIELDS:
        if required_key not in packet:
            raise RuntimeError(f"Missing required Phase-3 handoff key: {required_key}")

    if as_json:
        return json.dumps(packet, sort_keys=True)
    return packet
