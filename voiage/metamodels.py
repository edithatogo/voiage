# voiage/metamodels.py

"""Metamodels for Value of Information analysis."""

from typing import Protocol, TypedDict

import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tinygp

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


class MLP(nn.Module):
    """A simple MLP model."""

    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


class FlaxMetamodel:
    """A metamodel that uses a Flax MLP to predict the target values."""

    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.state = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        x_np = np.array(list(x.parameters.values())).T
        y_np = y.reshape(-1, 1)

        model = MLP(features=x_np.shape[1])
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


class _GPParams(TypedDict):
    kernel: tinygp.kernels.Kernel
    log_diag: jax.Array


class TinyGPMetamodel:
    """A metamodel that uses a tinygp GP to predict the target values."""

    def __init__(self):
        self.gp = None
        self.x_train = None

    def fit(self, x: ParameterSet, y: np.ndarray) -> None:
        """Fit the metamodel to the data."""
        self.x_train = np.array(list(x.parameters.values())).T
        self.y_train = y

        kernel = 1.0 * tinygp.kernels.Matern32(1.0)

        @jax.jit
        def loss(params: _GPParams):
            gp = tinygp.GaussianProcess(
                params["kernel"], self.x_train, diag=jnp.exp(params["log_diag"])
            )
            return -gp.log_probability(jnp.array(self.y_train))

        opt = optax.adam(0.01)
        params: _GPParams = {
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
            raise RuntimeError("The model has not been fitted yet.")

        x_np = np.array(list(x.parameters.values())).T
        _, cond = self.gp.condition(self.y_train, x_np)
        return cond.loc
