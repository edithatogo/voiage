"""
Computational backends for voiage.

This module provides a dispatch system for selecting computational backends
and includes implementations for different backends, starting with NumPy and JAX.
"""

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Abstract base class for computational backends."""

    @abstractmethod
    def evpi(self, net_benefit_array):
        """Calculate the Expected Value of Perfect Information (EVPI)."""
        pass


class NumpyBackend(Backend):
    """NumPy-based computational backend."""

    def evpi(self, net_benefit_array):
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
        evpi = expected_max_nb - max_expected_nb

        return evpi


# Global backend registry
_BACKENDS = {
    "numpy": NumpyBackend(),
}

# Default backend
_DEFAULT_BACKEND = "numpy"


def get_backend(name=None):
    """
    Get a computational backend by name.

    If no name is provided, returns the default backend.
    """
    if name is None:
        name = _DEFAULT_BACKEND

    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}")

    return _BACKENDS[name]


def set_backend(name):
    """Set the default computational backend."""
    global _DEFAULT_BACKEND
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}")
    _DEFAULT_BACKEND = name
