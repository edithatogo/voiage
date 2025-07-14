# voiage/backends.py

"""
Backend management for voiage.

This module will allow users to switch between different computational
backends, such as NumPy and JAX, for performance-critical calculations.
"""

from typing import Literal

Backend = Literal["numpy", "jax"]

_CURRENT_BACKEND: Backend = "numpy"


def set_backend(backend: Backend):
    """Set the computational backend for voiage.

    Args:
        backend (Backend): The backend to use, either "numpy" or "jax".
    """
    global _CURRENT_BACKEND
    if backend not in ["numpy", "jax"]:
        raise ValueError("Backend must be 'numpy' or 'jax'")
    _CURRENT_BACKEND = backend


def get_backend() -> Backend:
    """Get the current computational backend.

    Returns:
        Backend: The current backend, either "numpy" or "jax".
    """
    return _CURRENT_BACKEND
