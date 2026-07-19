"""Compatibility adapters for established VOIAGE runtime backends."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 - runtime cast target
from dataclasses import dataclass
import platform
import sys
from typing import Protocol, cast

import numpy as np

from voiage.contracts.capabilities import BackendCapabilities, Capability
from voiage.main_backends import AppleMetalBackend, Backend, JaxBackend, NumpyBackend


class _JaxDevice(Protocol):
    platform: str


@dataclass(frozen=True)
class LegacyBackendAdapter:
    """Expose an existing backend through the capability-aware pilot contract."""

    backend: Backend
    capabilities: BackendCapabilities

    def calculate_evpi(self, net_benefits: np.ndarray) -> float:
        """Delegate without changing established backend return behavior."""
        return float(self.backend.evpi(net_benefits))


def adapt_backend(backend: Backend) -> LegacyBackendAdapter:
    """Describe a built-in backend without adding abstract legacy methods."""
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    python_free_threaded = gil_probe is not None and not gil_probe()
    free_threaded: frozenset[Capability] = (
        frozenset({Capability.FREE_THREADED}) if python_free_threaded else frozenset()
    )
    if isinstance(backend, NumpyBackend):
        capabilities = BackendCapabilities(
            backend_name="numpy",
            backend_version=np.__version__,
            method_families=frozenset({"evpi", "enbs"}),
            dtypes=frozenset({"float32", "float64"}),
            devices=frozenset({"cpu"}),
            features=frozenset({Capability.DENSE_ARRAY, Capability.DETERMINISTIC})
            | free_threaded,
        )
    elif isinstance(backend, JaxBackend):
        import jax

        list_devices = cast("Callable[[], list[_JaxDevice]]", jax.devices)
        devices = list_devices()

        capabilities = BackendCapabilities(
            backend_name="jax",
            backend_version=jax.__version__,
            method_families=frozenset({"evpi", "enbs", "evppi", "evsi"}),
            dtypes=frozenset({"float32", "float64"}),
            devices=frozenset(item.platform for item in devices),
            features=frozenset(
                {
                    Capability.DENSE_ARRAY,
                    Capability.DETERMINISTIC,
                    Capability.JIT,
                    Capability.AUTODIFF,
                    Capability.BATCHING,
                }
            )
            | free_threaded,
        )
    elif isinstance(backend, AppleMetalBackend):
        capabilities = BackendCapabilities(
            backend_name="apple_metal",
            backend_version=platform.mac_ver()[0] or "unknown",
            method_families=frozenset({"evpi", "enbs"}),
            dtypes=frozenset({"float32"}),
            devices=frozenset({"mps"}),
            features=frozenset({Capability.DENSE_ARRAY, Capability.DETERMINISTIC}),
        )
    else:
        raise TypeError(f"Unsupported legacy backend type: {type(backend).__name__}")
    return LegacyBackendAdapter(backend=backend, capabilities=capabilities)
