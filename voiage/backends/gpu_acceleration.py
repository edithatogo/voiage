"""GPU acceleration utilities for the JAX backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np


class GpuAcceleration:
    """Utilities for GPU acceleration and memory management."""

    @staticmethod
    def detect_gpu() -> list[jax.Device]:
        """Detect available GPU devices."""
        devices = jax.devices()
        return [device for device in devices if "gpu" in device.device_kind.lower()]

    @staticmethod
    def get_memory_info() -> dict[str, object]:
        """Get GPU memory information."""
        try:
            # This is a placeholder - actual implementation would depend on JAX version
            return {
                "gpu_available": len(GpuAcceleration.detect_gpu()) > 0,
                "gpu_count": len(GpuAcceleration.detect_gpu()),
                "memory_info": "Available via jax.lib.xla_bridge",
            }
        except Exception:
            return {
                "gpu_available": False,
                "gpu_count": 0,
                "memory_info": "Unable to query memory info",
            }

    @staticmethod
    def optimize_for_gpu(data: object) -> object:
        """Optimize data layout for GPU processing."""
        # Ensure data is in optimal format for GPU
        if hasattr(data, "device_buffer"):
            # JAX array - already optimized
            return data
        # Convert to JAX array with optimal dtype
        return jnp.asarray(data, dtype=jnp.float32)  # float32 often faster on GPU

    @staticmethod
    def memory_efficient_batch_process(
        data_batches: Sequence[np.ndarray],
        process_func: Callable[[list[np.ndarray]], object],
        max_memory_mb: int = 1000,
    ) -> object:
        """Process large datasets in memory-efficient batches."""
        results: list[np.ndarray] = []
        current_memory = 0

        for i, batch in enumerate(data_batches):
            batch_size_mb = batch.nbytes / (1024 * 1024)

            if current_memory + batch_size_mb > max_memory_mb:
                # Process current results to free memory
                print(f"Processing batch {i} to free memory")
                results = process_func(results)
                current_memory = 0

            # Add batch to current memory usage
            current_memory += batch_size_mb
            results.append(batch)

        # Process final results
        return process_func(results)
