# tests/test_advanced_voi_performance.py

"""Performance benchmarks for advanced VOI methods."""

import time

import numpy as np
import pytest

from voiage.methods.network_meta_analysis import (
    NetworkMetaAnalysisData,
    calculate_nma_evpi,
)
from voiage.methods.structural import (
    JAX_AVAILABLE,
    structural_evpi,
    structural_evpi_jit,
)
from voiage.schema import ParameterSet, ValueArray


@pytest.mark.benchmark
class TestStructuralVOIPerformance:
    """Performance benchmarks for structural VOI methods."""

    def _create_structures(self, n_samples, n_structures, n_strategies):
        """Create test structures for benchmarking."""
        np.random.seed(42)

        def make_evaluator(idx):
            def evaluator(psa):
                values = np.random.normal(100 + idx * 5, 10, (n_samples, n_strategies))
                return ValueArray.from_numpy(
                    values, [f"S{i}" for i in range(n_strategies)]
                )

            return evaluator

        evaluators = [make_evaluator(i) for i in range(n_structures)]
        probabilities = np.ones(n_structures) / n_structures
        psa_samples = [
            ParameterSet.from_numpy_or_dict({"p": np.random.rand(n_samples)})
            for _ in range(n_structures)
        ]

        return evaluators, probabilities, psa_samples

    def test_structural_evpi_small(self, benchmark) -> None:
        """Benchmark structural EVPI with small dataset."""
        evaluators, probabilities, psa_samples = self._create_structures(1000, 3, 2)

        result = benchmark(structural_evpi, evaluators, probabilities, psa_samples)
        assert result >= 0

    def test_structural_evpi_large(self, benchmark) -> None:
        """Benchmark structural EVPI with large dataset."""
        evaluators, probabilities, psa_samples = self._create_structures(100000, 5, 4)

        result = benchmark(structural_evpi, evaluators, probabilities, psa_samples)
        assert result >= 0

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_structural_evpi_jit_vs_numpy(self) -> None:
        """Compare JAX JIT vs NumPy performance."""
        n_samples = 10000
        n_structures = 5
        n_strategies = 3

        np.random.seed(42)
        evaluators, probabilities, psa_samples = self._create_structures(
            n_samples, n_structures, n_strategies
        )

        # NumPy version
        start = time.perf_counter()
        result_numpy = structural_evpi(evaluators, probabilities, psa_samples)
        numpy_time = time.perf_counter() - start

        # JAX JIT version (first call includes compilation)
        nb_arrays = [
            np.random.normal(100 + i * 5, 10, (n_samples, n_strategies))
            for i in range(n_structures)
        ]

        # First call (compilation)
        _ = structural_evpi_jit(nb_arrays, probabilities)

        # Second call (actual performance)
        start = time.perf_counter()
        result_jit = structural_evpi_jit(nb_arrays, probabilities)
        jit_time = time.perf_counter() - start

        # Results should be similar (within 10%)
        tolerance = 0.1
        assert abs(result_numpy - result_jit) / result_numpy < tolerance

        print(f"\nNumPy time: {numpy_time:.4f}s, JAX JIT time: {jit_time:.4f}s")


@pytest.mark.benchmark
class TestNMAVOIPerformance:
    """Performance benchmarks for NMA VOI methods."""

    def _create_nma_data(self, n_samples, n_treatments):
        """Create NMA data for benchmarking."""
        np.random.seed(42)
        treatments = [f"T{i}" for i in range(n_treatments)]

        treatment_effects = {}
        for i in range(n_treatments):
            for j in range(i + 1, n_treatments):
                treatment_effects[(treatments[i], treatments[j])] = np.random.normal(
                    0.5, 0.2, n_samples
                )

        return NetworkMetaAnalysisData(
            treatment_effects=treatment_effects,
            n_studies=20,
            treatments=treatments,
            outcome_type="continuous",
        )

    def test_nma_evpi_small(self, benchmark) -> None:
        """Benchmark NMA-EVPI with small network."""
        nma_data = self._create_nma_data(1000, 4)
        result = benchmark(calculate_nma_evpi, nma_data, n_samples=1000)
        assert result >= 0

    def test_nma_evpi_large(self, benchmark) -> None:
        """Benchmark NMA-EVPI with large network."""
        nma_data = self._create_nma_data(10000, 8)
        result = benchmark(calculate_nma_evpi, nma_data, n_samples=10000)
        assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__])
