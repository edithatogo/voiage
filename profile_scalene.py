"""
Performance profiling script for voiage using Scalene.

Usage:
    scalene profile_scalene.py
    scalene --cli profile_scalene.py
    scalene --html --outfile profile.html profile_scalene.py
"""

import numpy as np

from voiage import ParameterSet, ValueArray, evpi, evppi


def profile_evpi():
    """Profile EVPI calculation with realistic data size."""
    np.random.seed(42)
    n_simulations = 10_000
    n_strategies = 5

    print(
        f"Profiling EVPI with {n_simulations} simulations, {n_strategies} strategies..."
    )
    values = ValueArray.from_numpy(
        np.random.rand(n_simulations, n_strategies) * 1000,
        [f"Strategy {i}" for i in range(n_strategies)],
    )

    evpi_value = evpi(values)
    print(f"EVPI: {evpi_value:.4f}")


def profile_evppi():
    """Profile EVPPI calculation with realistic data size."""
    np.random.seed(42)
    n_simulations = 10_000
    n_parameters = 10

    print(
        f"Profiling EVPPI with {n_simulations} simulations, {n_parameters} parameters..."
    )
    parameters = ParameterSet.from_numpy_or_dict(
        {f"param_{i}": np.random.rand(n_simulations) for i in range(n_parameters)}
    )
    values = ValueArray.from_numpy(
        np.random.rand(n_simulations, 3) * 1000,
        ["Standard care", "New treatment", "Comparator"],
    )

    evppi_value = evppi(values, parameters, ["param_0"])
    print(f"EVPPI: {evppi_value:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("voiage Performance Profiling")
    print("=" * 60)

    profile_evpi()
    print("-" * 60)

    profile_evppi()
    print("=" * 60)
    print("Profiling complete!")
