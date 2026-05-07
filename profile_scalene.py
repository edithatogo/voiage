"""
Performance profiling script for voiage using Scalene.

Usage:
    scalene voiage/profile_scalene.py
    scalene --cli voiage/profile_scalene.py
    scalene --html --outfile profile.html voiage/profile_scalene.py
"""

import numpy as np

from voiage.analysis import evpi, evppi


def profile_evpi():
    """Profile EVPI calculation with realistic data size."""
    np.random.seed(42)
    n_simulations = 10_000
    n_strategies = 5

    print(f"Profiling EVPI with {n_simulations} simulations, {n_strategies} strategies...")
    psa_outputs = np.random.rand(n_simulations, n_strategies) * 1000

    evpi_value = evpi({}, psa_outputs)
    print(f"EVPI: {evpi_value:.4f}")


def profile_evppi():
    """Profile EVPPI calculation with realistic data size."""
    np.random.seed(42)
    n_simulations = 10_000
    n_parameters = 10

    print(f"Profiling EVPPI with {n_simulations} simulations, {n_parameters} parameters...")
    psa_inputs = {f'param_{i}': np.random.rand(n_simulations) for i in range(n_parameters)}
    psa_outputs = np.random.rand(n_simulations, 3) * 1000

    evppi_value = evppi(psa_inputs, psa_outputs)
    print(f"EVPPI: {evppi_value:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("voiage Performance Profing")
    print("=" * 60)

    profile_evpi()
    print("-" * 60)

    profile_evppi()
    print("=" * 60)
    print("Profiling complete!")
