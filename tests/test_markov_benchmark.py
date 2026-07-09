import numpy as np
import pytest

from voiage.healthcare.utilities import markov_cohort_model


@pytest.mark.benchmark
def test_benchmark_markov_cohort_model(benchmark):
    """Benchmark the markov_cohort_model function."""
    # Setup a reasonably sized problem that shows performance differences
    n_states = 100
    n_cycles = 1000

    # Create a valid transition matrix
    np.random.seed(42)
    transition_matrix = np.random.rand(n_states, n_states)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    # Initial state (starts in state 0)
    initial_state = np.zeros(n_states)
    initial_state[0] = 1.0

    # Run the benchmark
    benchmark(markov_cohort_model, transition_matrix, initial_state, n_cycles)
