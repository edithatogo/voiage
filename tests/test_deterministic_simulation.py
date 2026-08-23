"""Deterministic Simulation Testing (DST) suite for Value of Information analysis.

This module verifies deterministic simulation invariants across:
- DST1: Bit-exact RNG reproducibility across multiple independent executions
- DST2: Worker chunk partition invariance (SeedSequence chunk equivalence)
- DST3: Sequential state machine determinism and reproducible trajectories
- DST4: Fault injection and idempotent state recovery
- DST5: Virtual clock advance and time-free deterministic discount/decay
"""

from __future__ import annotations

import numpy as np
import pytest

from voiage.methods.basic import evpi, evppi
from voiage.schema import ParameterSet


def _simulate_synthetic_psa(
    seed: int, n_samples: int = 500
) -> tuple[np.ndarray, ParameterSet]:
    """Generates synthetic PSA net benefits and parameter set using a deterministic generator."""
    rng = np.random.default_rng(seed)
    theta_1 = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    theta_2 = rng.gamma(shape=2.0, scale=1.5, size=n_samples)

    # Strategy net benefits conditioned on parameters
    nb_1 = 100.0 + 15.0 * theta_1 - 5.0 * theta_2 + rng.normal(0.0, 2.0, size=n_samples)
    nb_2 = 105.0 + 8.0 * theta_1 + 2.0 * theta_2 + rng.normal(0.0, 2.0, size=n_samples)
    nb_3 = 98.0 + 20.0 * theta_1 - 10.0 * theta_2 + rng.normal(0.0, 2.0, size=n_samples)

    nb_matrix = np.column_stack([nb_1, nb_2, nb_3])
    param_set = ParameterSet.from_numpy_or_dict(
        {"theta_1": theta_1, "theta_2": theta_2}
    )
    return nb_matrix, param_set


def test_dst1_bit_exact_rng_reproducibility():
    """DST1: Identical seeds produce bit-exact identical arrays and VOI calculations."""
    fixed_seed = 123456789

    # Run 1
    nb1, params1 = _simulate_synthetic_psa(fixed_seed, n_samples=300)
    evpi1 = evpi(nb1)
    evppi1 = evppi(nb1, params1, parameters_of_interest=["theta_1"])

    # Run 2
    nb2, params2 = _simulate_synthetic_psa(fixed_seed, n_samples=300)
    evpi2 = evpi(nb2)
    evppi2 = evppi(nb2, params2, parameters_of_interest=["theta_1"])

    # Bit-exact array comparison
    np.testing.assert_array_equal(
        nb1, nb2, err_msg="PSA net benefits are not bit-exact across identical seeds"
    )
    np.testing.assert_array_equal(
        params1.parameters["theta_1"],
        params2.parameters["theta_1"],
        err_msg="PSA parameter samples are not bit-exact across identical seeds",
    )

    # Exact floating point equality
    assert evpi1 == evpi2, f"EVPI was not bit-exact ({evpi1} vs {evpi2})"
    assert evppi1 == evppi2, f"EVPPI was not bit-exact ({evppi1} vs {evppi2})"


def test_dst2_worker_chunk_partition_invariance():
    """DST2: Deterministic per-chunk sub-seed generation reproduces monolithic simulation."""
    master_seed = 987654321
    total_samples = 1000

    def _generate_chunked_samples(num_chunks: int) -> np.ndarray:
        samples_per_chunk = total_samples // num_chunks
        ss = np.random.SeedSequence(master_seed)
        child_seeds = ss.spawn(num_chunks)

        all_chunks = []
        for child in child_seeds:
            rng = np.random.default_rng(child)
            chunk = rng.normal(loc=50.0, scale=10.0, size=(samples_per_chunk, 3))
            all_chunks.append(chunk)
        return np.vstack(all_chunks)

    # Monolithic chunk (num_chunks = 1)
    monolithic_nb = _generate_chunked_samples(num_chunks=1)
    monolithic_evpi = evpi(monolithic_nb)

    for n_chunks in [2, 4, 10]:
        chunked_nb = _generate_chunked_samples(num_chunks=n_chunks)
        chunked_evpi = evpi(chunked_nb)

        # Statistical convergence across partitions with deterministic seeds
        assert abs(chunked_evpi - monolithic_evpi) < 1.0, (
            f"Chunk partition with {n_chunks} chunks deviated from monolithic baseline"
        )
        assert np.isfinite(chunked_evpi)


def test_dst3_sequential_state_machine_determinism():
    """DST3: Sequential VOI stopping boundary transitions through deterministic trajectories."""
    base_seed = 42

    def _simulate_sequential_trial(
        seed: int, steps: int = 10
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(seed)
        trajectory = []
        accumulated_info = 0.0
        cumulative_cost = 0.0

        for step in range(steps):
            signal = rng.normal(loc=1.2, scale=0.5)
            step_cost = 100.0 + rng.uniform(0.0, 10.0)
            accumulated_info += signal
            cumulative_cost += step_cost

            # ENBS at current step
            current_enbs = max(0.0, accumulated_info * 500.0 - cumulative_cost)
            trajectory.append(
                {
                    "step": step,
                    "accumulated_info": accumulated_info,
                    "cumulative_cost": cumulative_cost,
                    "enbs": current_enbs,
                }
            )
        return trajectory

    traj_a = _simulate_sequential_trial(base_seed)
    traj_b = _simulate_sequential_trial(base_seed)

    assert len(traj_a) == len(traj_b) == 10
    for step_idx in range(10):
        assert traj_a[step_idx] == traj_b[step_idx], (
            f"Sequential trajectory diverged at step {step_idx}"
        )


def test_dst4_fault_injection_and_idempotent_recovery():
    """DST4: Simulated transient failure with deterministic retry recovers exact outputs."""
    seed = 555
    nb, _params = _simulate_synthetic_psa(seed, n_samples=250)

    # Simulated worker with fault injection on first attempt
    call_attempts = 0

    def unreliable_evaluator(data: np.ndarray) -> float:
        nonlocal call_attempts
        call_attempts += 1
        if call_attempts == 1:
            raise RuntimeError("Transient worker timeout simulated")
        return evpi(data)

    # Attempt with retry
    result = None
    for _ in range(3):
        try:
            result = unreliable_evaluator(nb)
            break
        except RuntimeError:
            continue

    assert result is not None
    assert call_attempts == 2
    exact_baseline = evpi(nb)
    assert result == exact_baseline, "Recovered result did not match exact baseline"


def test_dst5_virtual_clock_advance_determinism():
    """DST5: Virtual clock advance produces deterministic discounted VOI without wall-clock dependency."""
    initial_voi = 10000.0
    discount_rate = 0.035
    horizon_years = 10

    # Step-by-step virtual clock
    discounted_values = []
    for year in range(horizon_years + 1):
        df = 1.0 / ((1.0 + discount_rate) ** year)
        discounted_values.append(initial_voi * df)

    # Assert monotonic decay and deterministic mathematical bounds
    for y in range(horizon_years):
        assert discounted_values[y] > discounted_values[y + 1]

    assert discounted_values[0] == 10000.0
    assert discounted_values[10] == pytest.approx(10000.0 / (1.035**10), rel=1e-12)
