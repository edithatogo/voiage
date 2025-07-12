# tests/conftest.py

"""
Configuration for pytest.

This file can be used to define fixtures, hooks, and plugins that are
shared across multiple test files.
"""

import numpy as np
import pytest

from voiage.config import DEFAULT_DTYPE
from voiage.core.data_structures import (
    NetBenefitArray,
    PSASample,
    TrialArm,
    TrialDesign,
)

# --- Fixtures for basic data structures ---


@pytest.fixture(scope="session")  # Reused across all tests in a session
def sample_nb_data_array_2strat() -> np.ndarray:
    """Provide a sample NumPy array of net benefits (5 samples, 2 strategies)."""
    return np.array(
        [
            [100, 105],
            [110, 100],
            [90, 110],
            [120, 100],
            [95, 115],
        ],
        dtype=DEFAULT_DTYPE,
    )


@pytest.fixture(scope="session")
def sample_net_benefit_array_2strat(
    sample_nb_data_array_2strat: np.ndarray,
) -> NetBenefitArray:
    """Provide a NetBenefitArray instance (5 samples, 2 strategies)."""
    return NetBenefitArray(
        values=sample_nb_data_array_2strat,
        strategy_names=["Strategy A", "Strategy B"],
    )


@pytest.fixture(scope="session")
def sample_psa_data_dict_2params() -> dict:
    """Provide a sample dictionary of PSA parameters (3 samples, 2 parameters)."""
    return {
        "param1": np.array([1.0, 1.1, 1.2], dtype=DEFAULT_DTYPE),
        "param2": np.array([0.5, 0.4, 0.6], dtype=DEFAULT_DTYPE),
    }


@pytest.fixture(scope="session")
def sample_psa_2params(sample_psa_data_dict_2params: dict) -> PSASample:
    """Provide a PSASample instance (3 samples, 2 parameters)."""
    return PSASample(parameters=sample_psa_data_dict_2params)


@pytest.fixture(scope="session")
def sample_trial_design_2arms() -> TrialDesign:
    """Provide a sample TrialDesign instance with two arms."""
    arm1 = TrialArm(name="Treatment X", sample_size=100)
    arm2 = TrialArm(name="Control", sample_size=100)
    return TrialDesign(arms=[arm1, arm2])


# --- Fixtures for more complex scenarios (e.g., EVPPI) ---


@pytest.fixture(scope="session")
def evppi_test_data_simple():
    """Generate a simple dataset for testing EVPPI.

    Returns
    -------
    dict
        A dictionary containing 'nb_values', 'p_samples', and 'expected_evpi_approx'.
    """
    # - nb_values: Net benefits for 2 strategies.
    # - p_samples: Samples for one parameter of interest.
    # - expected_evpi: Pre-calculated EVPI for this data as a reference.
    np.random.seed(42)  # For reproducibility of this fixture
    n_samples = 500

    # Parameter of interest
    p_voi = np.random.normal(loc=20, scale=5, size=n_samples).astype(DEFAULT_DTYPE)

    # Net benefits dependent on p_voi and other noise
    nb_s1 = p_voi * 2.5 - 10 + np.random.normal(0, 10, n_samples)  # Strategy 1
    nb_s2 = 30 - 0.5 * p_voi + np.random.normal(0, 10, n_samples)  # Strategy 2

    nb_values = np.stack([nb_s1, nb_s2], axis=1)

    # Calculate reference EVPI (conceptually)
    # E[max(NB1, NB2)]
    expected_max_nb = np.mean(np.maximum(nb_s1, nb_s2))
    # max(E[NB1], E[NB2])
    max_expected_nb = np.maximum(np.mean(nb_s1), np.mean(nb_s2))
    expected_evpi_val = expected_max_nb - max_expected_nb

    return {
        "nb_values": nb_values,
        "p_samples": p_voi,
        "expected_evpi_approx": max(
            0.0, expected_evpi_val
        ),  # Ensure non-negative due to MC
    }


# --- Hooks (if needed) ---
# Example: Modify test collection or reporting
# def pytest_collection_modifyitems(config, items):
#     # ...
#     pass

# Example: Add command line options
# def pytest_addoption(parser):
#     parser.addoption(
#         "--runslow", action="store_true", default=False, help="run slow tests"
#     )

# Example: Skip tests based on command line options
# def pytest_configure(config):
#     config.addinivalue_line("markers", "slow: mark test as slow to run")

# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--runslow"):
#         # --runslow given in cli: do not skip slow tests
#         return
#     skip_slow = pytest.mark.skip(reason="need --runslow option to run")
#     for item in items:
#         if "slow" in item.keywords:
#             item.add_marker(skip_slow)

# This file will be automatically discovered by pytest.
# Fixtures defined here can be used as arguments in any test function
# within the `tests` directory and its subdirectories.


def pytest_configure(config):
    try:

        config.addinivalue_line(
            "markers", "sklearn: mark test as requiring scikit-learn"
        )
    except ImportError:
        config.addinivalue_line(
            "markers", "sklearn: mark test as requiring scikit-learn"
        )
        for item in config.items:
            if "sklearn" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="scikit-learn not available"))


print("conftest.py loaded by pytest.")
