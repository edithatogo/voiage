import numpy as np

from voiage.methods.distributional import value_of_distributional_information


def test_vdi_values_distribution_family_resolution() -> None:
    result = value_of_distributional_information({
        "family_a": np.array([[10.0, 0.0], [10.0, 0.0]]),
        "family_b": np.array([[0.0, 10.0], [0.0, 10.0]]),
    })
    assert result.value == 5.0
    assert result.family_optimal_strategy_indices == {"family_a": 0, "family_b": 1}


def test_vdi_rejects_empty_families() -> None:
    import pytest
    with pytest.raises(ValueError):
        value_of_distributional_information({})
