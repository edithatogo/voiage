import numpy as np

from voiage.methods.mcda import value_of_mcda_information


def test_mcda_voi_values_criterion_weight_information() -> None:
    result = value_of_mcda_information(
        np.array([[10.0, 0.0], [0.0, 10.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert result.baseline_value == 5.0
    assert result.flexible_value == 10.0
    assert result.value == 5.0
