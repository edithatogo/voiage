import numpy as np

from voiage.methods.implementation import expected_value_of_perfect_implementation


def test_expected_value_of_perfect_implementation_decomposes_loss() -> None:
    result = expected_value_of_perfect_implementation(np.array([[10.0, 0.0], [0.0, 10.0]]), 0.5)
    assert result.current_implementation_value == 2.5
    assert result.perfect_implementation_value == 10.0
    assert result.value == 7.5
