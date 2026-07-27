import numpy as np

from voiage.methods.signal import value_of_signal_information


def test_signal_information_value_uses_conditional_strategy() -> None:
    result = value_of_signal_information(np.array([[10.0, 0.0], [0.0, 10.0]]), ["a", "b"])
    assert result.value == 5.0
    assert result.signal_strategy_indices == {"a": 0, "b": 1}
