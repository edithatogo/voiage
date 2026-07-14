from __future__ import annotations

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

pytestmark = pytest.mark.benchmark


def _value_array() -> ValueArray:
    values = np.column_stack(
        [
            np.linspace(9.5, 10.5, 200),
            np.linspace(10.2, 11.2, 200),
        ]
    )
    return ValueArray.from_numpy(values, ["standard care", "new treatment"])


def test_evpi_numpy_benchmark(benchmark) -> None:
    analysis = DecisionAnalysis(nb_array=_value_array(), backend="numpy")
    result = benchmark(analysis.evpi)
    assert result >= 0.0


def test_evpi_jax_benchmark(benchmark) -> None:
    analysis = DecisionAnalysis(nb_array=_value_array(), backend="jax")
    result = benchmark(analysis.evpi)
    assert result >= 0.0


def test_evpi_numpy_and_jax_agree() -> None:
    numpy_result = DecisionAnalysis(nb_array=_value_array(), backend="numpy").evpi()
    jax_result = DecisionAnalysis(nb_array=_value_array(), backend="jax").evpi()
    assert numpy_result == pytest.approx(jax_result, rel=1e-9, abs=1e-9)
