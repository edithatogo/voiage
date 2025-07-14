import numpy as np
import pytest

from voiage import evpi, evppi
from voiage.schema import ValueArray, ParameterSet
from voiage.config import DEFAULT_DTYPE


def test_evpi_wrapper():
    nb_array = np.array([[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]], dtype=DEFAULT_DTYPE)
    expected_evpi = 6.0
    calculated_evpi = evpi(nb_array)
    np.testing.assert_allclose(calculated_evpi, expected_evpi)


SKLEARN_AVAILABLE = True
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    SKLEARN_AVAILABLE = False

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_evppi_wrapper():
    nb_array = np.array([[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]], dtype=DEFAULT_DTYPE)
    parameter_samples = np.array([1, 2, 3, 4, 5], dtype=DEFAULT_DTYPE)
    calculated_evppi = evppi(nb_array, {"p1": parameter_samples})
    assert calculated_evppi >= 0
