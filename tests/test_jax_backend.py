import numpy as np
import pytest

from voiage import evpi, set_backend
from voiage.config import get_default_dtype

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_evpi_jax():
    set_backend("jax")
    DEFAULT_DTYPE = get_default_dtype()
    nb_array = jnp.array([[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]], dtype=DEFAULT_DTYPE)
    expected_evpi = 6.0
    calculated_evpi = evpi(nb_array)
    np.testing.assert_allclose(calculated_evpi, expected_evpi, rtol=1e-6)
    set_backend("numpy")
