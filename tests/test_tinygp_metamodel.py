import numpy as np
import pytest
import xarray as xr

from voiage.schema import ParameterSet

try:
    from voiage.metamodels import TinyGPMetamodel
except ImportError:
    TinyGPMetamodel = None


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)  # For reproducible tests
    data = {
        "param1": ("n_samples", np.random.rand(100)),
        "param2": ("n_samples", np.random.rand(100)),
    }
    x = ParameterSet(dataset=xr.Dataset(data))
    y = np.random.rand(100)
    return x, y


@pytest.mark.skipif(TinyGPMetamodel is None, reason="tinygp is not installed")
def test_tinygp_score_rmse(sample_data):
    """Test score and rmse methods of TinyGPMetamodel."""
    x, y = sample_data
    model = TinyGPMetamodel()

    # Test error before fit
    with pytest.raises(RuntimeError, match="The model has not been fitted yet."):
        model.score(x, y)
    with pytest.raises(RuntimeError, match="The model has not been fitted yet."):
        model.rmse(x, y)

    # Fit model
    model.fit(x, y)

    # Test score
    score = model.score(x, y)
    assert hasattr(score, "dtype") or isinstance(score, float)

    # Test rmse
    rmse = model.rmse(x, y)
    assert hasattr(rmse, "dtype") or isinstance(rmse, float)
