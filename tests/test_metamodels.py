# tests/test_metamodels.py

import numpy as np
import pytest
import xarray as xr

from voiage.metamodels import FlaxMetamodel, tinygpMetamodel
from voiage.schema import ParameterSet


@pytest.fixture()
def sample_data():
    data = {
        "param1": ("n_samples", np.random.rand(100)),
        "param2": ("n_samples", np.random.rand(100)),
    }
    x = ParameterSet(dataset=xr.Dataset(data))
    y = np.random.rand(100)
    return x, y


def test_flax_metamodel(sample_data):
    x, y = sample_data
    model = FlaxMetamodel(n_epochs=10)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred.shape == (100, 1)


def test_tinygp_metamodel(sample_data):
    x, y = sample_data
    model = TinyGPMetamodel()
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred.shape == (100,)
