from __future__ import annotations

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.sample_information import evsi
from voiage.schema import DecisionOption, ParameterSet, TrialDesign, ValueArray

pytestmark = pytest.mark.benchmark


def _value_array() -> ValueArray:
    values = np.column_stack([
        np.linspace(9.5, 10.5, 200),
        np.linspace(10.2, 11.2, 200),
    ])
    return ValueArray.from_numpy(values, ["standard care", "new treatment"])


def _parameter_set() -> ParameterSet:
    return ParameterSet.from_numpy_or_dict({
        "treatment_effect": np.linspace(0.1, 0.9, 200),
        "cost_shift": np.linspace(-0.2, 0.2, 200),
    })


def _trial_design() -> TrialDesign:
    return TrialDesign(arms=[
        DecisionOption(name="New Treatment", sample_size=25),
        DecisionOption(name="Standard Care", sample_size=25),
    ])


def _model(psa: ParameterSet) -> ValueArray:
    treatment = psa.parameters["treatment_effect"]
    cost_shift = psa.parameters["cost_shift"]
    values = np.column_stack([
        100.0 + 2.0 * cost_shift,
        101.0 + 4.0 * treatment - cost_shift,
    ])
    return ValueArray.from_numpy(values, ["standard care", "new treatment"])


def test_evpi_benchmark(benchmark) -> None:
    analysis = DecisionAnalysis(nb_array=_value_array())
    result = benchmark(analysis.evpi)
    assert result >= 0.0


def test_evppi_benchmark(benchmark) -> None:
    analysis = DecisionAnalysis(nb_array=_value_array(), parameter_samples=_parameter_set())
    result = benchmark(analysis.evppi, n_regression_samples=50)
    assert result >= 0.0


def test_evsi_benchmark(benchmark) -> None:
    result = benchmark(
        evsi,
        model_func=_model,
        psa_prior=_parameter_set(),
        trial_design=_trial_design(),
        method="efficient",
    )
    assert result >= 0.0
