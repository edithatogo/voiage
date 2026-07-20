from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from voiage import (
    DecisionOption,
    ParameterSet,
    TrialDesign,
    ValueArray,
    enbs,
    evpi,
    evppi,
    evsi,
)
from voiage.exceptions import InputError
from voiage.methods.ceaf import calculate_ceaf
from voiage.methods.dominance import calculate_dominance


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_stable_methods_reject_non_finite_inputs(non_finite: float) -> None:
    with pytest.raises(InputError, match="finite"):
        evpi(np.array([[0.0, non_finite], [1.0, 2.0]]))

    parameters = ParameterSet.from_numpy_or_dict(
        {"theta": np.array([0.0, 1.0, 2.0, 3.0])}
    )
    with pytest.raises(InputError, match="finite"):
        evppi(
            np.array([[0.0, 1.0], [1.0, 0.0], [2.0, non_finite], [3.0, 4.0]]),
            parameters,
            ["theta"],
        )

    prior = ParameterSet.from_numpy_or_dict(
        {"theta": np.array([0.0, 1.0, non_finite, 3.0])}
    )
    design = TrialDesign(arms=[DecisionOption(name="A", sample_size=1)])

    def model(psa: ParameterSet) -> ValueArray:
        values = np.column_stack((psa.parameters["theta"], np.ones(psa.n_samples)))
        return ValueArray.from_numpy(values, ["A", "B"])

    with pytest.raises(InputError, match="finite"):
        evsi(model, prior, design, method="moment_based")

    with pytest.raises(InputError, match="finite"):
        enbs(non_finite, 1.0)

    dataset = xr.Dataset(
        {
            "net_benefit": (
                ("n_samples", "n_strategies", "threshold"),
                [[[1.0, 2.0], [2.0, 1.0]]],
            )
        },
        coords={"strategy": ("n_strategies", ["A", "B"]), "threshold": [0, 1]},
    )
    with pytest.raises(InputError, match="finite"):
        calculate_ceaf(ValueArray.from_dataset(dataset), [0.0, non_finite])

    with pytest.raises(InputError, match="finite"):
        calculate_dominance([1.0, non_finite], [1.0, 2.0])
