"""End-to-end workflow smoke tests for the main VOI path."""

from pathlib import Path

import numpy as np

from voiage.analysis import DecisionAnalysis
from voiage.methods.sample_information import evsi
from voiage.plot.ceac import plot_ceac
from voiage.schema import DecisionOption, ParameterSet, TrialDesign, ValueArray


def _value_array() -> ValueArray:
    values = np.array(
        [
            [100.0, 103.0],
            [101.0, 102.5],
            [99.5, 104.0],
            [100.5, 101.5],
        ],
        dtype=np.float64,
    )
    return ValueArray.from_numpy(values, ["standard care", "new treatment"])


def _parameter_set() -> ParameterSet:
    return ParameterSet.from_numpy_or_dict(
        {
            "effect": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            "cost_shift": np.array([0.0, 0.1, 0.0, 0.2], dtype=np.float64),
        }
    )


def _model(psa: ParameterSet) -> ValueArray:
    effect = psa.parameters["effect"]
    cost_shift = psa.parameters["cost_shift"]
    values = np.column_stack(
        [
            100.0 + cost_shift,
            101.0 + 3.0 * effect - cost_shift,
        ]
    )
    return ValueArray.from_numpy(values, ["standard care", "new treatment"])


def _trial_design() -> TrialDesign:
    return TrialDesign(
        arms=[
            DecisionOption(name="Standard Care", sample_size=10),
            DecisionOption(name="New Treatment", sample_size=10),
        ]
    )


def test_end_to_end_workflow(tmp_path: Path) -> None:
    value_array = _value_array()
    parameter_set = _parameter_set()
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameter_set)

    evpi_value = analysis.evpi()
    evppi_value = analysis.evppi(parameters_of_interest=["effect"])
    evsi_value = evsi(
        model_func=_model,
        psa_prior=parameter_set,
        trial_design=_trial_design(),
        method="efficient",
    )

    assert evpi_value >= 0.0
    assert evppi_value >= 0.0
    assert evppi_value <= evpi_value + 1e-9
    assert evsi_value >= 0.0

    surface = ValueArray.from_numpy_perspectives(
        np.stack(
            [
                value_array.numpy_values,
                value_array.numpy_values + 1.0,
                value_array.numpy_values + 2.0,
            ],
            axis=-1,
        ),
        ["standard care", "new treatment"],
        ["wtp_0", "wtp_1", "wtp_2"],
    )
    ax = plot_ceac(surface, wtp_thresholds=[0.0, 1.0, 2.0])
    output_file = tmp_path / "ceac.png"
    ax.figure.savefig(output_file)
    assert output_file.exists()
