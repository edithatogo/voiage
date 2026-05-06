"""Integration smoke tests for the main DecisionAnalysis methods."""

import numpy as np

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray


def _analysis() -> DecisionAnalysis:
    values = np.array(
        [
            [100.0, 103.0, 102.0],
            [101.0, 102.5, 101.5],
            [99.5, 104.0, 100.5],
            [100.5, 101.5, 103.0],
        ],
        dtype=np.float64,
    )
    value_array = ValueArray.from_numpy(values, ["standard care", "new treatment", "monitor"])
    parameter_samples = {
        "effect": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        "bias": np.array([0.0, 0.1, 0.0, 0.2], dtype=np.float64),
    }
    return DecisionAnalysis(nb_array=value_array, parameter_samples=parameter_samples)


def _surface() -> ValueArray:
    base = np.array(
        [
            [100.0, 103.0, 102.0],
            [101.0, 102.5, 101.5],
            [99.5, 104.0, 100.5],
            [100.5, 101.5, 103.0],
        ],
        dtype=np.float64,
    )
    surface = np.stack([base, base + 1.0, base + 2.0], axis=-1)
    return ValueArray.from_numpy_perspectives(
        surface,
        ["standard care", "new treatment", "monitor"],
        ["wtp_0", "wtp_1", "wtp_2"],
    )


def test_decision_analysis_core_methods_smoke() -> None:
    analysis = _analysis()

    evpi_value = analysis.evpi()
    evppi_value = analysis.evppi(parameters_of_interest=["effect"])
    enbs_value = analysis.enbs(research_cost=0.5)

    assert evpi_value >= 0.0
    assert evppi_value >= 0.0
    assert evppi_value <= evpi_value + 1e-9
    assert enbs_value >= 0.0


def test_decision_analysis_frontier_methods_smoke() -> None:
    analysis_2d = _analysis()
    analysis_3d = DecisionAnalysis(nb_array=_surface())

    ceaf_result = analysis_3d.ceaf([0.0, 1.0, 2.0])
    dominance_result = analysis_2d.dominance([100.0, 102.0, 104.0], [1.0, 1.1, 1.2])
    heterogeneity_result = analysis_2d.value_of_heterogeneity(["A", "B", "A", "B"])
    distributional_result = analysis_2d.value_of_distributional_equity(
        ["A", "B", "A", "B"]
    )
    implementation_result = analysis_2d.value_of_implementation()
    perspective_surface = DecisionAnalysis(nb_array=_surface())
    perspective_result = perspective_surface.value_of_perspective(
        perspective_names=["payer", "societal", "public"]
    )

    assert ceaf_result is not None
    assert dominance_result is not None
    assert heterogeneity_result is not None
    assert distributional_result is not None
    assert implementation_result is not None
    assert perspective_result is not None
