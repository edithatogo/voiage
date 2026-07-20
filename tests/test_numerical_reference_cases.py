"""Independent hand-calculated numerical references for runtime conformance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "specs"
    / "numerical-reference"
    / "v1"
    / "evpi-cases.json"
)


def _cases() -> list[dict[str, object]]:
    document = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    assert document["schema_version"] == "1.0.0"
    assert document["method"] == "evpi"
    return document["cases"]


@pytest.mark.parametrize("case", _cases())
def test_reference_derivation_is_hand_reproducible(case: dict[str, object]) -> None:
    values = np.asarray(case["net_benefits"], dtype=np.float64)
    derivation = case["derivation"]
    expected = case["expected"]

    strategy_means = values.mean(axis=0)
    expected_current = float(strategy_means.max())
    expected_perfect = float(values.max(axis=1).mean())
    hand_evpi = expected_perfect - expected_current

    assert strategy_means.tolist() == pytest.approx(derivation["strategy_means"])
    assert expected_current == pytest.approx(derivation["expected_current_value"])
    assert expected_perfect == pytest.approx(derivation["expected_perfect_information"])
    assert hand_evpi == pytest.approx(expected["value"], abs=expected["atol"])
    assert expected["unit"] == "net_benefit_unit_per_decision"
    assert case["provenance"] == "independent_hand_calculation"


@pytest.mark.parametrize("case", _cases())
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_python_backends_match_independent_reference(
    case: dict[str, object], backend: str
) -> None:
    values = np.asarray(case["net_benefits"], dtype=np.float64)
    expected = case["expected"]
    names = [f"strategy-{index}" for index in range(values.shape[1])]
    value_array = ValueArray.from_numpy(values, names)

    result = DecisionAnalysis(nb_array=value_array, backend=backend).evpi()

    assert result == pytest.approx(expected["value"], abs=expected["atol"])
