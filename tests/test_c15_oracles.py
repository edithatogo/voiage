from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pytest

from voiage.c15_oracles import decimal_evpi, numpy_evpi
from voiage.main_backends import JaxBackend, NumpyBackend
from voiage.methods.basic import evpi
from voiage.schema import ValueArray

FIXTURE = Path(__file__).parents[1] / "specs/numerical-reference/v1/evpi-cases.json"


def _cases() -> list[dict[str, object]]:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    return [case for case in cases if case.get("assurance_profile") == "c15"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["id"]))
def test_cases_declare_classification_units_tolerances_and_assumptions(
    case: dict[str, object],
) -> None:
    assert case["classification"] in {
        "boundary",
        "near_tie",
        "tail",
        "higher_dimensional",
    }
    assert case["net_benefit_unit"] == "NZD_2025_per_person"
    assert case["expected"]["unit"] == "net_benefit_unit_per_decision"
    assert set(case["tolerances"]) == {
        "decimal_absolute",
        "numpy_absolute",
        "native_absolute",
        "jax_absolute",
        "polars_absolute",
    }
    assert case["assumptions"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["id"]))
def test_decimal_numpy_and_native_voiage_surfaces_match(
    case: dict[str, object],
) -> None:
    matrix = np.asarray(case["net_benefits"], dtype=np.float64)
    expected = Decimal(case["expected"]["decimal_value"])
    tolerances = case["tolerances"]
    assert decimal_evpi(case["net_benefits"]) == expected
    assert numpy_evpi(matrix) == pytest.approx(
        float(expected), abs=tolerances["numpy_absolute"]
    )
    assert evpi(matrix) == pytest.approx(
        float(expected), abs=tolerances["native_absolute"]
    )
    assert NumpyBackend().evpi(matrix) == pytest.approx(
        float(expected), abs=tolerances["native_absolute"]
    )
    names = [f"strategy-{index}" for index in range(matrix.shape[1])]
    assert evpi(ValueArray.from_numpy(matrix, names)) == pytest.approx(
        float(expected), abs=tolerances["native_absolute"]
    )


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["id"]))
def test_available_jax_and_polars_backends_match(case: dict[str, object]) -> None:
    jnp = pytest.importorskip("jax.numpy")
    pl = pytest.importorskip("polars")
    matrix = case["net_benefits"]
    expected = float(case["expected"]["value"])
    values = jnp.asarray(matrix, dtype=jnp.float64)
    jax_value = jnp.mean(jnp.max(values, axis=1)) - jnp.max(jnp.mean(values, axis=0))
    assert float(jax_value) == pytest.approx(
        expected, abs=case["tolerances"]["jax_absolute"]
    )
    assert JaxBackend().evpi(matrix) == pytest.approx(
        expected, abs=case["tolerances"]["jax_absolute"]
    )
    frame = pl.DataFrame(matrix, orient="row")
    polars_value = frame.max_horizontal().mean() - max(
        frame[column].mean() for column in frame.columns
    )
    assert polars_value == pytest.approx(
        expected, abs=case["tolerances"]["polars_absolute"]
    )


@pytest.mark.parametrize(
    "value", [None, "text", [], [[1]], [[1, 2], [3, float("inf")]]]
)
def test_oracles_reject_invalid_or_nonfinite_inputs(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        decimal_evpi(value)
    with pytest.raises((TypeError, ValueError)):
        numpy_evpi(value)


@pytest.mark.parametrize(
    "value",
    [[[1, 2], [3]], [["not-a-number", 1], [2, 3]], [[1], [2]]],
)
def test_decimal_oracle_rejects_ragged_nonnumeric_or_single_strategy(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        decimal_evpi(value)
