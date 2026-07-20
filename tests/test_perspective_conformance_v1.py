"""Cross-package conformance for the versioned directional EVoP fixture."""

import json
from pathlib import Path

import numpy as np
import pytest

from voiage.methods.perspective import (
    METHOD_CONTRACT_VERSION,
    perspective_result_to_arrow,
    value_of_perspective,
    write_perspective_result_parquet,
)

FIXTURE = Path(__file__).parent / "fixtures" / "perspective_conformance_v1.json"


def test_directional_current_information_evop_fixture() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["method_contract_version"] == METHOD_CONTRACT_VERSION
    item = next(
        entry for entry in payload["fixtures"] if entry["id"] == "directional_regret"
    )

    health_system = value_of_perspective(
        np.asarray(item["values"]),
        strategy_names=item["strategies"],
        perspective_names=item["perspectives"],
        reference_perspective="health_system",
    )
    societal = value_of_perspective(
        np.asarray(item["values"]),
        strategy_names=item["strategies"],
        perspective_names=item["perspectives"],
        reference_perspective="societal",
    )

    assert health_system.switching_values[1] == pytest.approx(
        item["expected"]["health_system_to_societal"]
    )
    assert societal.switching_values[0] == pytest.approx(
        item["expected"]["societal_to_health_system"]
    )
    assert (
        health_system.diagnostics["estimand"] == "directional_current_information_evop"
    )


def test_tie_policy_is_explicit() -> None:
    values = np.array([[[1.0], [1.0]]])
    with pytest.raises(Exception, match="Tied expected-value"):
        value_of_perspective(values, perspective_names=["p"], tie_policy="error")

    split = value_of_perspective(values, perspective_names=["p"], tie_policy="split")
    assert split.value == 0.0
    assert split.diagnostics["ties_detected"] == [True]


def test_first_tie_policy_uses_selected_strategy_for_uncertainty() -> None:
    values = np.array(
        [
            [[0.0, 0.0], [-1.0, 2.0]],
            [[2.0, 2.0], [-1.0, 0.0]],
        ]
    )
    first = value_of_perspective(
        values,
        perspective_names=["reference", "tied"],
        tie_policy="first",
    )
    split = value_of_perspective(
        values,
        perspective_names=["reference", "tied"],
        tie_policy="split",
    )

    assert first.diagnostics["switching_standard_errors"] == pytest.approx([0.0, 0.0])
    assert split.diagnostics["switching_standard_errors"] == pytest.approx([0.0, 1.0])


def test_directional_result_uses_arrow_interchange(tmp_path) -> None:
    result = value_of_perspective(
        np.array([[[10.0, 0.0], [0.0, 20.0]]]),
        strategy_names=["A", "B"],
        perspective_names=["left", "right"],
    )
    table = perspective_result_to_arrow(result)
    path = tmp_path / "perspective.parquet"
    write_perspective_result_parquet(result, str(path))

    assert table.num_rows == 4
    assert table.schema.metadata[b"voiage.method_contract_version"] == b"1.1.0"
    assert path.read_bytes()[:4] == b"PAR1"
