from __future__ import annotations

import json
from pathlib import Path


def _strategy_means(matrix: list[list[float]]) -> list[float]:
    sample_count = len(matrix)
    strategy_count = len(matrix[0])
    totals = [0.0] * strategy_count
    for row in matrix:
        for index, value in enumerate(row):
            totals[index] += value
    return [total / sample_count for total in totals]


def _row_max_mean(matrix: list[list[float]]) -> float:
    return sum(max(row) for row in matrix) / len(matrix)


def test_rust_evsi_stochastic_kernel_fixture_is_consistent() -> None:
    fixture_path = Path(
        "bindings/rust/tests/fixtures/evsi_stochastic_kernel.input.json"
    )
    expected_path = Path(
        "bindings/rust/tests/fixtures/evsi_stochastic_kernel.expected.json"
    )

    fixture = json.loads(fixture_path.read_text())
    expected = json.loads(expected_path.read_text())

    matrix = fixture["current_net_benefit"]["net_benefit"]
    trial_design = fixture["trial_design"]["arms"]
    sample_size = sum(arm["sample_size"] for arm in trial_design)

    assert fixture["method"] == expected["method"]
    assert expected["result"]["sample_size"] == sample_size
    assert expected["result"]["expected_current_value"] == max(_strategy_means(matrix))
    assert expected["result"]["expected_perfect_information"] == _row_max_mean(matrix)
    assert expected["result"]["evsi"] >= 0.0
    assert expected["reporting"]["seed"] == fixture["seed"]
