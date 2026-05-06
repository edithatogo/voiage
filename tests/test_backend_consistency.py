"""Integration checks for CLI and backend consistency."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from voiage import cli
from voiage.analysis import DecisionAnalysis
from voiage.main_backends import JAX_AVAILABLE
from voiage.schema import ValueArray

pytestmark = pytest.mark.integration

runner = CliRunner()

if TYPE_CHECKING:
    from pathlib import Path


def _write_net_benefit_csv(path: Path) -> np.ndarray:
    values = np.array(
        [
            [10.0, 12.0, 11.0],
            [9.0, 13.5, 10.5],
            [11.0, 12.5, 12.0],
            [10.5, 12.2, 11.8],
        ],
        dtype=float,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Strategy A", "Strategy B", "Strategy C"])
        writer.writerows(values.tolist())
    return values


def test_cli_evpi_matches_backend_and_jax_when_available(
    tmp_path: Path,
) -> None:
    """The CLI, NumPy backend, and JAX backend should agree on the same surface."""
    net_benefits_file = tmp_path / "net_benefits.csv"
    values = _write_net_benefit_csv(net_benefits_file)

    cli_result = runner.invoke(
        cli.app,
        ["--format", "json", "calculate-evpi", str(net_benefits_file)],
    )

    assert cli_result.exit_code == 0, cli_result.stdout
    payload = json.loads(cli_result.stdout)

    value_array = ValueArray.from_numpy(
        values, ["Strategy A", "Strategy B", "Strategy C"]
    )
    numpy_result = DecisionAnalysis(nb_array=value_array, backend="numpy").evpi()

    assert payload["command"] == "calculate-evpi"
    assert payload["metric"] == "EVPI"
    assert payload["value"] == pytest.approx(numpy_result)

    if JAX_AVAILABLE:
        jax_result = DecisionAnalysis(nb_array=value_array, backend="jax").evpi()
        assert jax_result == pytest.approx(numpy_result)
