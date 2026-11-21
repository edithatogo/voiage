"""Installed controls for named Arrow and Polars credal inputs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

import polars as pl
import pyarrow as pa
import pytest

from voiage.contracts.normalized_input import NormalizedInputBundle
from voiage.exceptions import InputError
from voiage.ingestion.dataframe import from_dataframe

if TYPE_CHECKING:
    from pathlib import Path


def _bounds() -> pa.Table:
    return pa.table({"parameter_id": ["x", "y"], "lower": [0, 0], "upper": [1, 0]})


def _inequalities() -> pa.Table:
    return pa.table(
        {
            "constraint_id": ["cap", "floor", "cap", "floor"],
            "parameter_id": ["y", "x", "x", "y"],
            "coefficient": [0, -3, 3, 0],
            "upper": [1, -1, 1, -1],
        }
    )


def _strategies() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy_id": ["affine", "constant", "affine", "constant"],
            "parameter_id": ["y", "x", "x", "y"],
            "coefficient": [2, 0, 3, 0],
            "intercept": [0, 1, 0, 1],
        }
    )


def _solve(
    *,
    parameters: pa.Table | NormalizedInputBundle | None = None,
    strategies: pl.DataFrame | None = None,
    max_bases: int = 128,
    include_inequalities: bool = True,
) -> dict[str, object]:
    experimental = import_module("voiage.experimental")
    return cast(
        "dict[str, object]",
        experimental.credal_decisions(
            _bounds() if parameters is None else parameters,
            _strategies() if strategies is None else strategies,
            inequalities=_inequalities() if include_inequalities else None,
            max_bases=max_bases,
        ),
    )


def test_credal_inputs_keep_parameter_and_support_order_across_arrow_ipc(
    tmp_path: Path,
) -> None:
    bounds = from_dataframe(_bounds(), dataset_id="credal-bounds", allow_copy=False)
    path = tmp_path / "bounds.arrow"
    bounds.write_ipc(path)
    restored = NormalizedInputBundle.read_ipc(path)
    report = _solve(parameters=restored)
    assert report["contract_version"] == 1
    assert report["status"] == "feasible"
    assert report["parameter_ids"] == ["x", "y"]
    assert report["support_rows"] == [
        {"kind": "inequality", "constraint_id": "cap"},
        {"kind": "inequality", "constraint_id": "floor"},
        {"kind": "lower", "parameter_id": "x"},
        {"kind": "upper", "parameter_id": "x"},
        {"kind": "lower", "parameter_id": "y"},
        {"kind": "upper", "parameter_id": "y"},
    ]
    certificates = cast("dict[str, dict[str, object]]", report["e_admissibility"])
    assert certificates["affine"]["point"] == [["1", "3"], ["0", "1"]]


def test_credal_inputs_reject_missing_or_duplicate_coordinates() -> None:
    strategies = _strategies()
    with pytest.raises(ValueError, match="every parameter"):
        _solve(strategies=strategies.slice(1))
    with pytest.raises(ValueError, match="duplicate"):
        _solve(strategies=pl.concat([strategies, strategies.head(1)]))


def test_credal_inputs_reject_binary64_rounding_and_keep_budget_status() -> None:
    with pytest.raises(ValueError, match="exactly representable"):
        _solve(parameters=_bounds().set_column(2, "upper", pa.array([2**53 + 1, 0])))
    assert _solve(max_bases=1)["status"] == "indeterminate"


def test_credal_inputs_accept_a_box_without_additional_inequalities() -> None:
    result = _solve(include_inequalities=False)
    ranges = cast("dict[str, dict[str, dict[str, object]]]", result["ranges"])
    assert result["status"] == "feasible"
    assert ranges["affine"]["minimum"]["value"] == ["0", "1"]
    assert ranges["affine"]["maximum"]["value"] == ["3", "1"]
    assert result["support_rows"] == [
        {"kind": "lower", "parameter_id": "x"},
        {"kind": "upper", "parameter_id": "x"},
        {"kind": "lower", "parameter_id": "y"},
        {"kind": "upper", "parameter_id": "y"},
    ]


@pytest.mark.parametrize("budget", [0, True])
def test_credal_inputs_require_positive_integer_budget(budget: int) -> None:
    with pytest.raises(ValueError, match="positive integer basis"):
        _solve(max_bases=budget)


def test_credal_inputs_reject_absent_and_duplicate_parameters() -> None:
    bounds = _bounds()
    with pytest.raises(ValueError, match="at least one parameter"):
        _solve(parameters=bounds.slice(0, 0))
    repeated = bounds.set_column(0, "parameter_id", pa.array(["x", "x"]))
    with pytest.raises(ValueError, match="nonempty and unique"):
        _solve(parameters=repeated)


def test_credal_inputs_require_columns_and_an_unambiguous_bundle() -> None:
    bounds = _bounds()
    with pytest.raises(ValueError, match="requires unique columns"):
        _solve(parameters=bounds.drop_columns(["lower"]))
    first = from_dataframe(bounds, dataset_id="box", table_id="first")
    second = from_dataframe(bounds, dataset_id="box", table_id="second")
    bundle = NormalizedInputBundle(
        manifest=first.manifest.model_copy(
            update={"tables": first.manifest.tables + second.manifest.tables}
        ),
        tables={**first.tables, **second.tables},
    )
    with pytest.raises(ValueError, match="must contain one table"):
        _solve(parameters=bundle)


def test_credal_inputs_reject_unknown_coordinates_and_conflicting_constants() -> None:
    strategies = _strategies()
    with pytest.raises(ValueError, match="declared parameter"):
        _solve(strategies=strategies.with_columns(pl.lit("z").alias("parameter_id")))
    different_constants = strategies.with_columns(pl.Series("intercept", [0, 1, 0, 2]))
    with pytest.raises(ValueError, match="share its constant"):
        _solve(strategies=different_constants)


def test_credal_inputs_preserve_native_domain_error() -> None:
    reversed_bounds = _bounds().set_column(1, "lower", pa.array([2, 0]))
    with pytest.raises(InputError):
        _solve(parameters=reversed_bounds)


def test_credal_inputs_reject_misordered_native_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("voiage.experimental.credal")
    compute = module.compute_credal_decisions

    def mismatched_coordinates(problem: str, *, max_bases: int) -> dict[str, object]:
        result = compute(problem, max_bases=max_bases)
        result["parameter_ids"] = ["y", "x"]
        return result

    monkeypatch.setattr(module, "compute_credal_decisions", mismatched_coordinates)
    with pytest.raises(RuntimeError, match="incompatible contract"):
        _solve()
