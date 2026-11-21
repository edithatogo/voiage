"""Experimental exact credal decisions from named affine inputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from voiage._runtime import compute_credal_decisions
from voiage.contracts.normalized_input import NormalizedInputBundle

from .paired_draws import _arrow_table, _finite_number, _identifier

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

type CredalFrame = pa.Table | pl.DataFrame | NormalizedInputBundle


def credal_decisions(
    parameters: CredalFrame,
    strategies: CredalFrame,
    *,
    inequalities: CredalFrame | None = None,
    max_bases: int,
) -> dict[str, object]:
    """Evaluate a bounded affine credal decision problem in Rust.

    Parameter rows declare `parameter_id`, `lower`, and `upper` in coordinate
    order. Strategy rows declare `strategy_id`, `parameter_id`, `coefficient`,
    and `intercept`. Optional inequality rows use `constraint_id`,
    `parameter_id`, `coefficient`, and `upper`. Each strategy or inequality
    supplies a coefficient for every parameter and repeats its constant on
    each row. Single-table input bundles retain their Arrow interchange format.

    Witness values are decimal-string fraction pairs. Support multipliers
    follow `support_rows`. The native solver reports infeasibility or
    incomplete basis searches in the result status.
    """
    if type(max_bases) is not int or max_bases <= 0:
        raise ValueError("A positive integer basis budget is required")
    bounds = _select(parameters, ("parameter_id", "lower", "upper"))
    coordinates: dict[str, dict[str, object]] = {}
    for row in bounds.to_pylist():
        name = row["parameter_id"]
        if not _identifier(name) or name in coordinates:
            raise ValueError("Parameter IDs must be nonempty and unique")
        coordinates[name] = {
            "id": name,
            "lower": _binary64(row["lower"]),
            "upper": _binary64(row["upper"]),
        }
    parameter_ids = tuple(coordinates)
    if not parameter_ids:
        raise ValueError("The credal problem requires at least one parameter")
    objectives = _affine_rows(strategies, "strategy_id", "intercept", parameter_ids)
    constraints = (
        {}
        if inequalities is None
        else _affine_rows(inequalities, "constraint_id", "upper", parameter_ids)
    )
    problem = {
        "parameters": list(coordinates.values()),
        "inequalities": [
            {"upper": maximum, "coefficients": coefficients}
            for maximum, coefficients in constraints.values()
        ],
        "strategies": [
            {"strategy_id": name, "intercept": value, "coefficients": coefficients}
            for name, (value, coefficients) in objectives.items()
        ],
    }
    result = compute_credal_decisions(
        json.dumps(problem, allow_nan=False), max_bases=max_bases
    )
    if result["contract_version"] != 1 or result["parameter_ids"] != list(
        parameter_ids
    ):
        raise RuntimeError("The native credal result uses an incompatible contract")
    result["support_rows"] = [
        *({"kind": "inequality", "constraint_id": name} for name in constraints),
        *(
            {"kind": endpoint, "parameter_id": parameter}
            for parameter in parameter_ids
            for endpoint in ("lower", "upper")
        ),
    ]
    return result


def _select(frame: CredalFrame, columns: tuple[str, ...]) -> pa.Table:
    if isinstance(frame, NormalizedInputBundle):
        if len(frame.tables) != 1:
            raise ValueError("A credal input bundle must contain one table")
        table = next(iter(frame.tables.values()))
    else:
        table = _arrow_table(frame)
    if len(set(table.column_names)) != table.num_columns or not set(columns).issubset(
        table.column_names
    ):
        raise ValueError(f"Credal input requires unique columns {columns!r}")
    return table.select(columns)


def _binary64(value: object) -> float:
    if not _finite_number(value) or float(value) != value:
        raise ValueError(
            "Credal inputs must be finite and exactly representable as binary64"
        )
    return float(value)


def _affine_rows(
    frame: CredalFrame,
    id_column: str,
    constant_column: str,
    parameter_ids: tuple[str, ...],
) -> dict[str, tuple[float, list[float]]]:
    table = _select(frame, (id_column, "parameter_id", "coefficient", constant_column))
    expected = set(parameter_ids)
    groups: dict[str, tuple[float, dict[str, float]]] = {}
    for row in table.to_pylist():
        name, parameter = row[id_column], row["parameter_id"]
        if not (_identifier(name) and _identifier(parameter) and parameter in expected):
            raise ValueError(
                "Affine rows require a valid identity and a declared parameter"
            )
        constant = _binary64(row[constant_column])
        declared, coefficients = groups.setdefault(name, (constant, {}))
        if constant != declared:
            raise ValueError("Every row for an affine function must share its constant")
        if parameter in coefficients:
            raise ValueError(
                "Affine rows contain a duplicate parameter for the same function"
            )
        coefficients[parameter] = _binary64(row["coefficient"])
    if any(set(coefficients) != expected for _, coefficients in groups.values()):
        raise ValueError(
            "Each affine function must provide a coefficient for every parameter"
        )
    return {
        name: (constant, [coefficients[parameter] for parameter in parameter_ids])
        for name, (constant, coefficients) in groups.items()
    }
