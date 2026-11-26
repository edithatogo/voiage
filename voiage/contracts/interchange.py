"""Arrow interchange for canonical v2 analysis-result envelopes."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math
from pathlib import Path  # noqa: TC003 - public API runtime annotation
from typing import Literal, TypeVar, cast

import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq
from pydantic import model_validator

from voiage.contracts.analysis import AnalysisResult, ContractModel

PayloadT = TypeVar("PayloadT", bound=ContractModel)
SIMULATION_OUTCOME_PROFILE_VERSION_V1 = "voiage.simulation-outcome.v1"


def schema_fingerprint(schema: pa.Schema) -> str:
    """Return the canonical SHA-256 identity of an Arrow logical schema."""
    fields = [
        {"arrow_type": str(field.type), "name": field.name, "nullable": field.nullable}
        for field in schema.remove_metadata()
    ]
    canonical = json.dumps(fields, separators=(",", ":"), sort_keys=True)
    return sha256(canonical.encode()).hexdigest()


def _with_interchange_marker(table: pa.Table, marker: bytes) -> pa.Table:
    """Replace only the shared container marker while preserving identity."""
    return table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"vop_voiage.interchange": marker}
    )


def analysis_result_table[PayloadT: ContractModel](
    result: AnalysisResult[PayloadT],
) -> pa.Table:
    """Return a one-row Arrow table with stable contract metadata."""
    result_json = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    table = pa.table(
        {
            "analysis_id": [result.analysis_id],
            "method_family": [result.method_family],
            "schema_version": [result.schema_version],
            "result_json": [result_json],
        }
    )
    fingerprint = schema_fingerprint(table.schema)
    return table.replace_schema_metadata(
        {
            b"voiage.contract": b"analysis-result",
            b"voiage.schema_version": result.schema_version.encode("ascii"),
            b"voiage.arrow_schema_version": (
                result.interchange_identity.arrow_schema_version.encode("ascii")
            ),
            b"vop_voiage.contract_version": result.schema_version.encode("ascii"),
            b"vop_voiage.schema_id": b"analysis-result",
            b"vop_voiage.schema_version": (
                result.interchange_identity.arrow_schema_version.encode("ascii")
            ),
            b"vop_voiage.schema_fingerprint": fingerprint.encode("ascii"),
            b"vop_voiage.producer": b"voiage",
            b"vop_voiage.method_contract_version": (
                result.method_contract_version.encode("utf-8")
            ),
            b"vop_voiage.interchange": b"apache-arrow",
        }
    )


def write_analysis_result_ipc[PayloadT: ContractModel](
    result: AnalysisResult[PayloadT], path: Path
) -> None:
    """Write a deterministic Arrow IPC file for one result envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _with_interchange_marker(analysis_result_table(result), b"apache-arrow-ipc")
    with ipc.new_file(path, table.schema) as writer:
        writer.write_table(table)


def write_analysis_result_parquet[PayloadT: ContractModel](
    result: AnalysisResult[PayloadT], path: Path
) -> None:
    """Write a Zstandard-compressed Parquet file for one result envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _with_interchange_marker(
        analysis_result_table(result), b"apache-arrow-parquet"
    )
    pq.write_table(table, path, compression="zstd")


_OUTCOME_CONTRACT_KEY = b"voiage.contract"
_OUTCOME_VERSION_KEY = b"voiage.simulation_outcome.profile_version"
_OUTCOME_METADATA_KEY = b"voiage.simulation_outcome.metadata"
_OUTCOME_FINGERPRINT_KEY = b"vop_voiage.schema_fingerprint"
_OUTCOME_INTERCHANGE_KEY = b"vop_voiage.interchange"
_REQUIRED_OUTCOME_META_KEYS = (
    _OUTCOME_CONTRACT_KEY,
    _OUTCOME_VERSION_KEY,
    _OUTCOME_METADATA_KEY,
    _OUTCOME_FINGERPRINT_KEY,
    _OUTCOME_INTERCHANGE_KEY,
)
_CORE_OUTCOME_COLUMNS = (
    "draw_namespace",
    "draw_key",
    "strategy_id",
    "value",
    "weight",
)
_PARAM_COLUMN_PREFIX = "param::"


def _as_finite_float(raw: object, field_name: str) -> float:
    """Require a finite numeric value without coercing booleans."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise TypeError(f"{field_name} must be a finite number")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if value == 0.0 else value


def _validate_draw_weights(weights: list[float], semantics: str) -> None:
    """Enforce equal, probability, or relative_weight draw constraints."""
    if not weights:
        raise ValueError("outcomes must not be empty")
    total = math.fsum(weights)
    if semantics == "equal":
        first = weights[0]
        if first <= 0.0 or any(
            weight <= 0.0 or abs(weight - first) > 1e-12 * first for weight in weights
        ):
            raise ValueError(
                "equal weight semantics requires identical positive weights for every draw"
            )
    elif semantics == "probability":
        if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
            raise ValueError(f"probability weights must sum to 1.0 (got {total})")
    elif semantics == "relative_weight":
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("relative weights must have a finite positive sum")
    else:
        raise ValueError(f"unsupported weight semantics: {semantics!r}")


def _validate_mapping(raw: object, label: str) -> Mapping[str, object]:
    """Require a JSON-like mapping with string keys."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast("Mapping[str, object]", raw)


def validate_untrimmed_nonempty(value: str, field_name: str) -> None:
    """Require a non-empty string with no leading or trailing whitespace."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain leading or trailing whitespace"
        )


class SimulationOutcomeProvenance(ContractModel):
    """Producer provenance preserved on simulation-outcome interchange tables."""

    producer_id: str
    producer_version: str
    model_revision: str
    valuation_source: str
    artifact_digest: str

    @model_validator(mode="after")
    def validate_identifiers(self) -> SimulationOutcomeProvenance:
        """Reject blank or padded provenance identifiers without coercing them."""
        for field_name, value in (
            ("producer_id", self.producer_id),
            ("producer_version", self.producer_version),
            ("model_revision", self.model_revision),
            ("valuation_source", self.valuation_source),
            ("artifact_digest", self.artifact_digest),
        ):
            validate_untrimmed_nonempty(value, field_name)
        return self


class SimulationOutcomeValueDeclaration(ContractModel):
    """Explicit unit, perspective, convention, and scope declaration."""

    unit: str
    perspective: str
    convention: Literal["utility", "net_benefit", "loss"]
    objective: Literal["maximize", "minimize"]
    currency_price_year: str | None = None
    population_basis: str | None = None
    horizon: str | None = None
    discounting_status: str | None = None

    @model_validator(mode="after")
    def validate_declaration(self) -> SimulationOutcomeValueDeclaration:
        """Enforce non-empty declarations and convention/objective compatibility."""
        validate_untrimmed_nonempty(self.unit, "unit")
        validate_untrimmed_nonempty(self.perspective, "perspective")
        for field_name, optional_value in (
            ("currency_price_year", self.currency_price_year),
            ("population_basis", self.population_basis),
            ("horizon", self.horizon),
            ("discounting_status", self.discounting_status),
        ):
            if optional_value is not None:
                validate_untrimmed_nonempty(optional_value, field_name)
        expected_objective = "minimize" if self.convention == "loss" else "maximize"
        if self.objective != expected_objective:
            raise ValueError(
                "incompatible value convention and objective direction without implicit conversion"
            )
        return self


class SimulationOutcomeProfileMetadata(ContractModel):
    """Immutable metadata header embedded alongside simulation-outcome tables."""

    profile_version: Literal["voiage.simulation-outcome.v1"]
    provenance: SimulationOutcomeProvenance
    draw_namespace: str
    strategy_ids: tuple[str, ...]
    value_declaration: SimulationOutcomeValueDeclaration
    weight_semantics: Literal["equal", "probability", "relative_weight"]
    draw_keys: tuple[str, ...]
    parameter_names: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_profile_metadata(self) -> SimulationOutcomeProfileMetadata:
        """Reject empty or duplicated strategy, draw, and parameter identifiers."""
        validate_untrimmed_nonempty(self.draw_namespace, "draw_namespace")
        if not self.strategy_ids:
            raise ValueError("strategy_ids must not be empty")
        seen_strategies: set[str] = set()
        for strategy in self.strategy_ids:
            validate_untrimmed_nonempty(strategy, "strategy_ids")
            if strategy in seen_strategies:
                raise ValueError("strategy_ids must be unique")
            seen_strategies.add(strategy)
        if not self.draw_keys:
            raise ValueError("draw_keys must not be empty")
        seen_draws: set[str] = set()
        for draw_key in self.draw_keys:
            validate_untrimmed_nonempty(draw_key, "draw_keys")
            if draw_key in seen_draws:
                raise ValueError("draw_keys must be unique")
            seen_draws.add(draw_key)
        if self.parameter_names is not None:
            if not self.parameter_names:
                raise ValueError("parameter_names must not be empty when supplied")
            seen_params: set[str] = set()
            for param_name in self.parameter_names:
                validate_untrimmed_nonempty(param_name, "parameter_names")
                if param_name in seen_params:
                    raise ValueError("parameter_names must be unique")
                seen_params.add(param_name)
        return self


def _validate_string(raw: object, label: str) -> str:
    """Require an untrimmed non-empty string."""
    if not isinstance(raw, str):
        raise TypeError(f"{label} must be a string")
    validate_untrimmed_nonempty(raw, label)
    return raw


def _validate_weight_semantics(
    raw: object,
) -> Literal["equal", "probability", "relative_weight"]:
    """Require one of the declared simulation-outcome weight modes."""
    semantics = _validate_string(raw, "weight_semantics")
    if semantics == "equal":
        return "equal"
    if semantics == "probability":
        return "probability"
    if semantics == "relative_weight":
        return "relative_weight"
    raise ValueError(f"unsupported weight_semantics: {semantics!r}")


def _validate_outcome_rows(
    raw_outcomes: object,
    *,
    draw_namespace: str,
    strategy_ids: tuple[str, ...],
    profile_decl: SimulationOutcomeValueDeclaration,
    weight_semantics: str,
) -> tuple[
    list[dict[str, object]],
    tuple[str, ...],
    dict[str, float],
]:
    """Validate paired `(draw_key, strategy_id)` records while preserving row order."""
    if not isinstance(raw_outcomes, Sequence) or isinstance(raw_outcomes, str | bytes):
        raise TypeError("outcomes must be a non-empty sequence")
    rows = list(raw_outcomes)
    if not rows:
        raise ValueError("outcomes must not be empty")

    strategy_set = set(strategy_ids)
    draws: dict[str, tuple[float, set[str]]] = {}
    validated_rows: list[dict[str, object]] = []
    for raw_row in rows:
        row = _validate_mapping(raw_row, "outcome record")
        row_ns = _validate_string(row.get("draw_namespace"), "draw_namespace")
        if row_ns != draw_namespace:
            raise ValueError("outcome record draw_namespace mismatch")
        draw_key = _validate_string(row.get("draw_key"), "draw_key")
        strategy_id = _validate_string(row.get("strategy_id"), "strategy_id")
        if strategy_id not in strategy_set:
            raise ValueError(f"unknown strategy_id {strategy_id!r}")
        value = _as_finite_float(row.get("value"), "value")
        weight = _as_finite_float(row.get("weight"), "weight")
        if weight < 0.0:
            raise ValueError(f"negative weight for draw_key {draw_key!r}")
        if "declaration" in row and row["declaration"] is not None:
            record_decl = SimulationOutcomeValueDeclaration.model_validate(
                row["declaration"]
            )
            if record_decl != profile_decl:
                raise ValueError(
                    "per-record value declaration conflicts with profile declaration"
                )
        existing = draws.get(draw_key)
        if existing is None:
            draws[draw_key] = (weight, {strategy_id})
        else:
            existing_weight, seen_strategies = existing
            if existing_weight != weight:
                raise ValueError(f"conflicting draw weight for draw_key {draw_key!r}")
            if strategy_id in seen_strategies:
                raise ValueError(
                    f"duplicate outcome record for ({draw_key!r}, {strategy_id!r})"
                )
            seen_strategies.add(strategy_id)
        validated_rows.append(
            {
                "draw_namespace": row_ns,
                "draw_key": draw_key,
                "strategy_id": strategy_id,
                "value": value,
                "weight": weight,
            }
        )

    sorted_keys = tuple(sorted(draws))
    draw_weights: dict[str, float] = {}
    for draw_key in sorted_keys:
        weight, seen_strategies = draws[draw_key]
        if seen_strategies != strategy_set:
            missing = min(strategy_set - seen_strategies)
            raise ValueError(
                f"incomplete draw {draw_key!r}: missing strategy {missing!r}"
            )
        draw_weights[draw_key] = weight
    _validate_draw_weights([draw_weights[key] for key in sorted_keys], weight_semantics)
    return validated_rows, sorted_keys, draw_weights


def _validate_parameter_samples(
    raw_samples: object,
    *,
    draw_namespace: str,
    expected_draw_keys: tuple[str, ...],
) -> tuple[tuple[str, ...] | None, dict[str, dict[str, float]]]:
    """Validate optional parameter draws joined strictly by `draw_key`."""
    if raw_samples is None:
        return None, {}
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, str | bytes):
        raise TypeError("parameter_samples must be a sequence when provided")
    expected_set = set(expected_draw_keys)
    by_draw: dict[str, dict[str, float]] = {}
    param_names: tuple[str, ...] | None = None
    for raw_sample in raw_samples:
        sample = _validate_mapping(raw_sample, "parameter sample")
        sample_ns = _validate_string(sample.get("draw_namespace"), "draw_namespace")
        if sample_ns != draw_namespace:
            raise ValueError("parameter sample draw_namespace mismatch")
        draw_key = _validate_string(sample.get("draw_key"), "draw_key")
        if draw_key in by_draw:
            raise ValueError(f"duplicate parameter draw {draw_key!r}")
        if draw_key not in expected_set:
            raise ValueError(f"unmatched parameter draw {draw_key!r}")
        raw_values = _validate_mapping(sample.get("values"), "parameter values")
        if not raw_values:
            raise ValueError(f"empty parameter values for draw {draw_key!r}")
        normalized: dict[str, float] = {}
        for raw_name, raw_val in raw_values.items():
            param_name = _validate_string(raw_name, "parameter_id")
            normalized[param_name] = _as_finite_float(raw_val, param_name)
        names = tuple(sorted(normalized))
        if param_names is None:
            param_names = names
        elif names != param_names:
            raise ValueError(f"inconsistent parameter schema for draw {draw_key!r}")
        by_draw[draw_key] = normalized
    missing_draws = sorted(expected_set - set(by_draw))
    if missing_draws:
        raise ValueError(f"missing parameter draw {missing_draws[0]!r}")
    return param_names, by_draw


def _chunked_table(
    columns: dict[str, tuple[pa.DataType, list[object]]],
    *,
    chunk_size: int | None,
) -> pa.Table:
    """Build an Arrow table cleanly with optional explicit chunk boundaries."""
    fields = [
        pa.field(name, arrow_type, nullable=False)
        for name, (arrow_type, _) in columns.items()
    ]
    schema = pa.schema(fields)
    if chunk_size is None:
        arrays = [
            pa.array(values, type=arrow_type) for arrow_type, values in columns.values()
        ]
        return pa.Table.from_arrays(arrays, schema=schema)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunked_arrays = []
    for arrow_type, values in columns.values():
        slices = [
            pa.array(values[start : start + chunk_size], type=arrow_type)
            for start in range(0, len(values), chunk_size)
        ]
        chunked_arrays.append(pa.chunked_array(slices, type=arrow_type))
    return pa.Table.from_arrays(chunked_arrays, schema=schema)


def simulation_outcome_table(
    profile: Mapping[str, object],
    *,
    chunk_size: int | None = None,
) -> pa.Table:
    """Build an Arrow table preserving `voiage.simulation-outcome.v1` semantics."""
    if profile.get("profile_version") != SIMULATION_OUTCOME_PROFILE_VERSION_V1:
        raise ValueError(
            f"unsupported simulation outcome profile_version: {profile.get('profile_version')!r}"
        )
    provenance = SimulationOutcomeProvenance.model_validate(profile.get("provenance"))
    draw_namespace = _validate_string(profile.get("draw_namespace"), "draw_namespace")
    raw_strategies = profile.get("strategy_ids")
    if not isinstance(raw_strategies, Sequence) or isinstance(
        raw_strategies, str | bytes
    ):
        raise TypeError("strategy_ids must be a sequence")
    strategy_ids = tuple(
        _validate_string(item, "strategy_ids") for item in raw_strategies
    )
    value_declaration = SimulationOutcomeValueDeclaration.model_validate(
        profile.get("value_declaration")
    )
    weight_semantics = _validate_weight_semantics(profile.get("weight_semantics"))

    outcome_rows, draw_keys, _ = _validate_outcome_rows(
        profile.get("outcomes"),
        draw_namespace=draw_namespace,
        strategy_ids=strategy_ids,
        profile_decl=value_declaration,
        weight_semantics=weight_semantics,
    )
    param_names, param_by_draw = _validate_parameter_samples(
        profile.get("parameter_samples"),
        draw_namespace=draw_namespace,
        expected_draw_keys=draw_keys,
    )
    metadata = SimulationOutcomeProfileMetadata(
        profile_version=SIMULATION_OUTCOME_PROFILE_VERSION_V1,
        provenance=provenance,
        draw_namespace=draw_namespace,
        strategy_ids=strategy_ids,
        value_declaration=value_declaration,
        weight_semantics=weight_semantics,
        draw_keys=draw_keys,
        parameter_names=param_names,
    )

    columns: dict[str, tuple[pa.DataType, list[object]]] = {
        "draw_namespace": (
            pa.string(),
            [row["draw_namespace"] for row in outcome_rows],
        ),
        "draw_key": (pa.string(), [row["draw_key"] for row in outcome_rows]),
        "strategy_id": (pa.string(), [row["strategy_id"] for row in outcome_rows]),
        "value": (pa.float64(), [row["value"] for row in outcome_rows]),
        "weight": (pa.float64(), [row["weight"] for row in outcome_rows]),
    }
    if param_names is not None:
        for param_name in param_names:
            columns[f"{_PARAM_COLUMN_PREFIX}{param_name}"] = (
                pa.float64(),
                [
                    param_by_draw[cast("str", row["draw_key"])][param_name]
                    for row in outcome_rows
                ],
            )

    table = _chunked_table(columns, chunk_size=chunk_size)
    fingerprint = schema_fingerprint(table.schema)
    metadata_json = json.dumps(
        metadata.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return table.replace_schema_metadata(
        {
            _OUTCOME_CONTRACT_KEY: b"simulation-outcome",
            _OUTCOME_VERSION_KEY: SIMULATION_OUTCOME_PROFILE_VERSION_V1.encode("ascii"),
            _OUTCOME_METADATA_KEY: metadata_json.encode("utf-8"),
            _OUTCOME_FINGERPRINT_KEY: fingerprint.encode("ascii"),
            _OUTCOME_INTERCHANGE_KEY: b"apache-arrow",
        }
    )


def parse_simulation_outcome_table(table: pa.Table) -> dict[str, object]:
    """Validate and restore a `voiage.simulation-outcome.v1` profile from Arrow."""
    schema_meta = table.schema.metadata or {}
    for key in _REQUIRED_OUTCOME_META_KEYS:
        if key not in schema_meta:
            raise ValueError(
                f"missing required simulation-outcome metadata key: {key.decode('ascii', errors='replace')}"
            )
    if schema_meta[_OUTCOME_CONTRACT_KEY] != b"simulation-outcome":
        raise ValueError("invalid simulation-outcome contract marker")
    version_bytes = schema_meta[_OUTCOME_VERSION_KEY]
    if (
        version_bytes.decode("utf-8", errors="strict")
        != SIMULATION_OUTCOME_PROFILE_VERSION_V1
    ):
        raise ValueError(
            "unsupported simulation-outcome profile_version in Arrow metadata"
        )
    expected_fingerprint = schema_meta[_OUTCOME_FINGERPRINT_KEY].decode(
        "ascii", errors="strict"
    )
    if schema_fingerprint(table.schema) != expected_fingerprint:
        raise ValueError("simulation-outcome Arrow schema fingerprint mismatch")

    try:
        profile_meta = SimulationOutcomeProfileMetadata.model_validate_json(
            schema_meta[_OUTCOME_METADATA_KEY]
        )
    except Exception as error:
        raise ValueError("invalid simulation-outcome metadata header") from error

    expected_columns: list[str] = list(_CORE_OUTCOME_COLUMNS)
    if profile_meta.parameter_names is not None:
        expected_columns.extend(
            f"{_PARAM_COLUMN_PREFIX}{name}" for name in profile_meta.parameter_names
        )
    if table.column_names != expected_columns:
        raise ValueError(
            "simulation-outcome table columns do not match metadata schema"
        )
    string_columns = {"draw_namespace", "draw_key", "strategy_id"}
    for field in table.schema:
        if field.name in string_columns:
            if not (
                pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
            ):
                raise ValueError(
                    f"simulation-outcome column {field.name!r} must have string Arrow type, got {field.type}"
                )
        elif not pa.types.is_float64(field.type):
            raise ValueError(
                f"simulation-outcome column {field.name!r} must have float64 Arrow type, got {field.type}"
            )

    raw_rows = table.to_pylist()
    reconstructed_outcomes = [
        {
            "draw_namespace": row["draw_namespace"],
            "draw_key": row["draw_key"],
            "strategy_id": row["strategy_id"],
            "value": row["value"],
            "weight": row["weight"],
        }
        for row in raw_rows
    ]
    reconstructed_params: list[dict[str, object]] | None = None
    if profile_meta.parameter_names is not None:
        param_by_draw: dict[str, dict[str, float]] = {}
        for row in raw_rows:
            draw_key = _validate_string(row.get("draw_key"), "draw_key")
            values = {
                name: _as_finite_float(row.get(f"{_PARAM_COLUMN_PREFIX}{name}"), name)
                for name in profile_meta.parameter_names
            }
            existing_params = param_by_draw.get(draw_key)
            if existing_params is None:
                param_by_draw[draw_key] = values
            elif existing_params != values:
                raise ValueError(
                    f"inconsistent repeated parameter values for draw_key {draw_key!r}"
                )
        reconstructed_params = [
            {
                "draw_namespace": profile_meta.draw_namespace,
                "draw_key": draw_key,
                "values": values,
            }
            for draw_key, values in sorted(param_by_draw.items())
        ]

    validated_rows, draw_keys, _ = _validate_outcome_rows(
        reconstructed_outcomes,
        draw_namespace=profile_meta.draw_namespace,
        strategy_ids=profile_meta.strategy_ids,
        profile_decl=profile_meta.value_declaration,
        weight_semantics=profile_meta.weight_semantics,
    )
    if draw_keys != profile_meta.draw_keys:
        raise ValueError("simulation-outcome draw_keys do not match metadata header")
    _ = _validate_parameter_samples(
        reconstructed_params,
        draw_namespace=profile_meta.draw_namespace,
        expected_draw_keys=draw_keys,
    )

    strategy_rank = {
        strategy: idx for idx, strategy in enumerate(profile_meta.strategy_ids)
    }
    validated_rows.sort(
        key=lambda row: (
            cast("str", row["draw_key"]),
            strategy_rank[cast("str", row["strategy_id"])],
        )
    )
    return {
        "profile_version": profile_meta.profile_version,
        "provenance": profile_meta.provenance.model_dump(
            mode="json", exclude_none=True
        ),
        "draw_namespace": profile_meta.draw_namespace,
        "strategy_ids": list(profile_meta.strategy_ids),
        "value_declaration": profile_meta.value_declaration.model_dump(
            mode="json", exclude_none=True
        ),
        "weight_semantics": profile_meta.weight_semantics,
        "outcomes": validated_rows,
        "parameter_samples": reconstructed_params,
    }


def write_simulation_outcome_ipc(
    profile: Mapping[str, object],
    path: Path,
    *,
    chunk_size: int | None = None,
) -> None:
    """Write a deterministic Arrow IPC file for `voiage.simulation-outcome.v1`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _with_interchange_marker(
        simulation_outcome_table(profile, chunk_size=chunk_size), b"apache-arrow-ipc"
    )
    with ipc.new_file(path, table.schema) as writer:
        if chunk_size is None:
            writer.write_table(table)
        else:
            for batch in table.to_batches(max_chunksize=chunk_size):
                writer.write_batch(batch)


def read_simulation_outcome_ipc(path: Path) -> dict[str, object]:
    """Read and validate a `voiage.simulation-outcome.v1` Arrow IPC file."""
    with ipc.open_file(path) as reader:
        table = reader.read_all()
    meta = table.schema.metadata or {}
    if meta.get(_OUTCOME_INTERCHANGE_KEY) != b"apache-arrow-ipc":
        raise ValueError("not a voiage simulation-outcome Arrow IPC file")
    return parse_simulation_outcome_table(table)


def write_simulation_outcome_parquet(
    profile: Mapping[str, object],
    path: Path,
    *,
    chunk_size: int | None = None,
) -> None:
    """Write a Zstandard-compressed Parquet file for `voiage.simulation-outcome.v1`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _with_interchange_marker(
        simulation_outcome_table(profile, chunk_size=chunk_size),
        b"apache-arrow-parquet",
    )
    pq.write_table(table, path, compression="zstd", row_group_size=chunk_size)


def read_simulation_outcome_parquet(path: Path) -> dict[str, object]:
    """Read and validate a `voiage.simulation-outcome.v1` Parquet file."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    if meta.get(_OUTCOME_INTERCHANGE_KEY) != b"apache-arrow-parquet":
        raise ValueError("not a voiage simulation-outcome Parquet file")
    return parse_simulation_outcome_table(table)
