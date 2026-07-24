"""Explicit conversion of normalized tables into existing VOI runtime inputs."""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from voiage.contracts.normalized_input import (
    BindingProfile,
    NormalizedInputBundle,
    VOIBinding,
)
from voiage.schema import ValueArray

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class DataQualityReport:
    """Machine-readable evidence that preparation retained the input population."""

    table_id: str
    selected_field_ids: tuple[str, ...]
    row_count: int
    null_counts: Mapping[str, int]
    duplicate_row_count: int
    unique_value_counts: Mapping[str, int]
    primary_key_fields: tuple[str, ...]
    primary_key_null_count: int
    primary_key_duplicate_count: int
    join_coverage: Mapping[str, float]
    coercions: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    selected_partitions: tuple[str, ...] = ()
    population_transforms: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedAnalysisInputs:
    """Existing runtime values with their normalized-data provenance attached."""

    net_benefits: ValueArray
    input_digest: str
    binding_profile_digest: str
    binding: VOIBinding
    quality_report: DataQualityReport


def prepare_analysis_inputs(bundle: NormalizedInputBundle) -> PreparedAnalysisInputs:
    """Prepare a wide net-benefit binding without implicit filtering or coercion."""
    binding_profile = getattr(bundle.manifest, "binding_profile", None)
    declared_bindings = (
        binding_profile.bindings
        if binding_profile is not None
        else bundle.manifest.bindings
    )
    bindings = tuple(item for item in declared_bindings if item.role == "net_benefit")
    if len(bindings) != 1:
        raise ValueError("exactly one net_benefit binding is required")
    binding = bindings[0]
    table = bundle.table(binding.table_id)
    selected = table.select(binding.field_ids)
    if selected.num_rows == 0:
        raise ValueError("net-benefit input must contain at least one row")
    arrays = []
    for field in binding.field_ids:
        column = selected[field]
        if column.null_count:
            raise ValueError(f"net-benefit field {field!r} contains nulls")
        try:
            arrays.append(column.combine_chunks().to_numpy(zero_copy_only=False))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
            raise ValueError(f"net-benefit field {field!r} is not numeric") from error
    values = np.column_stack(arrays).astype(float, copy=False)
    strategies = list(binding.strategy_names or binding.field_ids)
    rows = tuple(tuple(row.values()) for row in selected.to_pylist())
    table_manifest = next(
        item for item in bundle.manifest.tables if item.table_id == binding.table_id
    )
    primary_key_rows = tuple(
        tuple(row[field] for field in table_manifest.primary_key)
        for row in table.to_pylist()
    )
    primary_key_null_count = sum(
        any(value is None for value in row) for row in primary_key_rows
    )
    quality_report = DataQualityReport(
        table_id=binding.table_id,
        selected_field_ids=tuple(binding.field_ids),
        row_count=selected.num_rows,
        null_counts=MappingProxyType(
            {field: selected[field].null_count for field in binding.field_ids}
        ),
        duplicate_row_count=selected.num_rows - len(set(rows)),
        unique_value_counts=MappingProxyType(
            {
                field: len(set(selected[field].to_pylist()))
                for field in binding.field_ids
            }
        ),
        primary_key_fields=table_manifest.primary_key,
        primary_key_null_count=primary_key_null_count,
        primary_key_duplicate_count=(
            len(primary_key_rows) - len(set(primary_key_rows))
            if table_manifest.primary_key
            else 0
        ),
        join_coverage=MappingProxyType({}),
    )
    return PreparedAnalysisInputs(
        net_benefits=ValueArray.from_numpy(values, strategies),
        input_digest=bundle.content_digest,
        binding_profile_digest=(
            binding_profile.digest
            if binding_profile is not None
            else BindingProfile(bindings=(binding,)).digest
        ),
        binding=binding,
        quality_report=quality_report,
    )
