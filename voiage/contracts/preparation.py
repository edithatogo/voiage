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
from voiage.schema import ParameterSet, ValueArray

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
class MethodInputCapability:
    """Declared normalized-input requirements for one supported method family."""

    method_family: str
    required_binding_roles: tuple[str, ...]
    accepted_input_kinds: tuple[str, ...]
    requires_sample_alignment: bool
    allows_implicit_transforms: bool = False


_METHOD_INPUT_CAPABILITIES: Mapping[str, MethodInputCapability] = MappingProxyType(
    {
        "evpi": MethodInputCapability(
            method_family="evpi",
            required_binding_roles=("net_benefit",),
            accepted_input_kinds=("direct-python", "csv", "normalized-bundle"),
            requires_sample_alignment=True,
        ),
        "evppi": MethodInputCapability(
            method_family="evppi",
            required_binding_roles=("net_benefit", "parameter"),
            accepted_input_kinds=("direct-python", "csv", "normalized-bundle"),
            requires_sample_alignment=True,
        ),
        "evsi": MethodInputCapability(
            method_family="evsi",
            required_binding_roles=("net_benefit", "parameter"),
            accepted_input_kinds=("direct-python", "csv", "normalized-bundle"),
            requires_sample_alignment=True,
        ),
        "enbs": MethodInputCapability(
            method_family="enbs",
            required_binding_roles=("net_benefit",),
            accepted_input_kinds=("direct-python", "csv", "normalized-bundle"),
            requires_sample_alignment=True,
        ),
        "ceac": MethodInputCapability(
            method_family="ceac",
            required_binding_roles=("net_benefit",),
            accepted_input_kinds=("direct-python", "csv", "normalized-bundle"),
            requires_sample_alignment=True,
        ),
        "ceaf": MethodInputCapability(
            method_family="ceaf",
            required_binding_roles=("net_benefit",),
            accepted_input_kinds=("direct-python", "csv", "normalized-bundle"),
            requires_sample_alignment=True,
        ),
    }
)


def method_input_capability(method_family: str) -> MethodInputCapability:
    """Return the normalized-input contract for a method or fail closed."""
    try:
        return _METHOD_INPUT_CAPABILITIES[method_family]
    except KeyError as error:
        raise ValueError(
            f"no normalized input capability is declared for {method_family!r}"
        ) from error


@dataclass(frozen=True)
class PreparedAnalysisInputs:
    """Existing runtime values with their normalized-data provenance attached."""

    net_benefits: ValueArray
    input_digest: str
    binding_profile_digest: str
    binding: VOIBinding
    quality_report: DataQualityReport
    parameters: ParameterSet | None = None


def _long_net_benefit_values(
    selected: pa.Table, binding: VOIBinding
) -> tuple[np.ndarray, list[str]]:
    """Pivot an explicitly declared long table without changing its population."""
    sample_field = binding.sample_id_field_id
    strategy_field = binding.strategy_field_id
    value_field = binding.value_field_id
    if sample_field is None or strategy_field is None or value_field is None:
        raise ValueError("long net-benefit binding is incomplete")

    expected_strategies = tuple(binding.strategy_names)
    expected_set = set(expected_strategies)
    samples: dict[object, dict[str, float]] = {}
    sample_order: list[object] = []
    for row in selected.to_pylist():
        sample = row[sample_field]
        strategy = row[strategy_field]
        value = row[value_field]
        if sample is None or strategy is None or value is None:
            raise ValueError(
                "long net-benefit rows cannot contain null sample, strategy, or value"
            )
        if not isinstance(strategy, str) or strategy not in expected_set:
            raise ValueError("long net-benefit row names an undeclared strategy")
        if sample not in samples:
            samples[sample] = {}
            sample_order.append(sample)
        if strategy in samples[sample]:
            raise ValueError(
                "long net-benefit rows contain duplicate sample-strategy pairs"
            )
        try:
            samples[sample][strategy] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("long net-benefit value is not numeric") from error

    incomplete = [
        sample for sample in sample_order if set(samples[sample]) != expected_set
    ]
    if incomplete:
        raise ValueError(
            "long net-benefit samples must contain every declared strategy"
        )
    return (
        np.asarray(
            [
                [samples[sample][strategy] for strategy in expected_strategies]
                for sample in sample_order
            ],
            dtype=float,
        ),
        list(expected_strategies),
    )


def _resolve_binding(
    bindings: tuple[VOIBinding, ...], role: str, *, method_family: str
) -> VOIBinding:
    """Resolve a binding role and enforce its declared method applicability."""
    matches = tuple(binding for binding in bindings if binding.role == role)
    if len(matches) != 1:
        raise ValueError(f"exactly one {role} binding is required")
    binding = matches[0]
    if (
        binding.applicable_method_families
        and method_family not in binding.applicable_method_families
    ):
        raise ValueError(
            f"binding role {role!r} is not applicable to {method_family!r}"
        )
    return binding


def _wide_binding_values(table: pa.Table, binding: VOIBinding) -> np.ndarray:
    """Return explicit wide columns only after numeric and null validation."""
    arrays = []
    for field in binding.field_ids:
        column = table[field]
        if column.null_count:
            raise ValueError(f"net-benefit field {field!r} contains nulls")
        try:
            arrays.append(column.combine_chunks().to_numpy(zero_copy_only=False))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
            raise ValueError(f"net-benefit field {field!r} is not numeric") from error
    return np.column_stack(arrays).astype(float, copy=False)


def _prepared_parameters(
    bundle: NormalizedInputBundle,
    bindings: tuple[VOIBinding, ...],
    *,
    method_family: str,
    n_samples: int,
    table_id: str,
) -> ParameterSet | None:
    """Convert one explicit, row-aligned parameter binding without joins."""
    matches = tuple(binding for binding in bindings if binding.role == "parameter")
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("at most one parameter binding is supported")
    binding = _resolve_binding(matches, "parameter", method_family=method_family)
    if binding.layout != "wide" or binding.table_id != table_id:
        raise ValueError(
            "parameter binding must be a wide table aligned with net benefit"
        )
    values = _wide_binding_values(bundle.table(binding.table_id), binding)
    if values.shape[0] != n_samples:
        raise ValueError("parameter samples must align with net-benefit samples")
    names = binding.strategy_names or binding.field_ids
    return ParameterSet.from_numpy_or_dict(
        {name: values[:, index] for index, name in enumerate(names)}
    )


def prepare_analysis_inputs(
    bundle: NormalizedInputBundle,
    *,
    method_family: str = "evpi",
    willingness_to_pay: float | None = None,
) -> PreparedAnalysisInputs:
    """Prepare an explicitly bound wide or long net-benefit table."""
    capability = method_input_capability(method_family)
    binding_profile = getattr(bundle.manifest, "binding_profile", None)
    declared_bindings = (
        binding_profile.bindings
        if binding_profile is not None
        else bundle.manifest.bindings
    )
    net_benefit_bindings = tuple(
        binding for binding in declared_bindings if binding.role == "net_benefit"
    )
    derived = False
    wtp: float | None = None
    if net_benefit_bindings:
        binding = _resolve_binding(
            declared_bindings, "net_benefit", method_family=method_family
        )
    else:
        if willingness_to_pay is None:
            raise ValueError(
                "cost/outcome preparation requires a finite willingness_to_pay"
            )
        wtp = float(willingness_to_pay)
        if not np.isfinite(wtp):
            raise ValueError(
                "cost/outcome preparation requires a finite willingness_to_pay"
            )
        cost_binding = _resolve_binding(
            declared_bindings, "cost", method_family=method_family
        )
        outcome_binding = _resolve_binding(
            declared_bindings, "outcome", method_family=method_family
        )
        if (
            cost_binding.layout != "wide"
            or outcome_binding.layout != "wide"
            or cost_binding.table_id != outcome_binding.table_id
            or len(cost_binding.field_ids) != len(outcome_binding.field_ids)
        ):
            raise ValueError(
                "cost and outcome bindings must be aligned wide fields in one table"
            )
        cost_names = cost_binding.strategy_names or cost_binding.field_ids
        outcome_names = outcome_binding.strategy_names or outcome_binding.field_ids
        if cost_names != outcome_names:
            raise ValueError(
                "cost and outcome bindings must declare the same strategies"
            )
        binding = outcome_binding
        derived = True
    table = bundle.table(binding.table_id)
    selected_field_ids: tuple[str | None, ...] = (
        (
            binding.sample_id_field_id,
            binding.strategy_field_id,
            binding.value_field_id,
        )
        if binding.layout == "long"
        else (
            tuple(cost_binding.field_ids) + tuple(binding.field_ids)
            if derived
            else binding.field_ids
        )
    )
    if any(field is None for field in selected_field_ids):
        raise ValueError("long net-benefit binding is incomplete")
    selected = table.select(selected_field_ids)
    if selected.num_rows == 0:
        raise ValueError("net-benefit input must contain at least one row")
    if derived:
        if wtp is None:
            raise ValueError("cost/outcome preparation requires a willingness_to_pay")
        cost_values = _wide_binding_values(table, cost_binding)
        outcome_values = _wide_binding_values(table, binding)
        values = wtp * outcome_values - cost_values
        strategies = list(binding.strategy_names or binding.field_ids)
    elif binding.layout == "long":
        values, strategies = _long_net_benefit_values(selected, binding)
    else:
        values = _wide_binding_values(table, binding)
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
        selected_field_ids=tuple(
            field for field in selected_field_ids if field is not None
        ),
        row_count=selected.num_rows,
        null_counts=MappingProxyType(
            {
                field: selected[field].null_count
                for field in selected_field_ids
                if field is not None
            }
        ),
        duplicate_row_count=selected.num_rows - len(set(rows)),
        unique_value_counts=MappingProxyType(
            {
                field: len(set(selected[field].to_pylist()))
                for field in selected_field_ids
                if field is not None
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
        population_transforms=(
            ("net_benefit = willingness_to_pay * outcome - cost",) if derived else ()
        ),
    )
    parameters = _prepared_parameters(
        bundle,
        declared_bindings,
        method_family=method_family,
        n_samples=values.shape[0],
        table_id=binding.table_id,
    )
    if "parameter" in capability.required_binding_roles and parameters is None:
        raise ValueError(f"{method_family} requires an explicit parameter binding")
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
        parameters=parameters,
    )
