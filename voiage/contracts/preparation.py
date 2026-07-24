"""Explicit conversion of normalized tables into existing VOI runtime inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from voiage.contracts.normalized_input import (  # noqa: TC001 - public runtime API
    NormalizedInputBundle,
    VOIBinding,
)
from voiage.schema import ValueArray


@dataclass(frozen=True)
class PreparedAnalysisInputs:
    """Existing runtime values with their normalized-data provenance attached."""

    net_benefits: ValueArray
    input_digest: str
    binding: VOIBinding


def prepare_analysis_inputs(bundle: NormalizedInputBundle) -> PreparedAnalysisInputs:
    """Prepare a wide net-benefit binding without implicit filtering or coercion."""
    bindings = tuple(
        item for item in bundle.manifest.bindings if item.role == "net_benefit"
    )
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
    return PreparedAnalysisInputs(
        net_benefits=ValueArray.from_numpy(values, strategies),
        input_digest=bundle.content_digest,
        binding=binding,
    )
