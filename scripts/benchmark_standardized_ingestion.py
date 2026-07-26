"""Measure canonical standardized-ingestion stages without setting a CI budget."""

from __future__ import annotations

import json
from pathlib import Path
import time

from voiage.contracts import NormalizedInputBundle, VOIBinding, prepare_analysis_inputs
from voiage.ingestion import SourceAccessPolicy, default_registry
from voiage.methods.basic import evpi


def _median_ms(values: list[float]) -> float:
    """Return the deterministic upper median in milliseconds."""
    return sorted(values)[len(values) // 2] * 1000


def measure(descriptor: Path, *, repeats: int = 5) -> dict[str, float]:
    """Return median parse, preparation, and EVPI timings in milliseconds."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    policy = SourceAccessPolicy(descriptor.parent)
    parse: list[float] = []
    prepare: list[float] = []
    calculate: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        bundle = default_registry().ingest(descriptor, policy=policy)
        parse.append(time.perf_counter() - started)
        binding = VOIBinding(
            role="net_benefit",
            table_id="samples",
            field_ids=("strategy_a", "strategy_b"),
            strategy_names=("A", "B"),
        )
        bundle = NormalizedInputBundle(
            manifest=bundle.manifest.model_copy(update={"bindings": (binding,)}),
            tables=bundle.tables,
        )
        started = time.perf_counter()
        prepared = prepare_analysis_inputs(bundle)
        prepare.append(time.perf_counter() - started)
        started = time.perf_counter()
        evpi(prepared.net_benefits)
        calculate.append(time.perf_counter() - started)
    return {
        "parse_ms": _median_ms(parse),
        "prepare_ms": _median_ms(prepare),
        "evpi_ms": _median_ms(calculate),
    }


if __name__ == "__main__":
    root = Path(__file__).parents[1]
    descriptor = (
        root
        / "tests"
        / "fixtures"
        / "standardized_ingestion"
        / "canonical-decision.croissant.json"
    )
    print(json.dumps(measure(descriptor), sort_keys=True))
