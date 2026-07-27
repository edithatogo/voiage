"""Measure canonical standardized-ingestion stages without setting a CI budget.

The benchmark deliberately emits measurements instead of enforcing wall-clock
budgets: GitHub-hosted runners are not a stable performance reference.  The
canonical Croissant and Frictionless fixtures nevertheless give release review
a comparable parsing-to-Arrow, normalization, calculation, and peak-memory
record on a chosen runner.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import tracemalloc

from voiage.contracts import NormalizedInputBundle, VOIBinding, prepare_analysis_inputs
from voiage.ingestion import SourceAccessPolicy, default_registry
from voiage.methods.basic import evpi

# This fixture-sized ceiling catches accidental whole-file duplication without
# making a timing claim about heterogeneous local or hosted runners.
CANONICAL_PEAK_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024


def _median_ms(values: list[float]) -> float:
    """Return the deterministic upper median in milliseconds."""
    return sorted(values)[len(values) // 2] * 1000


def measure(descriptor: Path, *, repeats: int = 5) -> dict[str, float]:
    """Return median parse-to-Arrow, normalization, EVPI, and peak memory."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    policy = SourceAccessPolicy(descriptor.parent)
    parse: list[float] = []
    prepare: list[float] = []
    calculate: list[float] = []
    peak_bytes: list[float] = []
    for _ in range(repeats):
        tracemalloc.start()
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
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_bytes.append(float(peak))
    return {
        "parse_to_arrow_ms": _median_ms(parse),
        "normalize_ms": _median_ms(prepare),
        "evpi_ms": _median_ms(calculate),
        "peak_bytes": float(sorted(peak_bytes)[len(peak_bytes) // 2]),
    }


def measure_canonical_suite(*, repeats: int = 5) -> dict[str, dict[str, float]]:
    """Measure both built-in provider fixtures with one comparable protocol."""
    root = Path(__file__).parents[1] / "tests" / "fixtures" / "standardized_ingestion"
    return {
        "croissant": measure(
            root / "canonical-decision.croissant.json", repeats=repeats
        ),
        "frictionless": measure(
            root / "canonical-decision.datapackage.json", repeats=repeats
        ),
    }


def assert_memory_budget(results: dict[str, dict[str, float]]) -> None:
    """Reject a canonical fixture result that exceeds the release memory budget."""
    for provider, metrics in results.items():
        if metrics["peak_bytes"] > CANONICAL_PEAK_MEMORY_BUDGET_BYTES:
            raise AssertionError(
                f"{provider} peak memory exceeds the canonical fixture budget"
            )


if __name__ == "__main__":
    suite = measure_canonical_suite()
    assert_memory_budget(suite)
    print(json.dumps(suite, sort_keys=True))
