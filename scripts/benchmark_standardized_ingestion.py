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
import tempfile
import time
import tracemalloc
from typing import TypedDict

import pyarrow as pa
import pyarrow.ipc as paipc
import pyarrow.parquet as pq

from voiage.contracts import NormalizedInputBundle, VOIBinding, prepare_analysis_inputs
from voiage.ingestion import SourceAccessPolicy, default_registry
from voiage.methods.basic import evpi

# This fixture-sized ceiling catches accidental whole-file duplication without
# making a timing claim about heterogeneous local or hosted runners.
CANONICAL_PEAK_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024
BOUNDED_FIXTURE_MAX_ROWS = 3
BOUNDED_FIXTURE_MAX_RESOURCE_BYTES = 64 * 1024


class BenchmarkMetrics(TypedDict):
    """Local fixture measurements plus reproducible workload dimensions."""

    parse_to_arrow_ms: float
    normalize_ms: float
    evpi_ms: float
    peak_bytes: float
    input_bytes: int
    row_count: int


def _median_ms(values: list[float]) -> float:
    """Return the deterministic upper median in milliseconds."""
    return sorted(values)[len(values) // 2] * 1000


def measure(descriptor: Path, *, repeats: int = 5) -> BenchmarkMetrics:
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
    resource = bundle.manifest.resources[0]
    row_count = bundle.table("samples").num_rows
    if resource.byte_size is None:
        raise AssertionError("benchmark fixture must retain a byte receipt")
    if row_count > BOUNDED_FIXTURE_MAX_ROWS:
        raise AssertionError("benchmark fixture exceeds its declared row bound")
    if resource.byte_size > BOUNDED_FIXTURE_MAX_RESOURCE_BYTES:
        raise AssertionError("benchmark fixture exceeds its declared byte bound")
    return {
        "parse_to_arrow_ms": _median_ms(parse),
        "normalize_ms": _median_ms(prepare),
        "evpi_ms": _median_ms(calculate),
        "peak_bytes": float(sorted(peak_bytes)[len(peak_bytes) // 2]),
        "input_bytes": resource.byte_size,
        "row_count": row_count,
    }


def measure_canonical_suite(*, repeats: int = 5) -> dict[str, BenchmarkMetrics]:
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


def _write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON fixture metadata without source-side inference."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_strict_local_fixture_suite(root: Path) -> dict[str, Path]:
    """Create bounded local provider fixtures for a comparable format matrix.

    These fixtures exercise only the repository's strict local profiles.  They
    deliberately omit remote transport, archives, transforms, and external
    parser configuration, so their measurements are not claims about upstream
    libraries or networked catalogues.
    """
    root.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"strategy_a": [10.0, 30.0, 20.0], "strategy_b": [20.0, 10.0, 25.0]}
    )
    csv_text = "strategy_a,strategy_b\n10.0,20.0\n30.0,10.0\n20.0,25.0\n"
    (root / "samples.csv").write_text(csv_text, encoding="utf-8")
    _write_json(
        root / "samples.json",
        [
            {"strategy_a": 10.0, "strategy_b": 20.0},
            {"strategy_a": 30.0, "strategy_b": 10.0},
            {"strategy_a": 20.0, "strategy_b": 25.0},
        ],
    )
    pq.write_table(table, root / "samples.parquet")
    with (
        pa.OSFile(str(root / "samples.arrow"), "wb") as sink,
        paipc.new_file(sink, table.schema) as writer,
    ):
        writer.write_table(table)

    fields = [
        {"name": "strategy_a", "type": "number"},
        {"name": "strategy_b", "type": "number"},
    ]
    _write_json(
        root / "croissant-csv.croissant.json",
        {
            "@context": "https://mlcommons.org/croissant/1.1",
            "name": "bounded-local-croissant-csv",
            "distribution": [{"contentUrl": "samples.csv"}],
            "recordSet": [
                {
                    "name": "samples",
                    "field": [{"name": field["name"]} for field in fields],
                }
            ],
        },
    )
    descriptors: dict[str, Path] = {
        "croissant_csv": root / "croissant-csv.croissant.json"
    }
    for name, path, format_name in (
        ("frictionless_csv", "samples.csv", "csv"),
        ("frictionless_json", "samples.json", "json"),
        ("frictionless_parquet", "samples.parquet", "parquet"),
        ("frictionless_arrow", "samples.arrow", "arrow"),
    ):
        descriptor = root / f"{name}.datapackage.json"
        resource: dict[str, object] = {
            "name": "samples",
            "path": path,
            "schema": {"fields": fields},
        }
        if format_name != "csv":
            resource["format"] = format_name
        _write_json(
            descriptor, {"name": f"bounded-local-{name}", "resources": [resource]}
        )
        descriptors[name] = descriptor
    return descriptors


def measure_strict_local_suite(*, repeats: int = 5) -> dict[str, BenchmarkMetrics]:
    """Measure bounded CSV/JSON/Parquet/Arrow local-provider paths only."""
    with tempfile.TemporaryDirectory(prefix="voiage-ingestion-benchmark-") as directory:
        descriptors = write_strict_local_fixture_suite(Path(directory))
        return {
            name: measure(descriptor, repeats=repeats)
            for name, descriptor in descriptors.items()
        }


def assert_memory_budget(results: dict[str, BenchmarkMetrics]) -> None:
    """Reject a canonical fixture result that exceeds the release memory budget."""
    for provider, metrics in results.items():
        if metrics["peak_bytes"] > CANONICAL_PEAK_MEMORY_BUDGET_BYTES:
            raise AssertionError(
                f"{provider} peak memory exceeds the canonical fixture budget"
            )


if __name__ == "__main__":
    suite = measure_strict_local_suite()
    assert_memory_budget(suite)
    print(json.dumps(suite, sort_keys=True))
