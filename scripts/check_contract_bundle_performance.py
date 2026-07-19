"""Measure and enforce the pinned VOP-VOIAGE bundle verification budget."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import process_time
import tracemalloc
from typing import Any

import pyarrow

from voiage.contracts.bundle import (
    BundleVerificationError,
    ContractPerformanceBudget,
    verify_pinned_contract_bundle,
)

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLES = _ROOT / "specs" / "integration" / "vop-voiage" / "bundles"


def _read_budget(path: Path) -> tuple[int, ContractPerformanceBudget]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    repetitions = raw["verification_repetitions"]
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 1
    ):
        raise ValueError("verification_repetitions must be a positive integer")
    return repetitions, ContractPerformanceBudget(
        cpu_seconds=float(raw["cpu_seconds"]),
        peak_memory_bytes=int(raw["peak_memory_bytes"]),
        allocation_count=int(raw["allocation_count"]),
        serialization_bytes=int(raw["serialization_bytes"]),
    )


def measure_contract_bundle(
    *, bundle: Path, pin: Path, budget_path: Path
) -> dict[str, Any]:
    """Return durable measurements after enforcing the checked-in budget."""
    repetitions, budget = _read_budget(budget_path)
    serialization_bytes = sum(
        path.stat().st_size for path in bundle.rglob("*") if path.is_file()
    )
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    start = process_time()
    verified = None
    for _ in range(repetitions):
        verified = verify_pinned_contract_bundle(bundle, pin)
    cpu_seconds = process_time() - start
    after = tracemalloc.take_snapshot()
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    allocation_count = max(
        0, sum(stat.count_diff for stat in after.compare_to(before, "lineno"))
    )
    tracemalloc.stop()
    assert verified is not None
    evidence: dict[str, Any] = {
        "schema_version": "1.0.0",
        "bundle_sha256": verified.bundle_sha256,
        "arrow_schema_fingerprint": verified.arrow_schema_fingerprint,
        "verification_repetitions": repetitions,
        "measurements": {
            "cpu_seconds": cpu_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "allocation_count": allocation_count,
            "serialization_bytes": serialization_bytes,
        },
        "budget": asdict(budget),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pyarrow": pyarrow.__version__,
        },
    }
    try:
        budget.enforce(
            cpu_seconds=cpu_seconds,
            peak_memory_bytes=peak_memory_bytes,
            allocation_count=allocation_count,
            serialization_bytes=serialization_bytes,
        )
    except BundleVerificationError as exc:
        evidence.update(status="fail", violation=str(exc))
    else:
        evidence["status"] = "pass"
    return evidence


def main() -> int:
    """Run the canonical bundle performance gate and optionally retain evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=_BUNDLES / "1.0.0")
    parser.add_argument("--pin", type=Path, default=_BUNDLES / "UPSTREAM.json")
    parser.add_argument(
        "--budget", type=Path, default=_BUNDLES / "performance-budget.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = measure_contract_bundle(
        bundle=args.bundle, pin=args.pin, budget_path=args.budget
    )
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if evidence["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
