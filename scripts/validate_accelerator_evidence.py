"""Validate and index accelerator benchmark evidence packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED = {
    "runtime",
    "status",
    "source_command",
    "timestamp",
    "device_metadata",
    "workload_hash",
    "warmup_iterations",
    "timing_seconds",
    "throughput",
    "cpu_fallback",
}
RUNTIMES = {"gpu", "tpu", "metal", "cpu"}
STATUSES = {"passed", "blocked", "failed"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def validate_manifest(path: Path) -> dict[str, Any]:
    """Validate packet references and return a deterministic index."""
    manifest = _load(path)
    packets = manifest.get("packets")
    if not isinstance(packets, list) or not packets:
        raise ValueError("manifest.packets must be a non-empty list")
    index: list[dict[str, Any]] = []
    root = path.parent.resolve()
    for entry in packets:
        if not isinstance(entry, dict):
            raise TypeError("manifest packet entries must be objects")
        packet_ref = entry.get("artifact_path")
        if not isinstance(packet_ref, str) or not packet_ref:
            raise ValueError("packet artifact_path must be a non-empty string")
        packet_path = (root / packet_ref).resolve()
        try:
            packet_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact escapes manifest root: {packet_ref}") from exc
        if not packet_path.is_file():
            raise ValueError(f"missing packet artifact: {packet_ref}")
        packet = _load(packet_path)
        missing = sorted(REQUIRED - packet.keys())
        if missing:
            raise ValueError(f"{packet_ref}: missing fields {missing}")
        if packet["runtime"] not in RUNTIMES:
            raise ValueError(f"{packet_ref}: unsupported runtime {packet['runtime']!r}")
        if packet["status"] not in STATUSES:
            raise ValueError(f"{packet_ref}: unsupported status {packet['status']!r}")
        if (
            not isinstance(packet["device_metadata"], dict)
            or not packet["device_metadata"]
        ):
            raise ValueError(f"{packet_ref}: device_metadata must be non-empty")
        if (
            not isinstance(packet["workload_hash"], str)
            or len(packet["workload_hash"]) < 8
        ):
            raise ValueError(f"{packet_ref}: workload_hash must identify the workload")
        if packet["warmup_iterations"] < 0:
            raise ValueError(f"{packet_ref}: warmup_iterations must be non-negative")
        if packet["status"] == "passed" and (
            packet["timing_seconds"] is None or packet["throughput"] is None
        ):
            raise ValueError(
                f"{packet_ref}: passed packets require timing and throughput"
            )
        if packet["status"] in {"blocked", "failed"} and not packet.get(
            "blocked_reason"
        ):
            raise ValueError(
                f"{packet_ref}: blocked/failed packets require blocked_reason"
            )
        digest = hashlib.sha256(packet_path.read_bytes()).hexdigest()
        index.append(
            {
                "runtime": packet["runtime"],
                "status": packet["status"],
                "artifact_path": packet_ref,
                "sha256": digest,
            }
        )
    return {"schema_version": manifest.get("schema_version", "v1"), "packets": index}


def main() -> int:
    """Validate a manifest and optionally write its deterministic index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    index = validate_manifest(args.manifest)
    output = json.dumps(index, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
