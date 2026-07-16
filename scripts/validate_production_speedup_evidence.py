"""Validate production-scale benchmark packets and emit a deterministic index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED = {
    "backend",
    "status",
    "owner",
    "source_command",
    "timestamp",
    "workload_hash",
    "workload_scale",
    "warmup_iterations",
    "timing_seconds",
    "throughput",
    "device_metadata",
    "cpu_fallback",
    "cpu_comparison_seconds",
    "speedup",
    "review_status",
}
BACKENDS = {"cpu", "gpu", "tpu", "metal", "fpga", "asic"}
STATUSES = {"passed", "blocked", "not_available", "failed"}
REVIEW_STATUSES = {"reviewed", "pending", "not_applicable"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def validate_manifest(path: Path) -> dict[str, Any]:
    """Validate packets and return a stable SHA-256 index."""
    manifest = _load(path)
    packets = manifest.get("packets")
    if not isinstance(packets, list) or not packets:
        raise ValueError("manifest.packets must be a non-empty list")
    root = path.parent.resolve()
    index: list[dict[str, Any]] = []
    for entry in packets:
        if not isinstance(entry, dict):
            raise TypeError("manifest packet entries must be objects")
        artifact = entry.get("artifact_path")
        if not isinstance(artifact, str) or not artifact:
            raise ValueError("packet artifact_path must be a non-empty string")
        packet_path = (root / artifact).resolve()
        try:
            packet_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact escapes manifest root: {artifact}") from exc
        if not packet_path.is_file():
            raise ValueError(f"missing packet artifact: {artifact}")
        packet = _load(packet_path)
        missing = sorted(REQUIRED - packet.keys())
        if missing:
            raise ValueError(f"{artifact}: missing fields {missing}")
        if packet["backend"] not in BACKENDS:
            raise ValueError(f"{artifact}: unsupported backend {packet['backend']!r}")
        if packet["status"] not in STATUSES:
            raise ValueError(f"{artifact}: unsupported status {packet['status']!r}")
        if not isinstance(packet["owner"], str) or not packet["owner"].strip():
            raise ValueError(f"{artifact}: owner must be non-empty")
        if not isinstance(packet["device_metadata"], dict) or not packet["device_metadata"]:
            raise ValueError(f"{artifact}: device_metadata must be non-empty")
        if not isinstance(packet["workload_hash"], str) or len(packet["workload_hash"]) < 8:
            raise ValueError(f"{artifact}: workload_hash must identify the workload")
        if packet["workload_scale"] not in {"production", "representative", "smoke"}:
            raise ValueError(f"{artifact}: workload_scale must be classified")
        if not isinstance(packet["warmup_iterations"], int) or packet["warmup_iterations"] < 0:
            raise ValueError(f"{artifact}: warmup_iterations must be non-negative")
        if packet["review_status"] not in REVIEW_STATUSES:
            raise ValueError(f"{artifact}: invalid review_status")
        if packet["status"] == "passed":
            if packet["workload_scale"] != "production":
                raise ValueError(f"{artifact}: passed packets require production workload_scale")
            if packet["timing_seconds"] <= 0 or packet["throughput"] <= 0:
                raise ValueError(f"{artifact}: passed packets require positive measurements")
            if packet["cpu_comparison_seconds"] <= 0:
                raise ValueError(f"{artifact}: passed packets require CPU comparison")
            if packet["backend"] != "cpu" and packet["speedup"] <= 1:
                raise ValueError(f"{artifact}: accelerator packets require speedup > 1")
            if packet["backend"] != "cpu" and packet["review_status"] != "reviewed":
                raise ValueError(f"{artifact}: passed packets require reviewed status")
        elif not packet.get("blocked_reason"):
            raise ValueError(f"{artifact}: non-passed packets require blocked_reason")
        index.append(
            {
                "backend": packet["backend"],
                "status": packet["status"],
                "artifact_path": artifact,
                "sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            }
        )
    return {"schema_version": manifest.get("schema_version", "v1"), "packets": index}


def main() -> int:
    """Validate a manifest and optionally write its deterministic index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(validate_manifest(args.manifest), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
