"""Validate CUDA-class GPU production benchmark packets."""

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
    "device_metadata",
    "workload_hash",
    "workload_scale",
    "warmup_iterations",
    "timing_seconds",
    "throughput",
    "cpu_baseline_seconds",
    "speedup",
    "transfer_overhead_seconds",
    "compile_overhead_seconds",
    "result_parity",
    "cpu_fallback",
    "claim_level",
}
STATUSES = {"passed", "blocked", "not_available", "failed"}
CLAIM_LEVELS = {"reference", "blocked", "production_speedup"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def validate_manifest(path: Path) -> dict[str, Any]:
    """Validate packets and return a deterministic SHA-256 index."""
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
        if packet["backend"] not in {"cpu", "gpu"}:
            raise ValueError(f"{artifact}: backend must be cpu or gpu")
        if packet["status"] not in STATUSES:
            raise ValueError(f"{artifact}: unsupported status")
        if packet["claim_level"] not in CLAIM_LEVELS:
            raise ValueError(f"{artifact}: unsupported claim_level")
        if (
            not isinstance(packet["device_metadata"], dict)
            or not packet["device_metadata"]
        ):
            raise ValueError(f"{artifact}: device_metadata must be non-empty")
        if not isinstance(packet["owner"], str) or not packet["owner"].strip():
            raise ValueError(f"{artifact}: owner must be non-empty")
        if (
            not isinstance(packet["workload_hash"], str)
            or len(packet["workload_hash"]) < 8
        ):
            raise ValueError(f"{artifact}: workload_hash must identify workload")
        if packet["workload_scale"] not in {"production", "representative", "smoke"}:
            raise ValueError(f"{artifact}: invalid workload_scale")
        if (
            not isinstance(packet["warmup_iterations"], int)
            or packet["warmup_iterations"] < 0
        ):
            raise ValueError(f"{artifact}: warmup_iterations must be non-negative")
        if packet["status"] == "passed":
            if packet["timing_seconds"] <= 0 or packet["throughput"] <= 0:
                raise ValueError(
                    f"{artifact}: passed packets require positive measurements"
                )
            if packet["cpu_baseline_seconds"] <= 0 or not packet["result_parity"]:
                raise ValueError(
                    f"{artifact}: passed packets require CPU baseline and parity"
                )
            if (
                packet["transfer_overhead_seconds"] < 0
                or packet["compile_overhead_seconds"] < 0
            ):
                raise ValueError(f"{artifact}: overhead values must be non-negative")
            if packet["backend"] == "gpu" and packet["speedup"] <= 1:
                raise ValueError(f"{artifact}: GPU packet requires speedup > 1")
            if (
                packet["claim_level"] == "production_speedup"
                and packet["workload_scale"] != "production"
            ):
                raise ValueError(
                    f"{artifact}: production speedup requires production workload"
                )
        elif not packet.get("blocked_reason"):
            raise ValueError(f"{artifact}: non-passed packets require blocked_reason")
        index.append(
            {
                "backend": packet["backend"],
                "status": packet["status"],
                "claim_level": packet["claim_level"],
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
    output = (
        json.dumps(validate_manifest(args.manifest), indent=2, sort_keys=True) + "\n"
    )
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
