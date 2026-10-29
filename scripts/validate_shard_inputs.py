"""Validate shard inputs; this is not a coverage aggregate or policy gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def validate(
    manifest: dict[str, object],
    expected_identity: dict[str, object],
    expected_shards: int,
) -> None:
    """Reject incomplete, unsuccessful or inconsistent shard inputs."""
    count = manifest.get("expected_shards")
    identity = manifest.get("identity")
    shards = manifest.get("shards")
    keys = {
        "source_head",
        "lock_digest",
        "interpreter",
        "features",
        "coverage_config_digest",
    }
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("invalid expected shard count")
    if count != expected_shards:
        raise ValueError("aggregate shard count mismatch")
    if (
        not isinstance(identity, dict)
        or set(identity) != keys
        or not all(
            isinstance(value, str) and value.strip() for value in identity.values()
        )
    ):
        raise ValueError("invalid execution identity")
    if identity != expected_identity:
        raise ValueError("aggregate identity mismatch")
    if not isinstance(shards, list) or len(shards) != count:
        raise ValueError("missing or extra shards")
    seen: set[int] = set()
    for shard in shards:
        if not isinstance(shard, dict):
            raise TypeError("invalid shard")
        index = shard.get("index")
        if type(index) is not int or index not in range(count) or index in seen:
            raise ValueError("invalid or duplicate shard index")
        seen.add(index)
        if (
            shard.get("conclusion") != "success"
            or type(shard.get("exit_status")) is not int
            or shard["exit_status"] != 0
        ):
            raise ValueError("shard did not succeed")
        if shard.get("identity") != identity:
            raise ValueError("shard identity mismatch")
        path_value = shard.get("coverage")
        expected = shard.get("coverage_sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise TypeError("missing coverage artifact")
        path = Path(path_value)
        if not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("missing or oversized coverage artifact")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("coverage digest mismatch")
        report = json.loads(data)
        if not isinstance(report, dict) or not isinstance(report.get("files"), dict):
            raise TypeError("invalid coverage report")
        if not report["files"]:
            raise ValueError("empty coverage report")
        executed_count = 0
        for name, item in report["files"].items():
            if not isinstance(name, str) or not name or not isinstance(item, dict):
                raise TypeError("invalid coverage file entry")
            for field in ("executed_lines", "missing_lines"):
                lines = item.get(field)
                if not isinstance(lines, list) or any(
                    type(line) is not int or line < 1 for line in lines
                ):
                    raise ValueError("invalid coverage line numbers")
                if field == "executed_lines":
                    executed_count += len(lines)
        if executed_count == 0:
            raise ValueError("coverage report has no executed lines")


def load_manifest(path: Path) -> dict[str, object]:
    """Read a bounded manifest with a valid top-level shape."""
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("manifest exceeds size limit")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be an object")
    return manifest


def main() -> int:
    """Check the supplied shard aggregation manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()
    try:
        validate(
            load_manifest(Path(args.manifest)),
            load_manifest(Path(args.expected_identity)),
            args.expected_shards,
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"shard aggregation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
