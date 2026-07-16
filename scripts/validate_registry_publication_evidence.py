"""Validate the repository's external registry publication evidence packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED = {
    "channel",
    "package",
    "registry",
    "status",
    "owner",
    "next_action",
    "external_gate",
    "evidence_url",
    "checked_at",
}
STATUSES = {
    "readiness",
    "submitted",
    "published",
    "indexed",
    "approved",
    "blocked",
    "not-found",
    "not_present",
    "not_found",
    "unconfirmed",
    "no_released_versions",
    "external_manual",
}


def validate_manifest(path: Path) -> dict[str, Any]:
    """Validate every registry channel and return a stable summary."""
    value = json.loads(path.read_text(encoding="utf-8"))
    channels = value.get("channels") if isinstance(value, dict) else None
    if not isinstance(channels, list) or len(channels) < 13:
        raise ValueError("channels must contain all 13 registry targets")
    seen: set[str] = set()
    for channel in channels:
        if not isinstance(channel, dict):
            raise TypeError("registry channels must be objects")
        missing = sorted(REQUIRED - channel.keys())
        if missing:
            raise ValueError(f"channel missing fields {missing}")
        name = channel["channel"]
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("channel names must be unique and non-empty")
        seen.add(name)
        if channel["status"] not in STATUSES:
            raise ValueError(f"{name}: unsupported status")
        for field in (
            "owner",
            "next_action",
            "external_gate",
            "evidence_url",
            "checked_at",
        ):
            if not isinstance(channel[field], str) or not channel[field].strip():
                raise ValueError(f"{name}: {field} must be non-empty")
        if channel["status"] in {
            "blocked",
            "not-found",
            "not_found",
            "not_present",
            "unconfirmed",
            "no_released_versions",
            "external_manual",
        } and channel["external_gate"].strip().lower() in {"none", "n/a"}:
            raise ValueError(f"{name}: unresolved channel requires an external gate")
    return {
        "schema_version": value.get("schema_version", "v1"),
        "channel_count": len(channels),
        "channels": sorted(seen),
    }


def main() -> int:
    """Validate a registry evidence manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_manifest(args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
