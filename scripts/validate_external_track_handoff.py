"""Validate a Conductor external-gate handoff packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STATUSES = {
    "ready",
    "submitted",
    "published",
    "indexed",
    "approved",
    "blocked",
    "not_found",
}
REQUIRED = {
    "track_id",
    "channel",
    "status",
    "owner",
    "checked_at",
    "next_action",
    "external_gate",
    "evidence_urls",
    "commands",
}


def validate_handoff(path: Path) -> dict[str, Any]:
    """Validate a handoff and return its stable summary."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("handoff must be a JSON object")
    missing = sorted(REQUIRED - value.keys())
    if missing:
        raise ValueError(f"handoff missing fields {missing}")
    if value["status"] not in STATUSES:
        raise ValueError(f"unsupported handoff status: {value['status']}")
    for field in (
        "track_id",
        "channel",
        "owner",
        "checked_at",
        "next_action",
        "external_gate",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{field} must be non-empty")
    if not isinstance(value["evidence_urls"], list) or not all(
        isinstance(url, str) and url.startswith(("https://", "http://"))
        for url in value["evidence_urls"]
    ):
        raise ValueError("evidence_urls must contain HTTP(S) URLs")
    commands = value["commands"]
    if not isinstance(commands, list) or not commands:
        raise ValueError("commands must be a non-empty list")
    for command in commands:
        if not isinstance(command, dict):
            raise TypeError("command entries must be objects")
        for field in ("command", "runner", "status", "artifact", "details"):
            if not isinstance(command.get(field), str) or not command[field].strip():
                raise ValueError(f"command.{field} must be non-empty")
    if value["status"] in {"blocked", "not_found"} and value[
        "external_gate"
    ].strip().lower() in {"none", "n/a"}:
        raise ValueError("blocked handoffs require an external gate")
    return {
        "track_id": value["track_id"],
        "channel": value["channel"],
        "status": value["status"],
        "command_count": len(commands),
        "evidence_count": len(value["evidence_urls"]),
    }


def main() -> int:
    """Validate a handoff packet from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_handoff(args.handoff), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
