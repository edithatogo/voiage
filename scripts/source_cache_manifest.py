"""Freeze and check private source caches before network-denied native replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def inventory(root: Path) -> list[dict[str, Any]]:
    """Hash source bytes, allowing only relative mirror aliases inside the root."""
    if root.is_symlink():
        raise ValueError("Symlinked source cache root")
    root = root.resolve(strict=True)
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        link = None
        if path.is_symlink():
            link = path.readlink()
            if link.is_absolute() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("Source cache alias escapes root")
        if not path.is_file():
            raise ValueError("Source cache entry is not a regular file")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(
            {
                "path": str(path.relative_to(root)),
                "link": str(link) if link else None,
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    return records


def validate_replay_cache(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    mirror: list[dict[str, Any]],
) -> None:
    """Preserve frozen entries while allowing only authenticated mirror copies."""
    actual = {entry["path"]: entry for entry in after}
    if any(actual.get(entry["path"]) != entry for entry in before):
        raise ValueError("Frozen source cache entry changed during replay")
    authenticated = {entry["sha256"] for entry in before + mirror}
    if any(entry["sha256"] not in authenticated for entry in after):
        raise ValueError("Source cache contains unauthenticated replay additions")


def main() -> None:
    """Write once after online authentication, or compare without replacing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mirror-manifest", type=Path)
    parser.add_argument("--after-manifest", type=Path)
    args = parser.parse_args()
    if (args.mirror_manifest or args.after_manifest) and not args.check:
        parser.error("Replay manifest options require --check")
    records = inventory(args.root)
    if args.check:
        before = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.mirror_manifest:
            validate_replay_cache(
                before,
                records,
                json.loads(args.mirror_manifest.read_text(encoding="utf-8")),
            )
        elif before != records:
            raise ValueError("Source cache changed after authenticated fetch")
        if args.after_manifest:
            with args.after_manifest.open("x", encoding="utf-8") as stream:
                json.dump(records, stream, indent=2, sort_keys=True)
                stream.write("\n")
    else:
        with args.manifest.open("x", encoding="utf-8") as stream:
            json.dump(records, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
