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


def main() -> None:
    """Write once after online authentication, or compare without replacing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = inventory(args.root)
    if args.check:
        if json.loads(args.manifest.read_text(encoding="utf-8")) != records:
            raise ValueError("Source cache changed after authenticated fetch")
    else:
        with args.manifest.open("x", encoding="utf-8") as stream:
            json.dump(records, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
