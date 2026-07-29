"""Validate or deterministically refresh standardized-ingestion fixtures.

The source descriptors and CSV resources are deliberately kept as small,
human-reviewable fixture corpora.  Their companion manifests pin every source
artifact, so an intentional fixture update must be explicit: run this script
with ``--write`` and review the resulting digest change.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

DEFAULT_FIXTURE_ROOT = (
    Path(__file__).parents[1] / "tests" / "fixtures" / "standardized_ingestion"
)


def _source_files(manifest_path: Path) -> dict[str, str]:
    """Return deterministic SHA-256 digests for one fixture's source files."""
    stem = manifest_path.name.removesuffix(".manifest.json")
    return {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(manifest_path.parent.glob(f"{stem}.*"))
        if path != manifest_path
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read and minimally validate a fixture manifest before comparing it."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: fixture manifest must be a JSON object")
    if payload.get("schema_version") != "1.0.0":
        raise ValueError(f"{path}: unsupported fixture schema_version")
    if not isinstance(payload.get("dataset_id"), str) or not payload["dataset_id"]:
        raise ValueError(f"{path}: fixture manifest requires a dataset_id")
    return payload


def validate_fixture_manifest(path: Path, *, write: bool = False) -> bool:
    """Validate one manifest, or refresh only its generated ``files`` mapping.

    Returns ``True`` when the checked-in manifest already matches the source
    artifacts.  ``--write`` is deliberately limited to the digest mapping:
    dataset and binding semantics always remain review-owned metadata.
    """
    payload = _read_manifest(path)
    expected = _source_files(path)
    if payload.get("files") == expected:
        return True
    if write:
        payload["files"] = expected
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return False
    return False


def fixture_manifests(root: Path) -> tuple[Path, ...]:
    """Find fixture manifests in stable order and reject an empty corpus."""
    manifests = tuple(sorted(root.glob("*.manifest.json")))
    if not manifests:
        raise ValueError(f"{root}: no standardized-ingestion fixture manifests")
    return manifests


def main(argv: list[str] | None = None) -> int:
    """Run deterministic digest validation for the complete fixture corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument(
        "--write",
        action="store_true",
        help="refresh only stale generated digest mappings",
    )
    args = parser.parse_args(argv)
    stale = [
        path
        for path in fixture_manifests(args.root)
        if not validate_fixture_manifest(path, write=args.write)
    ]
    if stale and not args.write:
        for path in stale:
            print(f"stale fixture manifest: {path}")
        return 1
    if stale:
        for path in stale:
            print(f"refreshed fixture manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
