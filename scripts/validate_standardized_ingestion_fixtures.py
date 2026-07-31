"""Validate or deterministically refresh standardized-ingestion fixtures.

The source descriptors and CSV resources are deliberately kept as small,
human-reviewable fixture corpora.  Their companion manifests pin every source
artifact and the format-neutral normalized identity, so an intentional fixture
update must be explicit: run this script with ``--write`` and review the
resulting digest change.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pyarrow.csv as pacsv

from voiage.contracts import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
    VOIBinding,
)

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


def _bindings(payload: dict[str, Any]) -> tuple[VOIBinding, ...]:
    """Load the singular and plural binding forms used by the fixture corpus."""
    plural = payload.get("bindings")
    if plural is None:
        raw_bindings = [payload.get("binding")]
    elif isinstance(plural, list):
        raw_bindings = plural
    else:
        raise TypeError("fixture bindings must be a list")
    return tuple(
        VOIBinding.model_validate_json(json.dumps(binding)) for binding in raw_bindings
    )


def normalized_identity(path: Path) -> dict[str, str]:
    """Return the reproducible direct normalized identity for one CSV fixture.

    The identity is format-neutral.  It deliberately excludes source-specific
    descriptor URIs and governance metadata, which optional providers preserve
    separately and which cannot be meaningfully identical across formats.
    """
    payload = _read_manifest(path)
    stem = path.name.removesuffix(".manifest.json")
    resource_path = path.parent / f"{stem}.csv"
    table = pacsv.read_csv(resource_path)
    resource_sha256 = sha256(resource_path.read_bytes()).hexdigest()
    bindings = _bindings(payload)
    table_id = bindings[0].table_id
    bundle = NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id=payload["dataset_id"],
            tables=(
                TableManifest(
                    table_id=table_id,
                    fields=tuple(
                        FieldManifest(field_id=field.name, dtype=str(field.type))
                        for field in table.schema
                    ),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="direct-fixture",
                source_uri=f"urn:voiage:fixture:{payload['dataset_id']}",
                descriptor_digest=resource_sha256,
            ),
            bindings=bindings,
        ),
        tables={table_id: table},
    )
    return {
        "content_digest": bundle.content_digest,
        "resource_sha256": resource_sha256,
        "schema_fingerprint": bundle.schema_fingerprint,
    }


def validate_fixture_manifest(path: Path, *, write: bool = False) -> bool:
    """Validate one manifest, or refresh its generated digest mappings.

    Returns ``True`` when the checked-in manifest already matches the source
    artifacts and normalized identity.  ``--write`` is deliberately limited
    to derived digest mappings:
    dataset and binding semantics always remain review-owned metadata.
    """
    payload = _read_manifest(path)
    expected_files = _source_files(path)
    expected_normalized = normalized_identity(path)
    if (
        payload.get("files") == expected_files
        and payload.get("normalized") == expected_normalized
    ):
        return True
    if write:
        payload["files"] = expected_files
        payload["normalized"] = expected_normalized
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
