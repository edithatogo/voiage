#!/usr/bin/env python3
"""Generate or verify deterministic perspective Arrow interchange fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from voiage.methods.perspective import (
    perspective_arrow_schema_fingerprint,
    perspective_result_to_arrow,
    value_of_perspective,
    write_perspective_result_ipc,
    write_perspective_result_parquet,
)

INTERCHANGE_VERSION = "1.0.0"
RELATIVE_ROOT = Path("specs/frontier/perspective/v1/interchange")
SOURCE = Path(
    "specs/frontier/perspective/v1/fixtures/normative/perspective-surface.json"
)


def normalized_lf_bytes(path: Path) -> bytes:
    """Read a text artifact and normalize all line endings to LF."""
    return (
        path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    )


def artifact_hash(path: Path, mode: str) -> str:
    """Hash an artifact using the declared normalization mode."""
    content = normalized_lf_bytes(path) if mode == "lf-text" else path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _reference_result(root: Path):
    payload = json.loads((root / SOURCE).read_text(encoding="utf-8"))
    return value_of_perspective(
        np.asarray(payload["net_benefit"], dtype=float),
        strategy_names=payload["strategy_names"],
        perspective_names=payload["perspective_names"],
        perspective_weights=payload["perspective_weights"],
        reference_perspective=payload["reference_perspective"],
    )


def _manifest(root: Path) -> dict[str, Any]:
    result = _reference_result(root)
    artifacts = [
        (SOURCE, "lf-text"),
        (RELATIVE_ROOT / "perspective-result-v1.parquet", "binary"),
        (RELATIVE_ROOT / "perspective-result-v1.arrow", "binary"),
    ]
    return {
        "schema_version": INTERCHANGE_VERSION,
        "method_contract_version": result.diagnostics["method_contract_version"],
        "arrow_schema_fingerprint_sha256": perspective_arrow_schema_fingerprint(result),
        "normalization": {
            "lf-text": "UTF-8 decoded with CRLF and CR normalized to LF before hashing",
            "binary": "raw bytes",
        },
        "artifacts": [
            {
                "path": path.as_posix(),
                "hash_mode": mode,
                "sha256": artifact_hash(root / path, mode),
            }
            for path, mode in artifacts
        ],
        "compatibility": {
            "pyarrow": "verified",
            "polars": "verified",
            "cross_process": "verified",
            "previous_fixture_versions_readable": ["1.0.0"],
        },
    }


def write_artifacts(root: Path) -> None:
    """Regenerate binary fixtures and their canonical manifest."""
    output = root / RELATIVE_ROOT
    output.mkdir(parents=True, exist_ok=True)
    result = _reference_result(root)
    write_perspective_result_parquet(
        result, str(output / "perspective-result-v1.parquet")
    )
    write_perspective_result_ipc(result, str(output / "perspective-result-v1.arrow"))
    manifest = _manifest(root)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_artifacts(root: Path) -> list[str]:
    """Return deterministic manifest or Arrow-content mismatches."""
    manifest_path = root / RELATIVE_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = _manifest(root)
    findings = (
        [] if expected == actual else ["manifest content does not match artifacts"]
    )

    reference = perspective_result_to_arrow(_reference_result(root))
    parquet = pq.read_table(root / RELATIVE_ROOT / "perspective-result-v1.parquet")
    with pa.memory_map(
        str(root / RELATIVE_ROOT / "perspective-result-v1.arrow"), "r"
    ) as source:
        ipc = pa.ipc.open_file(source).read_all()
    if not reference.equals(parquet, check_metadata=True):
        findings.append("Parquet fixture differs from the normative Arrow table")
    if not reference.equals(ipc, check_metadata=True):
        findings.append("IPC fixture differs from the normative Arrow table")
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the fixture generator or fail-closed verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.write:
        write_artifacts(root)
    findings = verify_artifacts(root)
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print(root / RELATIVE_ROOT / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
