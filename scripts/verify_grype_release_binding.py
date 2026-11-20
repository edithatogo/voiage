"""Bind a retained SBOM to the reviewed distributions and verified attestations.

The caller must first run ``gh attestation verify`` with the repository, signer
workflow, source digest and CycloneDX predicate policy. This module checks its
successful JSON output; it is not a signature verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular evidence file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular subject: {path.name}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_binding(
    manifest: Path,
    sbom: Path,
    distributions: Path,
    attestations: Path,
    source_sha: str,
    release_tag: str,
    resolved_sbom: Path,
) -> dict[str, Any]:
    """Reject a changed, incomplete or differently attested release candidate."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("invalid expected source commit")
    document = _read_json(manifest)
    inventory = _read_json(sbom)
    if document.get("schema_version") != "1.0.0":
        raise ValueError("unsupported release manifest")
    if document.get("source_commit") != source_sha:
        raise ValueError("release manifest source does not match the signed tag")
    properties = inventory.get("metadata", {}).get("properties", [])
    for name, value in (
        ("voiage:source:commit", source_sha),
        ("voiage:source:tag", release_tag),
    ):
        matches = [item.get("value") for item in properties if item.get("name") == name]
        if matches != [value]:
            raise ValueError(f"SBOM identity mismatch: {name}")

    wheels = sorted(distributions.glob("*.whl"))
    sdists = sorted(distributions.glob("*.tar.gz"))
    if not wheels or len(sdists) != 1:
        raise ValueError("expected reviewed wheels and exactly one source distribution")
    subjects = [*wheels, *sdists]
    expected = {path.name: path for path in [*subjects, sbom, resolved_sbom]}
    rows = document.get("artifacts")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("release manifest does not describe the exact candidate set")
    observed: set[str] = set()
    for row in rows:
        name, digest = row.get("name"), row.get("sha256")
        if name not in expected or name in observed:
            raise ValueError("unknown or duplicate release manifest subject")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid release subject digest")
        if _sha256(expected[name]) != digest:
            raise ValueError(f"release digest mismatch: {name}")
        observed.add(name)

    for subject in subjects:
        results = _read_json(attestations / f"{subject.name}.attestation.json")
        if not isinstance(results, list) or not results:
            raise ValueError(f"no verified SBOM attestation: {subject.name}")
        digest = _sha256(subject)
        matched = False
        for result in results:
            statement = result.get("verificationResult", {}).get("statement", {})
            matching_subject = any(
                entry.get("digest", {}).get("sha256") == digest
                for entry in statement.get("subject", [])
            )
            if (
                statement.get("predicateType") == "https://cyclonedx.org/bom"
                and statement.get("predicate") == inventory
                and matching_subject
            ):
                matched = True
                break
        if not matched:
            raise ValueError(
                f"verified predicate differs from scanned SBOM: {subject.name}"
            )
    return {
        "status": "verified",
        "source_commit": source_sha,
        "release_tag": release_tag,
        "sbom_sha256": _sha256(sbom),
        "distribution_count": len(subjects),
    }


def main() -> None:
    """Check a reviewed release without modifying any publication input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--resolved-sbom", type=Path, required=True)
    parser.add_argument("--distributions", type=Path, required=True)
    parser.add_argument("--attestations", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    result = verify_binding(
        args.manifest,
        args.sbom,
        args.distributions,
        args.attestations,
        args.source_sha,
        args.release_tag,
        args.resolved_sbom,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
