#!/usr/bin/env python3
"""Create and validate digest-bound release evidence manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SWHID_PATTERN = re.compile(r"^swh:1:snp:[0-9a-f]{40}$")
PROVENANCE_PREDICATE = "https://slsa.dev/provenance/v1"
PACKAGE_SUFFIXES = (".whl", ".tar.gz")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    """Load JSON from *path* with a stable error boundary."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load JSON evidence {path}: {error}") from error


def _package_assets(assets: list[dict[str, Any]]) -> dict[str, str]:
    """Return package artifact names and digests from asset records."""
    return {
        str(asset["name"]): str(asset["sha256"])
        for asset in assets
        if str(asset["name"]).endswith(PACKAGE_SUFFIXES)
    }


def _checksum_entries(path: Path) -> dict[str, str]:
    """Parse a GNU-style SHA256SUMS file."""
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not SHA256_PATTERN.fullmatch(parts[0]):
            raise ValueError(f"invalid SHA256SUMS line {line_number}: {line!r}")
        name = parts[1].lstrip("*")
        if not name or name in entries:
            raise ValueError(f"duplicate or empty SHA256SUMS name: {name!r}")
        entries[name] = parts[0]
    return entries


def _attestation_record(
    report_path: Path,
    *,
    package_assets: dict[str, str],
    commit: str,
) -> dict[str, Any]:
    """Select provenance covering the exact package artifact set and commit."""
    report = _load_json(report_path)
    if not isinstance(report, list):
        raise TypeError("attestation report must be a JSON list")

    for candidate in report:
        if not isinstance(candidate, dict):
            continue
        verification = candidate.get("verificationResult")
        if not isinstance(verification, dict):
            continue
        statement = verification.get("statement")
        if not isinstance(statement, dict):
            continue
        if statement.get("predicateType") != PROVENANCE_PREDICATE:
            continue

        subject_map = {
            str(item.get("name")): str(item.get("digest", {}).get("sha256"))
            for item in statement.get("subject", [])
            if isinstance(item, dict) and isinstance(item.get("digest"), dict)
        }
        if any(
            subject_map.get(name) != digest for name, digest in package_assets.items()
        ):
            continue

        predicate = statement.get("predicate")
        if not isinstance(predicate, dict):
            continue
        build_definition = predicate.get("buildDefinition")
        if not isinstance(build_definition, dict):
            continue
        dependencies = build_definition.get("resolvedDependencies", [])
        source = next(
            (
                dependency
                for dependency in dependencies
                if isinstance(dependency, dict)
                and isinstance(dependency.get("digest"), dict)
                and dependency["digest"].get("gitCommit") == commit
            ),
            None,
        )
        if source is None:
            continue

        run_details = predicate.get("runDetails")
        metadata = (
            run_details.get("metadata", {}) if isinstance(run_details, dict) else {}
        )
        return {
            "status": "verified",
            "predicate_type": PROVENANCE_PREDICATE,
            "source_commit": commit,
            "source_uri": source.get("uri"),
            "invocation": (
                metadata.get("invocationId") if isinstance(metadata, dict) else None
            ),
            "subjects": [
                {"name": name, "sha256": digest}
                for name, digest in sorted(package_assets.items())
            ],
        }

    raise ValueError(
        "no matching provenance attestation covers the exact package assets "
        f"and source commit {commit}"
    )


def create_manifest(
    *,
    repository: str,
    version: str,
    tag: str,
    commit: str,
    release_url: str,
    asset_directory: Path,
    sbom_path: Path,
    sbom_recorded_path: str,
    software_heritage_id: str,
    attestation_report: Path,
    sbom_attached_to_release: bool,
) -> dict[str, Any]:
    """Create a release manifest from files and verified provenance evidence."""
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("release commit must be a full lowercase Git SHA")
    if tag != f"v{version}":
        raise ValueError(f"release tag {tag!r} does not bind version {version!r}")
    if not SWHID_PATTERN.fullmatch(software_heritage_id):
        raise ValueError("Software Heritage ID must be a snapshot SWHID")
    if not asset_directory.is_dir():
        raise ValueError(f"release asset directory does not exist: {asset_directory}")
    if not sbom_path.is_file():
        raise ValueError(f"SBOM does not exist: {sbom_path}")

    assets = [
        {
            "name": path.name,
            "path": f"{asset_directory.name}/{path.name}",
            "sha256": _sha256(path),
            "size": path.stat().st_size,
            "url": (
                f"{release_url.rsplit('/tag/', maxsplit=1)[0]}"
                f"/download/{tag}/{path.name}"
            ),
        }
        for path in sorted(asset_directory.iterdir())
        if path.is_file()
    ]
    package_assets = _package_assets(assets)
    if not package_assets:
        raise ValueError("release evidence contains no wheel or source distribution")

    checksum_path = asset_directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError("release evidence is missing SHA256SUMS")
    checksums = _checksum_entries(checksum_path)
    for name, digest in package_assets.items():
        if checksums.get(name) != digest:
            raise ValueError(f"SHA256SUMS does not match release asset {name}")

    sbom_document = _load_json(sbom_path)
    if not isinstance(sbom_document, dict):
        raise TypeError("CycloneDX SBOM must be a JSON object")
    sbom_digest = _sha256(sbom_path)
    if sbom_attached_to_release and not any(
        asset["name"] == Path(sbom_recorded_path).name
        and asset["sha256"] == sbom_digest
        for asset in assets
    ):
        raise ValueError(
            "SBOM cannot be marked attached without a matching release asset"
        )

    return {
        "schema_version": "1.0.0",
        "release": {
            "repository": repository,
            "version": version,
            "tag": tag,
            "commit": commit,
            "url": release_url,
        },
        "release_assets": assets,
        "attestation": _attestation_record(
            attestation_report,
            package_assets=package_assets,
            commit=commit,
        ),
        "sbom": {
            "format": sbom_document.get("bomFormat"),
            "spec_version": sbom_document.get("specVersion"),
            "path": sbom_recorded_path,
            "sha256": sbom_digest,
            "source_commit": commit,
            "source_tag": tag,
            "attached_to_github_release": sbom_attached_to_release,
            "attachment_status": (
                "attached" if sbom_attached_to_release else "not_attached"
            ),
        },
        "software_heritage": {
            "origin": f"https://github.com/{repository}",
            "snapshot_swhid": software_heritage_id,
        },
    }


def _validate_git_binding(
    release: dict[str, Any],
    repository_root: Path,
) -> list[str]:
    """Validate the manifest tag against the local Git object database."""
    tag = release.get("tag")
    commit = release.get("commit")
    if not isinstance(tag, str) or not isinstance(commit, str):
        return ["release tag and commit are required before Git validation"]
    result = subprocess.run(  # noqa: S603 - arguments are a validated tag.
        ["/usr/bin/git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [f"release tag is not available locally: {tag}"]
    resolved = result.stdout.strip()
    if resolved != commit:
        return [
            f"release tag {tag} resolves to {resolved}, not manifest commit {commit}"
        ]
    return []


def validate_manifest(
    manifest: dict[str, Any],
    *,
    artifact_root: Path,
    repository_root: Path | None,
    require_asset_files: bool = True,
) -> list[str]:
    """Return fail-closed findings for a release evidence manifest."""
    errors: list[str] = []
    if manifest.get("schema_version") != "1.0.0":
        errors.append("unsupported release-evidence schema version")

    release = manifest.get("release")
    if not isinstance(release, dict):
        return [*errors, "release object is required"]
    version = release.get("version")
    tag = release.get("tag")
    commit = release.get("commit")
    if not isinstance(version, str) or tag != f"v{version}":
        errors.append("release version and tag are not bound")
    if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
        errors.append("release commit is not a full lowercase Git SHA")
    if repository_root is not None:
        errors.extend(_validate_git_binding(release, repository_root))

    assets = manifest.get("release_assets")
    if not isinstance(assets, list) or not assets:
        errors.append("release assets are required")
        assets = []
    names: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            errors.append("release asset records must be objects")
            continue
        name = asset.get("name")
        digest = asset.get("sha256")
        recorded_path = asset.get("path")
        if not isinstance(name, str) or not name or name in names:
            errors.append(f"release asset name is invalid or duplicated: {name!r}")
            continue
        names.add(name)
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            errors.append(f"release asset digest is invalid: {name}")
        if require_asset_files:
            if not isinstance(recorded_path, str):
                errors.append(f"release asset path is missing: {name}")
                continue
            local_path = artifact_root / recorded_path
            if not local_path.is_file():
                errors.append(f"release asset is missing: {recorded_path}")
            elif _sha256(local_path) != digest:
                errors.append(f"release asset digest mismatch: {name}")

    package_assets = _package_assets(
        [asset for asset in assets if isinstance(asset, dict)]
    )
    attestation = manifest.get("attestation")
    if not isinstance(attestation, dict):
        errors.append("provenance attestation record is required")
    else:
        if attestation.get("status") != "verified":
            errors.append("provenance attestation is not verified")
        if attestation.get("predicate_type") != PROVENANCE_PREDICATE:
            errors.append("provenance predicate type is not SLSA v1")
        if attestation.get("source_commit") != commit:
            errors.append("provenance source commit does not match release commit")
        subject_map = {
            str(item.get("name")): str(item.get("sha256"))
            for item in attestation.get("subjects", [])
            if isinstance(item, dict)
        }
        if subject_map != package_assets:
            errors.append("provenance subjects do not match release package assets")

    sbom = manifest.get("sbom")
    if not isinstance(sbom, dict):
        errors.append("SBOM evidence is required")
    else:
        sbom_path = sbom.get("path")
        sbom_digest = sbom.get("sha256")
        if sbom.get("format") != "CycloneDX":
            errors.append("SBOM format must be CycloneDX")
        if not isinstance(sbom_digest, str) or not SHA256_PATTERN.fullmatch(
            sbom_digest
        ):
            errors.append("SBOM SHA-256 digest is invalid")
        if sbom.get("source_commit") != commit or sbom.get("source_tag") != tag:
            errors.append("SBOM source revision does not match the release")
        if not isinstance(sbom_path, str):
            errors.append("SBOM path is missing")
        else:
            local_sbom = artifact_root / sbom_path
            if not local_sbom.is_file():
                errors.append(f"SBOM is missing: {sbom_path}")
            else:
                if _sha256(local_sbom) != sbom_digest:
                    errors.append("SBOM digest mismatch")
                try:
                    sbom_document = _load_json(local_sbom)
                except ValueError as error:
                    errors.append(str(error))
                else:
                    if not isinstance(sbom_document, dict):
                        errors.append("SBOM must be a JSON object")
                    else:
                        if sbom_document.get("bomFormat") != sbom.get("format"):
                            errors.append("SBOM format metadata mismatch")
                        if sbom_document.get("specVersion") != sbom.get("spec_version"):
                            errors.append("SBOM specification version mismatch")
                        metadata = sbom_document.get("metadata", {})
                        component = (
                            metadata.get("component", {})
                            if isinstance(metadata, dict)
                            else {}
                        )
                        if (
                            not isinstance(component, dict)
                            or component.get("version") != version
                        ):
                            errors.append(
                                "SBOM component version does not match release"
                            )
                        properties = (
                            metadata.get("properties", [])
                            if isinstance(metadata, dict)
                            else []
                        )
                        property_map = {
                            str(item.get("name")): str(item.get("value"))
                            for item in properties
                            if isinstance(item, dict)
                        }
                        if property_map.get("voiage:source:commit") != commit:
                            errors.append("SBOM does not record the release commit")
                        if property_map.get("voiage:source:tag") != tag:
                            errors.append("SBOM does not record the release tag")
        attached = sbom.get("attached_to_github_release")
        status = sbom.get("attachment_status")
        if not isinstance(attached, bool):
            errors.append("SBOM release-attachment state must be Boolean")
        elif attached and status != "attached":
            errors.append("attached SBOM must have attachment status attached")
        elif attached and not any(
            isinstance(asset, dict)
            and asset.get("name") == Path(str(sbom_path)).name
            and asset.get("sha256") == sbom_digest
            for asset in assets
        ):
            errors.append("attached SBOM is not present in the release asset set")
        elif not attached and status != "not_attached":
            errors.append("unattached SBOM must have attachment status not_attached")

    software_heritage = manifest.get("software_heritage")
    if not isinstance(software_heritage, dict) or not SWHID_PATTERN.fullmatch(
        str(software_heritage.get("snapshot_swhid", ""))
    ):
        errors.append("a valid Software Heritage snapshot SWHID is required")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a release manifest")
    create.add_argument("--repository", required=True)
    create.add_argument("--version", required=True)
    create.add_argument("--tag", required=True)
    create.add_argument("--commit", required=True)
    create.add_argument("--release-url", required=True)
    create.add_argument("--asset-directory", type=Path, required=True)
    create.add_argument("--sbom", type=Path, required=True)
    create.add_argument("--sbom-recorded-path", required=True)
    create.add_argument("--software-heritage-id", required=True)
    create.add_argument("--attestation-report", type=Path, required=True)
    create.add_argument("--sbom-attached-to-release", action="store_true")
    create.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="validate a release manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--artifact-root", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path)
    validate.add_argument("--allow-missing-release-assets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release-evidence command-line interface."""
    args = _build_parser().parse_args(argv)
    if args.command == "create":
        try:
            manifest = create_manifest(
                repository=args.repository,
                version=args.version,
                tag=args.tag,
                commit=args.commit,
                release_url=args.release_url,
                asset_directory=args.asset_directory,
                sbom_path=args.sbom,
                sbom_recorded_path=args.sbom_recorded_path,
                software_heritage_id=args.software_heritage_id,
                attestation_report=args.attestation_report,
                sbom_attached_to_release=args.sbom_attached_to_release,
            )
        except (TypeError, ValueError) as error:
            print(f"release evidence creation failed: {error}", file=sys.stderr)
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
        return 0

    try:
        manifest = _load_json(args.manifest)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print("release evidence manifest must be a JSON object", file=sys.stderr)
        return 1
    errors = validate_manifest(
        manifest,
        artifact_root=args.artifact_root,
        repository_root=args.repository_root,
        require_asset_files=not args.allow_missing_release_assets,
    )
    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"validated release evidence: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
