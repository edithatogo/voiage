#!/usr/bin/env python3
"""Build twice from one revision and compare byte and archive inventories."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import TypedDict, cast
import zipfile


class ArchiveEntry(TypedDict):
    """One normalized archive member."""

    path: str
    sha256: str
    size: int


class ArtifactEvidence(TypedDict):
    """One build artifact identity."""

    filename: str
    sha256: str
    size: int
    inventory_sha256: str
    entries: int


class ArtifactComparison(TypedDict):
    """Comparison of one artifact across two builds."""

    filename: str
    byte_identical: bool
    inventory_identical: bool
    differing_entries: list[str]
    first: ArtifactEvidence
    second: ArtifactEvidence


class BuildReport(TypedDict):
    """Deterministic build-comparison envelope."""

    schema_version: str
    names_match: bool
    artifact_set_complete: bool
    reproducible: bool
    artifacts: list[ArtifactComparison]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive_inventory(path: Path) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                payload = archive.read(name)
                entries.append(
                    {"path": name, "sha256": _sha256(payload), "size": len(payload)}
                )
        return entries
    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unable to read archive member: {member.name}")
                payload = extracted.read()
                entries.append(
                    {
                        "path": member.name,
                        "sha256": _sha256(payload),
                        "size": len(payload),
                    }
                )
        return entries
    raise ValueError(f"unsupported build artifact: {path.name}")


def artifact_evidence(path: Path) -> ArtifactEvidence:
    """Return byte and normalized-content identities for one artifact."""
    payload = path.read_bytes()
    inventory = _archive_inventory(path)
    canonical = json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode()
    return {
        "filename": path.name,
        "sha256": _sha256(payload),
        "size": len(payload),
        "inventory_sha256": _sha256(canonical),
        "entries": len(inventory),
    }


def normalize_sdist(path: Path, *, epoch: int) -> None:
    """Rewrite an sdist with deterministic member order and metadata."""
    normalized = path.with_name(f".{path.name}.normalized")
    with (
        tarfile.open(path, mode="r:gz") as source,
        normalized.open("wb") as raw_output,
        gzip.GzipFile(
            fileobj=raw_output, mode="wb", filename="", mtime=epoch
        ) as zipped,
        tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as target,
    ):
        for member in sorted(source.getmembers(), key=lambda item: item.name):
            canonical = tarfile.TarInfo(member.name)
            canonical.type = member.type
            canonical.linkname = member.linkname
            canonical.mode = member.mode
            canonical.uid = 0
            canonical.gid = 0
            canonical.uname = ""
            canonical.gname = ""
            canonical.mtime = epoch
            canonical.size = member.size if member.isfile() else 0
            extracted = source.extractfile(member) if member.isfile() else None
            target.addfile(canonical, extracted)
    normalized.replace(path)


def compare_build_directories(first: Path, second: Path) -> BuildReport:
    """Compare two build directories and return a deterministic evidence envelope."""

    def distributions(root: Path) -> dict[str, Path]:
        return {
            path.name: path
            for path in root.iterdir()
            if path.is_file()
            and (path.suffix in {".whl", ".zip"} or path.name.endswith(".tar.gz"))
        }

    first_paths = distributions(first)
    second_paths = distributions(second)
    names_match = first_paths.keys() == second_paths.keys()
    artifact_set_complete = (
        len(first_paths) == 2
        and sum(name.endswith(".whl") for name in first_paths) == 1
        and sum(name.endswith(".tar.gz") for name in first_paths) == 1
    )
    artifacts: list[ArtifactComparison] = []
    for name in sorted(first_paths.keys() & second_paths.keys()):
        left = artifact_evidence(first_paths[name])
        right = artifact_evidence(second_paths[name])
        left_inventory = {
            item["path"]: item for item in _archive_inventory(first_paths[name])
        }
        right_inventory = {
            item["path"]: item for item in _archive_inventory(second_paths[name])
        }
        artifacts.append(
            {
                "filename": name,
                "byte_identical": left["sha256"] == right["sha256"],
                "inventory_identical": left["inventory_sha256"]
                == right["inventory_sha256"],
                "differing_entries": sorted(
                    path
                    for path in left_inventory.keys() | right_inventory.keys()
                    if left_inventory.get(path) != right_inventory.get(path)
                ),
                "first": left,
                "second": right,
            }
        )
    return {
        "schema_version": "1.0.0",
        "names_match": names_match,
        "artifact_set_complete": artifact_set_complete,
        "reproducible": names_match
        and artifact_set_complete
        and bool(artifacts)
        and all(item["byte_identical"] for item in artifacts),
        "artifacts": artifacts,
    }


def _source_date_epoch(repo: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required")
    completed = subprocess.run(  # noqa: S603
        [git, "log", "-1", "--format=%ct"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _embed_sdist_provenance(repo: Path) -> None:
    """Create the immutable identity used when rebuilding from an sdist.

    A wheel built from the checkout can query Git directly, but the second
    build performed by this script intentionally happens from the Git-less
    source archive produced by the first build.  Generate the sdist-only
    provenance file before either build so that both paths have the same
    source identity.
    """
    script = repo / "scripts/embed_sdist_provenance.py"
    if not script.is_file():
        raise RuntimeError(f"missing sdist provenance helper: {script}")
    _ = subprocess.run(  # noqa: S603 - repository-local fixed script
        [shutil.which("python") or sys.executable, str(script)],
        cwd=repo,
        check=True,
    )


def _build_environment(
    repo: Path,
    target_dir: Path,
    *,
    platform_name: str = os.name,
    source_date_epoch: str | None = None,
) -> dict[str, str]:
    """Return deterministic inputs shared by two independent native builds."""
    environment = {
        **os.environ,
        "SOURCE_DATE_EPOCH": source_date_epoch or _source_date_epoch(repo),
        "CARGO_TARGET_DIR": str(target_dir.resolve()),
    }
    if platform_name == "nt":
        inherited = environment.get("RUSTFLAGS", "").strip()
        environment["RUSTFLAGS"] = f"{inherited} -C link-arg=/Brepro".strip()
    return environment


def verify_reproducible_build(
    repo: Path, *, output_dir: Path | None = None
) -> BuildReport:
    """Run two isolated uv builds from the same source revision."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable is required")
    provenance = repo / "rust/crates/voiage-python/source-provenance.txt"
    generated_provenance = not provenance.exists()
    _embed_sdist_provenance(repo)
    try:
        with (
            tempfile.TemporaryDirectory(prefix="voiage-build-a-") as first_temp,
            tempfile.TemporaryDirectory(prefix="voiage-build-b-") as second_temp,
            tempfile.TemporaryDirectory(prefix="voiage-cargo-target-") as target_temp,
        ):
            first = Path(first_temp)
            second = Path(second_temp)
            target = Path(target_temp)
            environment = _build_environment(repo, target)
            source_date_epoch = environment["SOURCE_DATE_EPOCH"]
            for destination in (first, second):
                shutil.rmtree(target)
                target.mkdir()
                _ = subprocess.run(  # noqa: S603
                    [uv, "build", "--out-dir", str(destination)],
                    cwd=repo,
                    env=environment,
                    check=True,
                )
                for sdist in destination.glob("*.tar.gz"):
                    normalize_sdist(sdist, epoch=int(source_date_epoch))
            report = compare_build_directories(first, second)
            if output_dir is not None and report["reproducible"]:
                output_dir.mkdir(parents=True, exist_ok=True)
                for artifact in first.iterdir():
                    if artifact.is_file():
                        _ = shutil.copy2(artifact, output_dir / artifact.name)
            return report
    finally:
        if generated_provenance:
            provenance.unlink(missing_ok=True)


def main() -> int:
    """Run the reproducibility check from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--dist-dir", type=Path)
    args = parser.parse_args()
    repo = cast("Path", args.repo)
    output = cast("Path", args.output)
    dist_dir = cast("Path | None", args.dist_dir)
    report = verify_reproducible_build(repo.resolve(), output_dir=dist_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _ = print(json.dumps(report, sort_keys=True))
    return 0 if report["reproducible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
