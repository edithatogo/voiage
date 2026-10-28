"""Qualify a current source archive without claiming native build acceptance."""

from __future__ import annotations

import argparse
from copy import deepcopy
from email.parser import Parser
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

PROVENANCE = "rust/crates/voiage-python/source-provenance.txt"
MAX_METADATA = 2 * 1024 * 1024
MAX_UNPACKED = 1024 * 1024 * 1024
MAX_MEMBERS = 100000


def _git(repository: Path, *arguments: str) -> bytes:
    """Read fixed Git data using a selected immutable revision."""
    git = shutil.which("git")
    if git is None:
        raise ValueError("Git is required for source authentication")
    return subprocess.check_output([git, "-C", str(repository), *arguments])  # noqa: S603


def _requirements(values: list[str]) -> dict[str, tuple[str, str, str]]:
    result = {}
    for value in values:
        requirement = Requirement(value)
        name = canonicalize_name(requirement.name)
        if name in result:
            raise ValueError(f"Duplicate requirement: {name}")
        result[name] = (
            str(requirement.specifier),
            str(requirement.marker or ""),
            requirement.url or "",
        )
    return result


def qualify_source(
    archive: Path, repository: Path, revision: str, expected_sha256: str, policy: Path
) -> dict[str, Any]:
    """Check real payloads against Git, including the bounded sdist manifest rewrite."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("A full immutable revision and archive SHA-256 are required")
    with archive.open("rb") as stream:
        observed_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed_sha != expected_sha256:
        raise ValueError("Archive checksum mismatch")
    tree = _git(repository, "rev-parse", revision + "^{tree}").decode().strip()
    entries = {}
    for entry in _git(repository, "ls-tree", "-r", "-z", revision).split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, oid = header.split()
        if kind == b"blob":
            entries[path.decode()] = (mode.decode(), oid.decode())
    required = {name for name in entries if name.startswith("voiage/")}
    if not required or "rust/crates/voiage-python/Cargo.toml" not in entries:
        raise ValueError("Selected source lacks the Python/native package")
    special = {"pyproject.toml", "PKG-INFO", PROVENANCE, "rust/Cargo.toml"}
    metadata: dict[str, bytes] = {}
    seen: set[str] = set()
    roots: set[str] = set()
    verified: set[str] = set()
    total = 0
    with tarfile.open(archive, "r|*") as package:
        for count, member in enumerate(package, 1):
            if count > MAX_MEMBERS:
                raise ValueError("Archive exceeds the bounded source inventory")
            parts = PurePosixPath(member.name).parts
            if (
                member.name.startswith("/")
                or ".." in parts
                or "\\" in member.name
                or len(parts) < 1
            ):
                raise ValueError("Unsafe archive member")
            roots.add(parts[0])
            name = "/".join(parts[1:])
            if member.isdir():
                continue
            if not member.isfile() or not name or name in seen:
                raise ValueError("Nonregular or duplicate archive member")
            seen.add(name)
            total += member.size
            if total > MAX_UNPACKED:
                raise ValueError("Archive exceeds the bounded source inventory")
            if (
                name in special or name.endswith("Cargo.toml")
            ) and member.size > MAX_METADATA:
                raise ValueError("Oversized source metadata")
            selected = package.extractfile(member)
            if selected is None:
                raise ValueError("Unreadable source member")
            digest = hashlib.sha1(
                b"blob " + str(member.size).encode() + b"\0", usedforsecurity=False
            )
            captured = bytearray()
            while chunk := selected.read(65536):
                digest.update(chunk)
                if name in special or name.endswith("Cargo.toml"):
                    captured.extend(chunk)
            if name in special or name.endswith("Cargo.toml"):
                metadata[name] = bytes(captured)
            if name not in {
                "PKG-INFO",
                PROVENANCE,
                "rust/Cargo.toml",
            }:
                expected = entries.get(name)
                if (
                    expected is None
                    or expected[0] not in {"100644", "100755"}
                    or digest.hexdigest() != expected[1]
                ):
                    raise ValueError(f"Packaged source differs from Git: {name}")
                verified.add(name)
            if name == "pyproject.toml" and digest.hexdigest() != entries[name][1]:
                raise ValueError("Project metadata differs from Git")
    if len(roots) != 1 or not required <= verified:
        raise ValueError("Incomplete or multi-root source archive")
    if not special <= metadata.keys():
        raise ValueError("Missing immutable source metadata")
    expected_record = f"revision={revision}\ntree={tree}\nclean=true\n".encode()
    if metadata[PROVENANCE] != expected_record:
        raise ValueError("Embedded source provenance mismatch")
    canonical = json.loads(policy.read_text())
    configuration = tomllib.loads(metadata["pyproject.toml"].decode())
    project = configuration.get("project", {})
    original_workspace = tomllib.loads(
        _git(repository, "show", revision + ":rust/Cargo.toml").decode()
    )
    version = project.get("version")
    if version is None and project.get("dynamic") == ["version"]:
        if (
            configuration.get("tool", {}).get("maturin", {}).get("manifest-path")
            != "rust/crates/voiage-python/Cargo.toml"
        ):
            raise ValueError("Unqualified dynamic version source")
        version = (
            original_workspace.get("workspace", {}).get("package", {}).get("version")
        )
    if version is None:
        raise ValueError("Missing static or Cargo-bound package version")
    if not isinstance(version, str):
        raise TypeError("Package version must be a string")
    if project.get("name") != "voiage" or roots != {f"voiage-{version}"}:
        raise ValueError("Source package identity mismatch")
    if project["requires-python"] != canonical["requires_python"]:
        raise ValueError("Python runtime policy mismatch")
    expected_requirements = _requirements(canonical["runtime"])
    if _requirements(project["dependencies"]) != expected_requirements:
        raise ValueError("Source runtime dependency policy mismatch")
    retired = {"pandas", "xarray"}
    for group in [
        project["dependencies"],
        *project.get("optional-dependencies", {}).values(),
    ]:
        if retired & _requirements(group).keys():
            raise ValueError("Retired engine in published dependencies")
    package_info = Parser().parsestr(metadata["PKG-INFO"].decode())
    if (
        package_info["Name"] != project["name"]
        or package_info["Version"] != version
        or SpecifierSet(package_info["Requires-Python"] or "")
        != SpecifierSet(canonical["requires_python"])
    ):
        raise ValueError("PKG-INFO disagrees with project identity/runtime")
    core_requirements = []
    for text in package_info.get_all("Requires-Dist", []):
        requirement = Requirement(text)
        if canonicalize_name(requirement.name) in retired:
            raise ValueError("Retired engine in distribution metadata")
        if requirement.marker is None:
            core_requirements.append(text)
    if _requirements(core_requirements) != expected_requirements:
        raise ValueError("Distribution core dependency metadata mismatch")
    packaged_workspace = tomllib.loads(metadata["rust/Cargo.toml"].decode())
    expected_workspace = deepcopy(original_workspace)
    original_members = original_workspace["workspace"]["members"]
    packaged_members = [
        name for name in original_members if f"rust/{name}/Cargo.toml" in seen
    ]
    if "crates/voiage-python" not in packaged_members:
        raise ValueError("Missing Python extension crate")
    expected_workspace["workspace"]["members"] = packaged_members
    if packaged_workspace != expected_workspace:
        raise ValueError("Unqualified Cargo workspace transformation")
    required_native = {
        path
        for path in entries
        if any(path.startswith("rust/" + member + "/") for member in packaged_members)
    } | {"rust/Cargo.lock"}
    if not required_native <= verified:
        raise ValueError("Incomplete packaged native source")
    for name, raw in metadata.items():
        if not name.endswith("Cargo.toml") or name == "rust/Cargo.toml":
            continue
        cargo = tomllib.loads(raw.decode())
        tables = [cargo, *cargo.get("target", {}).values()]
        for table in tables:
            for group in ["dependencies", "build-dependencies", "dev-dependencies"]:
                for dep_name, declared in table.get(group, {}).items():
                    if not isinstance(declared, dict):
                        continue
                    base = str(PurePosixPath(name).parent)
                    concrete = declared
                    if declared.get("workspace"):
                        concrete = (
                            original_workspace["workspace"]
                            .get("dependencies", {})
                            .get(dep_name, {})
                        )
                        base = "rust"
                    if isinstance(concrete, dict) and "path" in concrete:
                        manifest = posixpath.normpath(
                            base + "/" + concrete["path"] + "/Cargo.toml"
                        )
                        if not manifest.startswith("rust/") or manifest not in seen:
                            raise ValueError(
                                f"Missing local native dependency: {dep_name}"
                            )
    return {
        "schema_version": "1.0",
        "python": sys.version.split()[0],
        "source_archive_qualified": True,
        "source_revision": revision,
        "package_version": version,
        "source_tree": tree,
        "archive_sha256": observed_sha,
        "archive_bytes": archive.stat().st_size,
        "runtime_files_verified": len(required),
        "native_files_verified": len(
            {name for name in verified if name.startswith("rust/")}
        ),
        "total_source_files_verified": len(verified),
        "omitted_workspace_members": [
            name for name in original_members if name not in packaged_members
        ],
        "cargo_workspace_transform": "Only omit nonpackaged workspace members; local dependency closure remains present",
        "native_solver_executed": False,
        "native_build_executed": False,
        "installed_module_acceptance": False,
    }


def main() -> int:
    """Write exact source proof; native installation is deliberately a separate gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--policy", type=Path, default=Path("specs/dependency-runtime-policy.json")
    )
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 15, 1):
        parser.error("Source qualification requires Python 3.15.1")
    try:
        result = qualify_source(
            args.archive, args.repository, args.revision, args.sha256, args.policy
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        tarfile.TarError,
        subprocess.CalledProcessError,
    ) as error:
        print(
            json.dumps(
                {"source_archive_qualified": False, "error": str(error)}, indent=2
            )
        )
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
