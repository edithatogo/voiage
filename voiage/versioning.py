"""Version synchronization helpers for release automation.

The production Cargo workspace is authoritative. Package adapters and external
binding manifests must remain coherent with it, and release tags are accepted
only when they exactly encode that workspace version.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any

from defusedxml import ElementTree

try:  # Python 3.11+.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib  # type: ignore[import-not-found]

from collections.abc import Callable

REPO_ROOT = Path.cwd()
_DESCRIPTION_VERSION_RE = re.compile(r"^Version:\s*(?P<version>\S+)\s*$")


@dataclass(frozen=True, slots=True)
class VersionTarget:
    """A manifest whose version must match the canonical repo version."""

    label: str
    path: Path
    reader: Callable[[Path], str]


@dataclass(frozen=True, slots=True)
class VersionMismatch:
    """A manifest whose version does not match the canonical repo version."""

    label: str
    path: Path
    expected: str
    found: str


class VersionSyncError(RuntimeError):
    """Raised when version synchronization fails."""


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise VersionSyncError(f"{path}: expected TOML document object")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise VersionSyncError(f"{path}: expected JSON object")
    return data


def _read_canonical_version(pyproject_path: Path) -> str:
    """Read the Cargo workspace version adjacent to ``pyproject.toml``."""
    project = _load_toml(pyproject_path).get("project")
    if not isinstance(project, dict):
        raise VersionSyncError(f"{pyproject_path}: missing [project] table")
    dynamic = project.get("dynamic", [])
    if not isinstance(dynamic, list) or "version" not in dynamic:
        raise VersionSyncError(
            f"{pyproject_path}: project.version must be dynamic and Cargo-backed"
        )
    return _read_toml_version(
        pyproject_path.parent / "rust/Cargo.toml",
        key_path=("workspace", "package", "version"),
    )


def _read_json_version(path: Path) -> str:
    version = _load_json(path).get("version")
    if not isinstance(version, str) or not version.strip():
        raise VersionSyncError(f"{path}: missing version")
    return version


def _read_toml_version(path: Path, *, key_path: tuple[str, ...] = ()) -> str:
    data: Any = _load_toml(path)
    if key_path:
        for key in key_path:
            if not isinstance(data, dict) or key not in data:
                raise VersionSyncError(f"{path}: missing {'.'.join(key_path)}")
            data = data[key]
    else:
        data = data.get("version")
    if not isinstance(data, str) or not data.strip():
        raise VersionSyncError(f"{path}: missing version")
    return data


def _read_cargo_version(path: Path) -> str:
    return _read_toml_version(path, key_path=("package", "version"))


def _read_workspace_inherited_cargo_version(path: Path) -> str:
    data = _load_toml(path)
    package = data.get("package")
    if not isinstance(package, dict):
        raise VersionSyncError(f"{path}: missing package table")
    version = package.get("version")
    if isinstance(version, dict) and version.get("workspace") is True:
        return _read_toml_version(
            path.parents[2] / "Cargo.toml",
            key_path=("workspace", "package", "version"),
        )
    if isinstance(version, str) and version.strip():
        raise VersionSyncError(f"{path}: package version must inherit from workspace")
    raise VersionSyncError(f"{path}: package version must inherit from workspace")


def _read_csproj_version(path: Path) -> str:
    tree = ElementTree.parse(path)
    root = tree.getroot()
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag != "Version":
            continue
        version = (element.text or "").strip()
        if version:
            return version
    raise VersionSyncError(f"{path}: missing <Version>")


def _read_description_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _DESCRIPTION_VERSION_RE.match(line.strip())
        if match:
            version = match.group("version").strip()
            if version:
                return version
    raise VersionSyncError(f"{path}: missing Version field")


VERSION_TARGETS: tuple[VersionTarget, ...] = (
    VersionTarget(
        "Python Rust adapter",
        Path("rust/crates/voiage-python/Cargo.toml"),
        _read_workspace_inherited_cargo_version,
    ),
    VersionTarget(
        "TypeScript", Path("bindings/typescript/package.json"), _read_json_version
    ),
    VersionTarget("Julia", Path("bindings/julia/Project.toml"), _read_toml_version),
    VersionTarget("Rust", Path("bindings/rust/Cargo.toml"), _read_cargo_version),
    VersionTarget(
        ".NET",
        Path("bindings/dotnet/src/Voiage.Core/Voiage.Core.csproj"),
        _read_csproj_version,
    ),
    VersionTarget(
        "R", Path("r-package/voiageR/DESCRIPTION"), _read_description_version
    ),
)


def validate_release_tag(tag: str, repo_root: Path = REPO_ROOT) -> str:
    """Fail closed unless ``tag`` exactly matches the Cargo workspace version."""
    canonical = _read_canonical_version(repo_root / "pyproject.toml")
    expected = f"v{canonical}"
    if tag != expected:
        raise VersionSyncError(
            f"release tag {tag!r} must match {expected!r} from rust/Cargo.toml"
        )
    return canonical


def collect_version_mismatches(
    repo_root: Path = REPO_ROOT,
) -> tuple[str, list[VersionMismatch]]:
    """Collect manifest version mismatches against the canonical repo version."""
    canonical = _read_canonical_version(repo_root / "pyproject.toml")
    mismatches: list[VersionMismatch] = []
    for target in VERSION_TARGETS:
        manifest_path = repo_root / target.path
        if not manifest_path.is_file():
            raise VersionSyncError(f"missing manifest: {manifest_path}")
        found = target.reader(manifest_path)
        if found != canonical:
            mismatches.append(
                VersionMismatch(
                    label=target.label,
                    path=manifest_path,
                    expected=canonical,
                    found=found,
                )
            )
    return canonical, mismatches


def format_version_mismatches(
    mismatches: list[VersionMismatch],
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Format version mismatches as a human-readable diagnostic."""
    lines = ["version synchronization failed:"]
    for mismatch in mismatches:
        relpath = mismatch.path.relative_to(repo_root).as_posix()
        lines.append(
            f"- {mismatch.label}: {relpath} expected {mismatch.expected!r} but found {mismatch.found!r}"
        )
    return "\n".join(lines)


def validate_version_sync(
    repo_root: Path = REPO_ROOT,
) -> tuple[str, list[VersionMismatch]]:
    """Validate that binding manifests match the canonical repo version."""
    canonical, mismatches = collect_version_mismatches(repo_root)
    if mismatches:
        raise VersionSyncError(
            format_version_mismatches(mismatches, repo_root=repo_root)
        )
    return canonical, mismatches


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the version synchronization validator."""
    parser = argparse.ArgumentParser(
        description="Validate that package manifests match the canonical repo version."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to validate (defaults to the current checkout).",
    )
    parser.add_argument(
        "--release-tag",
        help="Require an exact v<workspace-version> release tag.",
    )
    args = parser.parse_args(argv)

    try:
        canonical, _ = validate_version_sync(args.repo_root)
        if args.release_tag is not None:
            validate_release_tag(args.release_tag, args.repo_root)
    except VersionSyncError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"validated version synchronization against {args.repo_root / 'pyproject.toml'} @ {canonical}"
    )
    return 0


__all__ = [
    "REPO_ROOT",
    "VersionMismatch",
    "VersionSyncError",
    "VersionTarget",
    "collect_version_mismatches",
    "format_version_mismatches",
    "main",
    "validate_release_tag",
    "validate_version_sync",
]


if __name__ == "__main__":  # pragma: no cover - exercised by workflow commands.
    raise SystemExit(main())
