"""Verify isolated consumption of packaged simulation-outcome Rust crates (O08)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from typing import Any

REQUIRED_CRATES = ("voiage-domain", "voiage-diagnostics", "voiage-numerics")
FORBIDDEN_PRODUCER_TOKENS = frozenset(
    {"kairos", "paignia", "tobac", "vop", "innovate", "mars"}
)
_TEST_SUMMARY_RE = re.compile(
    r"test result:\s+ok\.\s+(?P<passed>\d+)\s+passed;\s+(?P<failed>\d+)\s+failed;"
)


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 hex digest using streaming reads."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_unpack_crate(archive: Path, destination: Path) -> Path:
    """Safely unpack a single-root Cargo `.crate` archive without path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as package:
        members = package.getmembers()
        if not members:
            raise RuntimeError(f"empty crate archive: {archive}")
        for member in members:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"unsafe path in crate archive: {member.name}")
        package.extractall(destination, filter="data")
    root_name = archive.name.removesuffix(".crate")
    unpacked = destination / root_name
    if not unpacked.is_dir():
        raise RuntimeError(f"expected unpacked crate directory {unpacked}")
    return unpacked


def _package_source_path(manifest_path: str) -> Path:
    return Path(manifest_path).resolve().parent


def inspect_dependency_graph(
    metadata: dict[str, Any],
    *,
    repo_root: Path,
    unpacked_root: Path,
    expected_version: str,
) -> dict[str, Any]:
    """Fail if any producer engine appears or crates resolve inside source checkout."""
    packages = metadata.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError("cargo metadata returned no packages")

    resolved_repo = repo_root.resolve()
    resolved_unpacked = unpacked_root.resolve()
    seen_required: dict[str, dict[str, str]] = {}
    all_names: list[str] = []

    for pkg in packages:
        if not isinstance(pkg, dict):
            raise TypeError("invalid package entry in cargo metadata")
        name = str(pkg.get("name", ""))
        version = str(pkg.get("version", ""))
        manifest_path = str(pkg.get("manifest_path", ""))
        all_names.append(name)

        tokens = set(re.split(r"[-_]", name.lower()))
        forbidden_hits = sorted(tokens & FORBIDDEN_PRODUCER_TOKENS)
        if forbidden_hits:
            raise RuntimeError(
                f"forbidden producer engine dependency resolved: {name} ({forbidden_hits})"
            )

        pkg_dir = _package_source_path(manifest_path)
        if pkg_dir.is_relative_to(resolved_repo):
            raise RuntimeError(
                f"isolated consumer resolved package inside source repository: {name} at {pkg_dir}"
            )

        if name in REQUIRED_CRATES:
            if version != expected_version:
                raise RuntimeError(
                    f"crate {name} version {version} != expected {expected_version}"
                )
            if not pkg_dir.is_relative_to(resolved_unpacked):
                raise RuntimeError(
                    f"required crate {name} did not resolve to unpacked artifacts: {pkg_dir}"
                )
            seen_required[name] = {
                "version": version,
                "manifest_path": str(pkg_dir / "Cargo.toml"),
            }

    missing = sorted(set(REQUIRED_CRATES) - set(seen_required))
    if missing:
        raise RuntimeError(f"missing required unpacked crates in graph: {missing}")

    return {
        "required_crates": seen_required,
        "total_resolved_packages": len(all_names),
        "resolved_package_names": sorted(all_names),
    }


def parse_cargo_test_counts(output: str) -> dict[str, int]:
    """Parse Cargo test summaries and require strictly nonzero passed tests."""
    matches = _TEST_SUMMARY_RE.findall(output)
    if not matches:
        raise RuntimeError("cargo test output contained no test result summary")
    total_passed = 0
    total_failed = 0
    for passed_text, failed_text in matches:
        total_passed += int(passed_text)
        total_failed += int(failed_text)
    if total_failed > 0:
        raise RuntimeError(
            f"isolated consumer tests reported {total_failed} failure(s)"
        )
    if total_passed == 0:
        raise RuntimeError("isolated consumer executed zero passing tests")
    return {"passed": total_passed, "failed": total_failed}


def _rewrite_consumer_manifest(
    consumer_dir: Path, unpacked_dirs: dict[str, Path]
) -> None:
    toml_path = consumer_dir / "Cargo.toml"
    text = toml_path.read_text(encoding="utf-8")
    for crate_name in ("voiage-domain", "voiage-numerics"):
        crate_dir = unpacked_dirs[crate_name].as_posix()
        text = re.sub(
            rf'^{re.escape(crate_name)}\s*=\s*"=[^"]+"\s*$',
            f'{crate_name} = {{ path = "{crate_dir}" }}',
            text,
            flags=re.MULTILINE,
        )
    patch_lines = [
        "",
        "[patch.crates-io]",
        *(
            f'{crate_name} = {{ path = "{unpacked_dirs[crate_name].as_posix()}" }}'
            for crate_name in REQUIRED_CRATES
        ),
    ]
    toml_path.write_text(
        text.rstrip() + "\n" + "\n".join(patch_lines) + "\n", encoding="utf-8"
    )


def _resolve_tool(name: str) -> str:
    """Resolve an executable path or fail explicitly if unavailable."""
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"required executable not found on PATH: {name}")
    return resolved


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(  # noqa: S603 - fixed argv with resolved executable
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def run_isolated_consumer_check(
    repo_root: Path,
    *,
    work_dir: Path | None = None,
    keep_work_dir: bool = False,
) -> dict[str, Any]:
    """Package local crates, unpack them outside checkout, audit graph, and run tests."""
    repo_root = repo_root.resolve()
    cargo_bin = _resolve_tool("cargo")
    git_bin = _resolve_tool("git")
    workspace_manifest = repo_root / "rust" / "Cargo.toml"
    workspace_data = tomllib.loads(workspace_manifest.read_text(encoding="utf-8"))
    expected_version = str(workspace_data["workspace"]["package"]["version"])

    git_proc = subprocess.run(  # noqa: S603 - fixed argv with resolved git executable
        [git_bin, "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    git_head = git_proc.stdout.strip() if git_proc.returncode == 0 else "unknown"

    managed_tmp: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        base_tmp = os.environ.get("TMPDIR")
        managed_tmp = tempfile.TemporaryDirectory(
            prefix="voiage-o08-consumer-", dir=base_tmp
        )
        work_path = Path(managed_tmp.name).resolve()
    else:
        work_path = work_dir.resolve()
        work_path.mkdir(parents=True, exist_ok=True)

    try:
        target_dir = work_path / "cargo-target"
        unpacked_root = work_path / "unpacked"
        consumer_dir = work_path / "consumer"
        env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))

        package_cmd = [
            cargo_bin,
            "package",
            "--locked",
            "--no-verify",
            "--allow-dirty",
            "--manifest-path",
            str(workspace_manifest),
        ]
        for crate_name in REQUIRED_CRATES:
            package_cmd.extend(["-p", crate_name])
        _run(package_cmd, cwd=repo_root, env=env)

        crate_artifacts: list[dict[str, Any]] = []
        unpacked_dirs: dict[str, Path] = {}
        for crate_name in REQUIRED_CRATES:
            crate_file = (
                target_dir / "package" / f"{crate_name}-{expected_version}.crate"
            )
            if not crate_file.is_file():
                raise RuntimeError(f"expected packaged crate missing: {crate_file}")
            unpacked_dir = safe_unpack_crate(crate_file, unpacked_root)
            unpacked_dirs[crate_name] = unpacked_dir
            crate_artifacts.append(
                {
                    "name": crate_name,
                    "version": expected_version,
                    "crate_file": crate_file.name,
                    "sha256": sha256_file(crate_file),
                }
            )

        fixture_src = repo_root / "tests" / "fixtures" / "simulation_outcome_consumer"
        shutil.copytree(fixture_src, consumer_dir)
        shutil.copy2(repo_root / "rust" / "Cargo.lock", consumer_dir / "Cargo.lock")
        _rewrite_consumer_manifest(consumer_dir, unpacked_dirs)

        lock_cmd = [
            cargo_bin,
            "generate-lockfile",
            "--offline",
            "--manifest-path",
            str(consumer_dir / "Cargo.toml"),
        ]
        lock_proc = subprocess.run(  # noqa: S603 - fixed argv with resolved cargo executable
            lock_cmd,
            cwd=consumer_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if lock_proc.returncode != 0:
            _run(
                [
                    cargo_bin,
                    "generate-lockfile",
                    "--manifest-path",
                    str(consumer_dir / "Cargo.toml"),
                ],
                cwd=consumer_dir,
                env=env,
            )

        meta_proc = _run(
            [
                cargo_bin,
                "metadata",
                "--locked",
                "--format-version",
                "1",
                "--manifest-path",
                str(consumer_dir / "Cargo.toml"),
            ],
            cwd=consumer_dir,
            env=env,
        )
        graph_summary = inspect_dependency_graph(
            json.loads(meta_proc.stdout),
            repo_root=repo_root,
            unpacked_root=unpacked_root,
            expected_version=expected_version,
        )

        test_proc = _run(
            [
                cargo_bin,
                "test",
                "--locked",
                "--manifest-path",
                str(consumer_dir / "Cargo.toml"),
            ],
            cwd=consumer_dir,
            env=env,
        )
        counts = parse_cargo_test_counts(test_proc.stdout + "\n" + test_proc.stderr)

        fixture_hashes = {
            path.relative_to(repo_root).as_posix(): sha256_file(path)
            for path in sorted(fixture_src.rglob("*"))
            if path.is_file()
        }
        return {
            "git_head": git_head,
            "workspace_version": expected_version,
            "crate_artifacts": crate_artifacts,
            "fixture_hashes": fixture_hashes,
            "graph_summary": graph_summary,
            "test_counts": counts,
            "synthetic_fixtures_only": True,
            "real_producer_qualified": False,
        }
    finally:
        if managed_tmp is not None and not keep_work_dir:
            managed_tmp.cleanup()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint emitting the isolated consumer verification receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to voiage repository root.",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    receipt = run_isolated_consumer_check(
        args.repo_root,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
