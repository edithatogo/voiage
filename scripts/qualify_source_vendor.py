"""Authenticate HPC inputs and replay Cargo metadata against a verified vendor tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tomllib
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_UNPACKED_BYTES = 16 * 1024**3


def compiler_vendor_contract(source: Path, manifests: list[Path]) -> dict[str, Any]:
    """Require the upstream dist/notice workspace closure before compilation."""
    module = source / "src/bootstrap/src/core/build_steps/vendor.rs"
    text = module.read_text(encoding="utf-8")
    function = re.search(
        r"^pub fn default_paths_to_vendor\b.*?^}", text, re.MULTILINE | re.DOTALL
    )
    paths = (
        re.findall(r'"([^"\n]+/Cargo\.toml)"', function.group())
        if function is not None
        else []
    )
    if not paths:
        raise ValueError("Unrecognized upstream compiler vendor contract")
    required = {"Cargo.toml", *paths}
    configured = {str(path.relative_to(source)) for path in manifests}
    if missing := required - configured:
        raise ValueError(
            f"Missing upstream compiler vendor workspaces: {sorted(missing)}"
        )
    return {
        "upstream_module": str(module.relative_to(source)),
        "upstream_module_sha256": digest(module),
        "required_manifests": sorted(required),
    }


def digest(path: Path) -> str:
    """Hash bytes without loading an archive into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def workspace_locks(
    source: Path,
    manifests: list[Path],
    cargo: Path,
    env: dict[str, str],
    output: Path,
) -> list[Path]:
    """Resolve shared workspace locks through Cargo without changing their bytes."""
    locks = []
    for index, manifest in enumerate(manifests):
        receipt = output / f"workspace-locate-{index}.json"
        run(
            [
                str(cargo),
                "locate-project",
                "--workspace",
                "--manifest-path",
                str(manifest),
            ],
            source,
            dict(env, CARGO_NET_OFFLINE="true"),
            receipt,
        )
        root = Path(json.loads(receipt.read_text(encoding="utf-8"))["root"]).resolve()
        lock = root.with_name("Cargo.lock")
        if not root.is_relative_to(source.resolve()) or not lock.is_file():
            raise ValueError(
                f"Missing or external authenticated workspace lock: {root}"
            )
        if lock not in locks:
            locks.append(lock)
    return locks


def fetch(source: dict[str, str], cache: Path, *, offline: bool = False) -> Path:
    """Authenticate cached bytes before use and never refetch corrupt inputs."""
    filename = source.get("filename") or source["url"].rsplit("/", 1)[-1]
    if Path(filename).name != filename or not re.fullmatch(
        r"[0-9a-f]{64}", source["sha256"]
    ):
        raise ValueError("Invalid source identity")
    path = cache / filename
    if cache.is_symlink() or path.is_symlink():
        raise ValueError("Shadowed source path")
    if not path.exists():
        if offline:
            raise ValueError("Forbidden fetch: source is absent")
        if urlsplit(source["url"]).scheme != "https":
            raise ValueError("Source transport must be HTTPS")
        cache.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        stream = temporary.open("xb")
        try:
            with stream as output, urlopen(source["url"], timeout=60) as response:  # noqa: S310 - HTTPS checked above
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Source archive exceeds byte limit")
                    output.write(block)
            if digest(temporary) != source["sha256"]:
                raise ValueError("Source digest mismatch")
            temporary.rename(path)
        finally:
            # This invocation created the file exclusively. Preserve pre-existing
            # partial files by failing before entering this ownership scope.
            temporary.unlink(missing_ok=True)
    if not path.is_file() or digest(path) != source["sha256"]:
        raise ValueError("Source digest mismatch")
    return path


def unpack(archive: Path, destination: Path) -> Path:
    """Use Python's data filter after bounding the authenticated archive inventory."""
    destination.mkdir()
    with tarfile.open(archive) as package:
        members = package.getmembers()
        if (
            len(members) > 500000
            or sum(member.size for member in members) > MAX_UNPACKED_BYTES
        ):
            raise ValueError("Source inventory exceeds limits")
        names: set[str] = set()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.name in names:
                raise ValueError("Unsafe or duplicate archive path")
            names.add(member.name)
        package.extractall(destination, filter="data")
    children = list(destination.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        raise ValueError("Expected one source root")
    return children[0]


def run(command: list[str], cwd: Path, env: dict[str, str], output: Path) -> None:
    """Keep the first failure and its complete command output."""
    with (
        output.open("wb") as stream,
        output.with_suffix(output.suffix + ".stderr").open("wb") as stderr,
    ):
        subprocess.run(  # noqa: S603 - fixed argv with validated SDK executables
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=stderr,
            check=True,
        )


def validate_vendor(vendor: Path, locks: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify registry identities and every checksummed vendor file."""
    expected: dict[tuple[str, str], str | None] = {}
    sources: dict[tuple[str, str], str] = {}
    for lock in locks:
        for package in lock["package"]:
            source = package.get("source", "")
            if source.startswith("registry+"):
                checksum = package["checksum"]
            elif source.startswith("git+"):
                if not re.fullmatch(r"[0-9a-f]{40}", urlsplit(source[4:]).fragment):
                    raise ValueError("Git source must have a full immutable revision")
                checksum = None
            else:
                # A synchronized workspace can build a crate locally while another
                # lock consumes that name/version from a registry. Only external
                # packages belong in the vendor inventory.
                continue
            identity = package["name"], package["version"]
            if identity in expected and (
                expected[identity] != checksum or sources[identity] != source
            ):
                raise ValueError("Conflicting vendor source identities")
            expected[identity] = checksum
            sources[identity] = source
    observed: set[tuple[str, str]] = set()
    records = []
    for directory in sorted(vendor.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Shadowed vendor directory")
        metadata = tomllib.loads(
            (directory / "Cargo.toml").read_text(encoding="utf-8")
        )["package"]
        identity = metadata["name"], metadata["version"]
        checks = json.loads(
            (directory / ".cargo-checksum.json").read_text(encoding="utf-8")
        )
        if identity in observed:
            raise ValueError("Duplicate vendor package identity")
        if identity not in expected or checks["package"] != expected[identity]:
            raise ValueError(f"Vendor package identity mismatch: {identity}")
        files = checks["files"]
        inventory = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Shadowed vendor file")
            if path.is_file():
                inventory.add(path.relative_to(directory).as_posix())
        if inventory != set(files) | {".cargo-checksum.json"}:
            raise ValueError(f"Vendor inventory mismatch: {identity}")
        for relative, checksum in files.items():
            selected = PurePosixPath(relative)
            if selected.is_absolute() or ".." in selected.parts:
                raise ValueError("Unsafe checksum path")
            if digest(directory / relative) != checksum:
                raise ValueError(f"Vendor file digest mismatch: {identity} {relative}")
        observed.add(identity)
        records.append(
            {
                "name": identity[0],
                "version": identity[1],
                "source": sources[identity],
                "package_sha256": checks["package"],
                "files": len(files),
                "checksum_manifest_sha256": digest(directory / ".cargo-checksum.json"),
            }
        )
    if expected.keys() - observed:
        raise ValueError(
            f"Missing vendor packages: {sorted(expected.keys() - observed)}"
        )
    return {
        "packages": records,
        "registry_packages": sum(value is not None for value in expected.values()),
        "git_packages": sum(value is None for value in expected.values()),
    }


def apply_lock_patch(source: Path, patch: dict[str, str], git: str) -> dict[str, Any]:
    """Apply an explicitly frozen sdist lock repair without selecting new crates."""
    lock = source / "Cargo.lock"
    path = Path(patch["path"]).resolve(strict=True)
    if digest(path) != patch["sha256"] or digest(lock) != patch["original_sha256"]:
        raise ValueError("Lock patch input digest mismatch")
    before = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    subprocess.run(  # noqa: S603 - authenticated patch and fixed Git argv
        [git, "apply", "--no-index", "--check", str(path)], cwd=source, check=True
    )
    subprocess.run(  # noqa: S603 - authenticated patch and fixed Git argv
        [git, "apply", "--no-index", str(path)], cwd=source, check=True
    )
    if digest(lock) != patch["qualified_sha256"]:
        raise ValueError("Lock patch result digest mismatch")
    after = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]

    def identities(packages: list[dict[str, Any]]) -> set[tuple[str, ...]]:
        return {
            (p["name"], p["version"], p.get("source", ""), p.get("checksum", ""))
            for p in packages
        }

    if not identities(after) <= identities(before):
        raise ValueError("Lock repair introduced a new package identity")
    return dict(patch, original_packages=len(before), qualified_packages=len(after))


def qualified_tool(path: Path, workspace: Path) -> Path:
    """Exclude source-supplied or PATH-shadowed tool executables."""
    selected = path.resolve(strict=True)
    if selected.is_relative_to(workspace.resolve()) or not os.access(selected, os.X_OK):
        raise ValueError("Unqualified tool path")
    return selected


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch once, vendor locked dependencies and prove the offline replay."""
    repository = Path.cwd()
    entry = json.loads(args.inputs.read_text(encoding="utf-8"))["entries"][args.entry]
    work = args.work.resolve()
    work.mkdir()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git is None:
        raise ValueError("Git is required")
    cargo = qualified_tool(args.cargo, work)
    rustc = qualified_tool(args.rustc, work)
    env = dict(
        os.environ,
        CARGO_NET_RETRY="0",
        CARGO_HOME=str(work / "cargo-home"),
        RUSTC=str(rustc),
        RUSTDOC=str(rustc.with_name("rustdoc")),
        RUSTC_WRAPPER="",
        RUSTC_WORKSPACE_WRAPPER="",
    )
    run(
        [str(cargo), "--version", "--verbose"],
        repository,
        env,
        output / "cargo-version.txt",
    )
    run(
        [str(rustc), "--version", "--verbose"],
        repository,
        env,
        output / "rustc-version.txt",
    )
    version = (output / "rustc-version.txt").read_text(encoding="utf-8")
    if cargo.parent != rustc.parent:
        raise ValueError("Cargo and rustc must come from the same SDK")
    if entry["toolchain"] == "1.99.1":
        if not version.startswith("rustc 1.99.1 "):
            raise ValueError("Expected the qualified stable SDK")
    else:
        nightly = json.loads(
            Path("packaging/current/rust-provider-sources.json").read_text(
                encoding="utf-8"
            )
        )["nightly"]
        if (
            not version.startswith("rustc 1.100.0-nightly ")
            or nightly["commit"] not in version
        ):
            raise ValueError("Expected the exact qualified nightly SDK")
    source_record: dict[str, Any]
    if entry["kind"] == "git":
        revision = entry["revision"]
        run(
            [git, "fetch", "--depth", "1", "origin", revision],
            repository,
            env,
            output / "source-fetch.txt",
        )
        tree = subprocess.check_output(  # noqa: S603 - qualified Git and immutable source revision
            [git, "rev-parse", revision + "^{tree}"], text=True
        ).strip()
        if tree != entry["tree"]:
            raise ValueError("Source Git tree mismatch")
        source = work / "source"
        run(
            [git, "worktree", "add", "--detach", str(source), revision],
            repository,
            env,
            output / "source-checkout.txt",
        )
        source_record = {"revision": revision, "tree": tree}
    else:
        archive = fetch(entry["source"], work / "archives")
        source = unpack(archive, work / "unpacked")
        source_record = dict(
            entry["source"],
            observed_sha256=digest(archive),
            bytes=archive.stat().st_size,
        )
    if "bootstrap" in entry:
        bootstrap = fetch(entry["bootstrap"], work / "archives")
        source_record["bootstrap"] = dict(
            entry["bootstrap"], observed_sha256=digest(bootstrap)
        )
    if "lock_patch" in entry:
        source_record["lock_repair"] = apply_lock_patch(
            source, entry["lock_patch"], git
        )
    if entry["manifests"] == "auto":
        locks = sorted(source.rglob("Cargo.lock"), key=lambda path: len(path.parts))
        if not locks:
            raise ValueError("Missing Cargo lockfile")
        manifests = [locks[0].with_name("Cargo.toml")]
    else:
        manifests = [source / name for name in entry["manifests"]]
    compiler_contract = (
        compiler_vendor_contract(source, manifests) if "bootstrap" in entry else None
    )
    for manifest in manifests:
        if not manifest.is_file():
            raise ValueError(
                f"Missing locked workspace: {manifest.relative_to(source)}"
            )
    lock_paths = workspace_locks(source, manifests, cargo, env, output)
    lock_hashes = {str(path.relative_to(source)): digest(path) for path in lock_paths}
    locks_data = [
        tomllib.loads(path.read_text(encoding="utf-8")) for path in lock_paths
    ]
    online = work / "online"
    online.mkdir()
    vendor = work / "vendor"
    command = [
        str(cargo),
        "vendor",
        "--locked",
        "--versioned-dirs",
        "--manifest-path",
        str(manifests[0]),
    ]
    for manifest in manifests[1:]:
        command.extend(["--sync", str(manifest)])
    command.append(str(vendor))
    with (
        (output / "vendor-config.toml").open("wb") as stdout,
        (output / "vendor-fetch.txt").open("wb") as stderr,
    ):
        subprocess.run(  # noqa: S603 - fixed argv with validated SDK executables
            command, cwd=online, env=env, stdout=stdout, stderr=stderr, check=True
        )
    result = validate_vendor(vendor, locks_data)
    replay = work / "offline"
    (replay / ".cargo").mkdir(parents=True)
    shutil.copyfile(output / "vendor-config.toml", replay / ".cargo/config.toml")
    offline_env = dict(
        env,
        CARGO_HOME=str(work / "empty-cargo-home"),
        CARGO_NET_OFFLINE="true",
        HTTP_PROXY="http://127.0.0.1:9",
        HTTPS_PROXY="http://127.0.0.1:9",
        CARGO_HTTP_PROXY="http://127.0.0.1:9",
    )
    for index, manifest in enumerate(manifests):
        run(
            [
                str(cargo),
                "metadata",
                "--locked",
                "--offline",
                "--all-features",
                "--format-version",
                "1",
                "--manifest-path",
                str(manifest),
            ],
            replay,
            offline_env,
            output / f"offline-metadata-{index}.json",
        )
    for index in range(len(manifests)):
        metadata = json.loads(
            (output / f"offline-metadata-{index}.json").read_text(encoding="utf-8")
        )
        for package in metadata["packages"]:
            location = Path(package["manifest_path"]).resolve()
            expected_root = vendor.resolve() if package["source"] else source.resolve()
            if not location.is_relative_to(expected_root):
                raise ValueError("Offline package escaped its qualified source tree")
    if any(
        digest(path) != lock_hashes[str(path.relative_to(source))]
        for path in lock_paths
    ):
        raise ValueError("Cargo changed an authenticated lockfile")
    return {
        "schema_version": "1.0",
        "entry": args.entry,
        "source": source_record,
        "locks": lock_hashes,
        "vendor": result,
        "compiler_vendor_contract": compiler_contract,
        "offline_metadata_passed": True,
        "native_build_accepted": False,
        "installed_module_accepted": False,
    }


def main() -> int:
    """Retain a terminal receipt even when qualification fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("packaging/current/source-vendor-inputs.json"),
    )
    parser.add_argument("--entry", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = qualify(args)
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
        tarfile.TarError,
    ) as error:
        (args.output / "qualification.json").write_text(
            json.dumps(
                {"entry": args.entry, "passed": False, "error": str(error)}, indent=2
            )
            + "\n"
        )
        raise
    result["passed"] = True
    (args.output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
