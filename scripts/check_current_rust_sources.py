"""Authenticate the dated Rust source and its stage-zero manifest binding."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tomllib


def validate(root: Path, archive: Path, bootstrap_manifest: Path) -> dict[str, object]:
    """Check supplied source metadata without extracting or executing upstream code."""
    packet = json.loads(
        (root / "packaging/current/rust-provider-sources.json").read_text()
    )
    nightly, bootstrap = packet["nightly"], packet["bootstrap"]
    if packet["native_build_qualified"] is not False:
        raise ValueError("Source qualification cannot assert a native build")
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != nightly["sha256"]:
        raise ValueError("Rust source checksum mismatch")
    stage0 = None
    with tarfile.open(archive, "r|xz", stream=True) as source:
        for member in source:
            if member.name.split("/", 1)[-1] != "src/stage0":
                continue
            if stage0 is not None or not member.isfile() or member.size > 2 * 1024**2:
                raise ValueError("Invalid or duplicate stage0 metadata")
            content = source.extractfile(member)
            if content is None:
                raise ValueError("Unreadable stage0 metadata")
            stage0 = content.read().decode()
    if stage0 is None:
        raise ValueError("Missing stage0 metadata")
    fields = dict(
        line.split("=", 1)
        for line in stage0.splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    manifest_bytes = bootstrap_manifest.read_bytes()
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != fields["compiler_channel_manifest_hash"]
    ):
        raise ValueError("Bootstrap manifest checksum mismatch")
    manifest = tomllib.loads(manifest_bytes.decode())
    date = fields["compiler_date"]
    if (
        bootstrap["version"] != fields["compiler_version"] + "-" + date
        or nightly["bootstrap"] != bootstrap["version"]
        or bootstrap["commit"] != fields["compiler_git_commit_hash"]
        or manifest["date"] != date
        or manifest["pkg"]["rust"]["git_commit_hash"] != bootstrap["commit"]
        or manifest["pkg"]["rust"]["version"] != bootstrap["compiler_version"]
    ):
        raise ValueError("Source bootstrap identity mismatch")
    expected_platforms = {
        system: {"aarch64", "x86_64"} for system in ("linux", "darwin")
    }
    if {
        system: set(hashes) for system, hashes in bootstrap["hashes"].items()
    } != expected_platforms:
        raise ValueError("All four reviewed bootstrap platforms are required")
    for system, hashes in bootstrap["hashes"].items():
        platform = {"linux": "unknown-linux-gnu", "darwin": "apple-darwin"}[system]
        for arch, digest in hashes.items():
            target = manifest["pkg"]["rust"]["target"][f"{arch}-{platform}"]
            url = f"https://static.rust-lang.org/dist/{date}/rust-beta-{arch}-{platform}.tar.xz"
            if (
                not target["available"]
                or target["xz_hash"] != digest
                or target["xz_url"] != url
            ):
                raise ValueError("Bootstrap platform checksum or URL mismatch")
    packages = root / "packaging/current/spack/packages"
    compiler = ast.parse((packages / "rust/package.py").read_text())
    matching = [
        call
        for call in ast.walk(compiler)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "version"
        and call.args
        and ast.literal_eval(call.args[0]) == nightly["version"]
    ]
    if len(matching) != 1 or {
        item.arg: ast.literal_eval(item.value) for item in matching[0].keywords
    } != {"sha256": checksum, "url": nightly["url"]}:
        raise ValueError("Compiler recipe source drift")
    sdk = ast.parse((packages / "rust-bootstrap/package.py").read_text())
    hashes = [
        ast.literal_eval(node.value)
        for node in ast.walk(sdk)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "dated_beta" for t in node.targets)
    ]
    if hashes != [bootstrap["hashes"]]:
        raise ValueError("Bootstrap recipe checksum drift")
    return {
        "source_archive_sha256": checksum,
        "bootstrap_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "stage0_binding_qualified": True,
        "bootstrap_platforms_checked": sum(map(len, bootstrap["hashes"].values())),
        "bootstrap_binaries_verified": False,
        "native_build_executed": False,
    }


def main() -> int:
    """Validate local authenticated inputs; never download or execute code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("bootstrap_manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 15, 1):
        parser.error("Rust source qualification requires Python 3.15.1")
    try:
        report = validate(args.root, args.archive, args.bootstrap_manifest)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SyntaxError,
        tarfile.TarError,
    ) as error:
        print(json.dumps({"stage0_binding_qualified": False, "error": str(error)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
