"""Replay a declared paper environment in an isolated exact-commit checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git") or "/usr/bin/git"
UV = shutil.which("uv") or "/usr/local/bin/uv"


def bound_file(root: Path, descriptor: dict[str, str]) -> Path:
    """Resolve a repository-relative regular file and verify its declared digest."""
    relative = Path(descriptor["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Declared artifact must remain inside the repository")
    if not path.is_file():
        raise ValueError(f"Missing declared artifact: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"Declared artifact digest mismatch: {relative}")
    return path


def verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Validate the declared artifacts and exact Git source before execution."""
    if manifest["schema_version"] != "voiage.paper.reproduction.v2":
        raise ValueError("An explicit v2 replay descriptor is required")
    if manifest["generator"] != "scripts/generate_paper_health_example.py":
        raise ValueError("Unexpected replay generator")
    if manifest["replay_extras"] != ["plotting"]:
        raise ValueError("Replay requires the declared plotting extra")
    source = manifest["source_reference"]
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Replay source must be an immutable full commit ID")
    for name in ("historical_receipt", "lockfile", "project_file"):
        bound_file(root, manifest[name])
    resolved = subprocess.check_output(  # noqa: S603 - fixed Git arguments and validated commit ID
        [GIT, "rev-parse", "--verify", f"{source}^{{commit}}"], cwd=root, text=True
    ).strip()
    if resolved != source:
        raise ValueError("Replay source does not resolve to the declared commit")
    for name, source_path in (
        ("lockfile", "uv.lock"),
        ("project_file", "pyproject.toml"),
    ):
        original = subprocess.check_output(  # noqa: S603 - fixed Git arguments and validated commit ID
            [GIT, "show", f"{source}:{source_path}"], cwd=root
        )
        if original != bound_file(root, manifest[name]).read_bytes():
            raise ValueError(f"Declared {name} does not match the exact source commit")
    receipt = bound_file(root, manifest["historical_receipt"]).read_bytes()
    source_receipt = subprocess.check_output(  # noqa: S603 - immutable validated Git source
        [GIT, "show", f"{source}:paper/reproduction-manifest.json"], cwd=root
    )
    if receipt != source_receipt:
        raise ValueError("Historical receipt does not match the selected source")
    original = json.loads(receipt)
    for field in (
        "synthetic_data",
        "generator",
        "supported_python_versions",
        "seeds",
        "inputs",
        "outputs",
        "rendering_note",
    ):
        if manifest[field] != original[field]:
            raise ValueError(
                f"Replay scientific contract differs from preserved receipt: {field}"
            )
    for output in manifest["outputs"]:
        bound_file(root, output)


def replay(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Run frozen verification without using or modifying the current environment."""
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    verify_manifest(root, manifest)
    source = manifest["source_reference"]
    with tempfile.TemporaryDirectory(prefix="voiage-paper-replay-") as temporary:
        checkout = Path(temporary) / "source"
        # Git performs extraction; no archive member or caller-provided shell text is executed.
        subprocess.run(  # noqa: S603 - argument arrays, validated source, isolated project
            [
                GIT,
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                "--",
                str(root),
                str(checkout),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(  # noqa: S603 - argument arrays, validated source, isolated project
            [GIT, "checkout", "--detach", source],
            cwd=checkout,
            check=True,
            capture_output=True,
        )
        for name, local in (
            ("lockfile", "uv.lock"),
            ("project_file", "pyproject.toml"),
        ):
            if (checkout / local).read_bytes() != bound_file(
                root, manifest[name]
            ).read_bytes():
                raise ValueError(f"Isolated {name} differs from declared environment")
        env = os.environ.copy()
        for key in (
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "UV_PROJECT",
            "UV_PROJECT_ENVIRONMENT",
            "UV_WORKING_DIRECTORY",
        ):
            env.pop(key, None)
        env["MPLBACKEND"] = "Agg"
        command = [
            UV,
            "run",
            "--frozen",
            "--extra",
            manifest["replay_extras"][0],
            "--project",
            str(checkout),
            "python",
            "scripts/generate_paper_health_example.py",
            "--verify-tracked",
        ]
        completed = subprocess.run(  # noqa: S603 - argument arrays, validated source, isolated project
            command, cwd=checkout, env=env, check=True, capture_output=True
        )
        runtime_query = (
            "import json, platform; "
            "print(json.dumps({'python_version': platform.python_version(), "
            "'python_implementation': platform.python_implementation(), "
            "'platform': platform.platform(), 'machine': platform.machine()}))"
        )
        runtime_result = subprocess.run(  # noqa: S603 - fixed query in the same isolated project
            [*command[:-2], "-c", runtime_query],
            cwd=checkout,
            env=env,
            check=True,
            capture_output=True,
        )
        runtime = json.loads(runtime_result.stdout)
        # Bind successful replay to both the selected source and the current declared outputs.
        for output in manifest["outputs"]:
            bound_file(checkout, output)
        for name, local in (
            ("lockfile", "uv.lock"),
            ("project_file", "pyproject.toml"),
        ):
            if (
                hashlib.sha256((checkout / local).read_bytes()).hexdigest()
                != manifest[name]["sha256"]
            ):
                raise ValueError(f"Replay changed its {name}")
        return {
            "schema_version": "voiage.paper.replay-result.v1",
            "status": "verified",
            "scope": "new exact-commit replay; not historical source attestation",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "source_reference": source,
            "runtime": runtime,
            "replay_extras": manifest["replay_extras"],
            "lockfile_sha256": manifest["lockfile"]["sha256"],
            "project_file_sha256": manifest["project_file"]["sha256"],
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            "outputs": manifest["outputs"],
        }


def main() -> None:
    """Print a digest-bound receipt only after the frozen replay succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("paper/reproduction-manifest.json")
    )
    args = parser.parse_args()
    try:
        result = replay(ROOT, ROOT / args.manifest)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(
                error.stderr.decode()
                if isinstance(error.stderr, bytes)
                else error.stderr,
                file=sys.stderr,
            )
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
