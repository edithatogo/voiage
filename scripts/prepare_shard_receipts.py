"""Prepare shard receipts and independently derive aggregate identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

FEATURES = "ci,dev,producer-tests"


def digest(path: Path) -> str:
    """Hash an existing, bounded artifact."""
    if not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("missing or oversized artifact")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(repo: Path) -> dict[str, str]:
    """Independently bind a clean checkout and current interpreter."""
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=repo,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],  # noqa: S607
        cwd=repo,
        text=True,
    )
    if status:
        raise ValueError("checkout must be clean")
    return {
        "source_head": head,
        "lock_digest": digest(repo / "uv.lock"),
        "interpreter": f"{platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "features": FEATURES,
        "coverage_config_digest": digest(repo / "pyproject.toml"),
    }


def execute(args: argparse.Namespace) -> None:
    """Write producer evidence or a fail-closed aggregate manifest."""
    repo = Path(args.repo).resolve(strict=True)
    root = Path(args.root).resolve(strict=True)
    expected = identity(repo)
    if not 1 <= args.expected_shards <= 128:
        raise ValueError("invalid shard count")
    if args.mode == "prepare":
        if not 0 <= args.index < args.expected_shards:
            raise ValueError("invalid shard index")
        directory = root / f"shard-{args.index}"
        receipt = {
            "index": args.index,
            "identity": expected,
            "coverage_sha256": digest(directory / "coverage.json"),
            "native_sha256": digest(directory / "coverage.data"),
        }
        destination = directory / "receipt.json"
        if destination.exists():
            raise ValueError("receipt already exists")
        destination.write_text(json.dumps(receipt), encoding="utf-8")
        return
    if args.matrix_result != "success":
        raise ValueError("matrix did not succeed")
    output = root / "manifest.json"
    identity_output = root / "expected-identity.json"
    if output.exists() or identity_output.exists():
        raise ValueError("aggregate output already exists")
    shards = []
    for index in range(args.expected_shards):
        directory = root / f"shard-{index}"
        receipt_path = directory / "receipt.json"
        if receipt_path.stat().st_size > 1024 * 1024:
            raise ValueError("oversized receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(receipt, dict)
            or receipt.get("index") != index
            or receipt.get("identity") != expected
        ):
            raise ValueError("receipt identity or index mismatch")
        report = directory / "coverage.json"
        native = directory / "coverage.data"
        if receipt.get("coverage_sha256") != digest(report) or receipt.get(
            "native_sha256"
        ) != digest(native):
            raise ValueError("artifact digest mismatch")
        shards.append(
            {
                "index": index,
                "conclusion": "success",
                "exit_status": 0,
                "identity": expected,
                "coverage": str(report),
                "coverage_sha256": receipt["coverage_sha256"],
                "native_data": str(native),
                "native_sha256": receipt["native_sha256"],
            }
        )
    identity_output.write_text(json.dumps(expected), encoding="utf-8")
    output.write_text(
        json.dumps(
            {
                "expected_shards": args.expected_shards,
                "identity": expected,
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    """Prepare or collect shard receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--expected-shards", type=int, default=2)
    parser.add_argument("--matrix-result")
    try:
        execute(parser.parse_args())
    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as error:
        print(f"shard receipt preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
