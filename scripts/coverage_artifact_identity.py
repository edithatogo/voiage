"""Seal and verify coverage bytes against their local execution identity.

The receipt is an integrity check, not a signature or proof of who ran the tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _identity(repo: Path, artifact: Path, features: str) -> dict[str, str]:
    if not features.strip():
        raise ValueError("feature configuration must be declared")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],  # noqa: S607
        cwd=repo,
        text=True,
    )
    if status:
        raise ValueError("source checkout must be clean")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,  # noqa: S607
    ).strip()
    return {
        "schema_version": "1.0.0",
        "source_head": head,
        "lock_digest": _digest(repo / "uv.lock"),
        "interpreter": (
            f"{platform.python_implementation()} {sys.version} "
            f"{sys.implementation.cache_tag} {sysconfig.get_platform()}"
        ),
        "coverage_config_digest": _digest(repo / "pyproject.toml"),
        "feature_configuration": features,
        "artifact_digest": _digest(artifact),
    }


def _verify(receipt: Path, identity: dict[str, str]) -> None:
    if receipt.stat().st_size > 16_384:
        raise ValueError("coverage artifact receipt exceeds size limit")
    retained = json.loads(receipt.read_text(encoding="utf-8"))
    if retained != identity:
        raise ValueError("coverage artifact identity mismatch")


def main() -> int:
    """Seal or verify one coverage identity receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "verify"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--features", required=True)
    args = parser.parse_args()
    try:
        identity = _identity(args.repo.resolve(), args.artifact, args.features)
        if args.mode == "seal":
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(
                json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            _verify(args.receipt, identity)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"coverage artifact identity failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
