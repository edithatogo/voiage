#!/usr/bin/env python3
"""Validate and execute the bounded example smoke manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "specs/examples/smoke-manifest-v1.json"


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return manifest coverage and command-policy errors."""
    discovered = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "examples").rglob("*")
        if path.suffix in {".py", ".ipynb"}
    }
    entries = manifest.get("entries", [])
    paths = [entry.get("path") for entry in entries]
    errors: list[str] = []
    if len(paths) != len(set(paths)):
        errors.append("duplicate example paths")
    if set(paths) != discovered:
        errors.append("manifest does not exactly cover discovered examples")
    for entry in entries:
        if entry.get("disposition") == "execute":
            command = entry.get("command")
            if not isinstance(command, list) or command[:3] != [
                "uv",
                "run",
                "--no-sync",
            ]:
                errors.append(f"{entry.get('path')}: unsafe or synchronizing command")
        elif entry.get("disposition") == "quarantine":
            if not entry.get("reason"):
                errors.append(f"{entry.get('path')}: missing quarantine reason")
        else:
            errors.append(f"{entry.get('path')}: invalid disposition")
    return errors


def run(manifest_path: Path = DEFAULT_MANIFEST) -> list[dict[str, object]]:
    """Execute selected examples without a shell or dependency synchronization."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest)
    if errors:
        return [
            {"path": "manifest", "passed": False, "error": error} for error in errors
        ]
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "UV_OFFLINE": "1",
        }
    )
    results: list[dict[str, object]] = []
    for entry in manifest["entries"]:
        if entry["disposition"] != "execute":
            continue
        try:
            completed = subprocess.run(  # noqa: S603 - manifest commands are allowlisted above
                entry["command"],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            results.append({"path": entry["path"], "passed": False, "error": "timeout"})
            continue
        results.append(
            {
                "path": entry["path"],
                "passed": completed.returncode == 0,
                "returncode": completed.returncode,
            }
        )
    return results


def main() -> int:
    """Validate or run the smoke manifest and emit JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.validate_only:
        errors = validate_manifest(manifest)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    results = run(args.manifest)
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
