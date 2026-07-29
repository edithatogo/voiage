#!/usr/bin/env python3
"""Detect ecosystem drift and emit non-applying, human-reviewed proposals."""

from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
from typing import Any

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"
DEFAULT_BASELINE = LANDSCAPE / "ecosystem-drift-baseline.json"
CATEGORIES = (
    "software-registry",
    "toolchains",
    "lockfiles",
    "github-actions",
    "documentation-plugin",
)
LOCKFILES = (
    "uv.lock",
    "pixi.lock",
    "rust/Cargo.lock",
    "rust/fuzz/Cargo.lock",
    "docs/astro-site/pnpm-lock.yaml",
)
ACTION_PATTERN = re.compile(r"^\s*uses:\s*([^#\s]+)", re.MULTILINE)


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _canonical_digest(value: object) -> str:
    return _digest_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _software_registry() -> dict[str, object]:
    registry = json.loads((LANDSCAPE / "registry.json").read_text(encoding="utf-8"))
    return {
        "registry_digest": _digest_file(LANDSCAPE / "registry.json"),
        "method_registry_digest": _digest_file(LANDSCAPE / "methods.json"),
        "searched_on": registry["searched_on"],
        "review_due": registry["review_due"],
        "tool_count": len(registry["tools"]),
    }


def _toolchains() -> dict[str, object]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package = json.loads(
        (ROOT / "docs" / "astro-site" / "package.json").read_text(encoding="utf-8")
    )
    cargo = tomllib.loads((ROOT / "rust" / "Cargo.toml").read_text(encoding="utf-8"))
    return {
        "python": pyproject["project"]["requires-python"],
        "rust": cargo["workspace"]["package"]["rust-version"],
        "node_package_manager": package["packageManager"],
        "pyproject_digest": _digest_file(ROOT / "pyproject.toml"),
        "cargo_workspace_digest": _digest_file(ROOT / "rust" / "Cargo.toml"),
        "docs_package_digest": _digest_file(
            ROOT / "docs" / "astro-site" / "package.json"
        ),
    }


def _lockfiles() -> dict[str, str]:
    return {
        relative: _digest_file(ROOT / relative)
        for relative in LOCKFILES
        if (ROOT / relative).is_file()
    }


def _github_actions() -> dict[str, object]:
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    uses = sorted(
        {
            match
            for workflow in workflows
            for match in ACTION_PATTERN.findall(workflow.read_text(encoding="utf-8"))
        }
    )
    combined = b"".join(
        str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes()
        for path in workflows
    )
    return {
        "workflow_count": len(workflows),
        "workflow_digest": _digest_bytes(combined),
        "action_references": uses,
        "mutable_action_references": [
            item
            for item in uses
            if not item.startswith("./") and not re.search(r"@[0-9a-f]{40}$", item)
        ],
    }


def _documentation_plugin() -> dict[str, str]:
    path = ".repo-tools/astro-polyglot"
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to inspect the documentation gitlink")
    result = subprocess.run(  # noqa: S603 - resolved executable and fixed arguments
        [git, "ls-files", "--stage", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    fields = result.stdout.split()
    if len(fields) < 4 or fields[0] != "160000":
        raise RuntimeError(f"{path} must be a tracked gitlink")
    url = subprocess.run(  # noqa: S603 - resolved executable and fixed arguments
        [
            git,
            "config",
            "-f",
            ".gitmodules",
            "--get",
            "submodule..repo-tools/astro-polyglot.url",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "path": path,
        "gitlink_commit": fields[1],
        "url": url,
    }


def snapshot() -> dict[str, object]:
    """Collect every locally observable, reviewable drift frontier."""
    return {
        "software-registry": _software_registry(),
        "toolchains": _toolchains(),
        "lockfiles": _lockfiles(),
        "github-actions": _github_actions(),
        "documentation-plugin": _documentation_plugin(),
    }


def proposal(
    baseline: dict[str, Any], current: dict[str, object], observed_on: date
) -> dict[str, object]:
    """Build a non-applying proposal from a reviewed baseline."""
    baseline_snapshot = baseline["snapshot"]
    proposals = [
        {
            "category": category,
            "baseline": baseline_snapshot.get(category),
            "observed": current.get(category),
            "required_action": "open-reviewed-change",
            "review_required": True,
        }
        for category in CATEGORIES
        if baseline_snapshot.get(category) != current.get(category)
    ]
    return {
        "schema_version": "1.0.0",
        "observed_on": observed_on.isoformat(),
        "baseline_digest": _canonical_digest(baseline_snapshot),
        "current_digest": _canonical_digest(current),
        "has_drift": bool(proposals),
        "auto_apply": False,
        "scientific_disposition_allowed": False,
        "proposals": proposals,
    }


def main() -> int:
    """Write a baseline, check it, or emit a bounded drift proposal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = snapshot()
    if args.write_baseline:
        args.baseline.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "reviewed_on": args.as_of.isoformat(),
                    "snapshot": current,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.baseline}")
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    report = proposal(baseline, current, args.as_of)
    if args.check:
        if report["has_drift"]:
            print(
                "ecosystem drift detected; generate and review a proposal",
                file=sys.stderr,
            )
            return 1
        print("ecosystem drift baseline is current")
        return 0
    if args.output is None:
        parser.error("--output is required unless --check or --write-baseline is used")
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
