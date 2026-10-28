"""Reject unreviewed legacy tabular references in every tracked text file."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

PATTERN = re.compile(r"\bpandas\b|\bxarray\b|\bpd\.|\bxr\.|itertuples")
CATEGORIES = {
    "migration_guide",
    "translation_reference",
    "producer_compatibility",
    "frozen_historical_evidence",
}
DEFAULT_POLICY = "conductor/tracks/pandas_free_io_20261026/reference-policy.json"


def digest(data: bytes) -> str:
    """Hash canonical checkout text, tolerating Git's platform newline conversion."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def check_files(root: Path, paths: list[str], policy: dict[str, Any]) -> dict[str, Any]:
    """Check exact file or reference-line fingerprints; never use path globs."""
    entries = policy.get("files", {})
    errors: list[str] = []
    findings: list[dict[str, Any]] = []
    scanned = 0
    observed: set[str] = set()
    for name in sorted(set(paths)):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            errors.append(f"Unsafe tracked path: {name}")
            continue
        if not path.is_file():
            errors.append(f"Missing tracked file: {name}")
            continue
        raw = path.read_bytes()
        if b"\0" in raw[:8000]:
            continue
        scanned += 1
        lines = raw.decode("utf-8", errors="replace").splitlines()
        hits = [
            (number, text)
            for number, text in enumerate(lines, 1)
            if PATTERN.search(text)
        ]
        entry = entries.get(name)
        if entry is not None:
            observed.add(name)
            if entry.get("category") not in CATEGORIES or not entry.get("reason"):
                errors.append(f"Unclassified permission: {name}")
                continue
            if "sha256" in entry:
                if (
                    entry["category"] != "frozen_historical_evidence"
                    or digest(raw) != entry["sha256"]
                ):
                    errors.append(f"Frozen evidence changed: {name}")
                continue
            actual = Counter(digest(text.encode()) for _, text in hits)
            expected = Counter(
                {item["sha256"]: item["count"] for item in entry.get("lines", [])}
            )
            if actual != expected:
                errors.append(f"Reference fingerprint drift: {name}")
        elif hits:
            findings.extend({"path": name, "line": n, "text": text} for n, text in hits)
    errors.extend(
        f"Unused or missing permission: {name}"
        for name in sorted(set(entries) - observed)
    )
    if not scanned:
        errors.append("No tracked text files scanned")
    return {
        "passed": not findings and not errors,
        "scanned_files": scanned,
        "unapproved": findings,
        "errors": errors,
    }


def check_repository(root: Path, policy_path: Path | None = None) -> dict[str, Any]:
    """Inspect worktree contents for the complete Git-tracked inventory."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required for the complete inventory")
    result = subprocess.run(  # noqa: S603 - fixed read-only Git command and selected root
        [git, "-C", str(root), "ls-files", "--stage", "-z"],
        capture_output=True,
        check=True,
    )
    paths = [
        item.split(b"\t", 1)[1].decode()
        for item in result.stdout.split(b"\0")
        if item and not item.startswith(b"160000 ")
    ]
    policy = json.loads((policy_path or root / DEFAULT_POLICY).read_text())
    return check_files(root, paths, policy)


def main() -> int:
    """Run the repository gate without refreshing permissions automatically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = check_repository(args.root.resolve())
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
