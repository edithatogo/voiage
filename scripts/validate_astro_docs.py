#!/usr/bin/env python3
"""Validate repository-owned links in the Astro documentation tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
GITHUB_BLOB = re.compile(
    r"^https://github\.com/edithatogo/voiage/blob/[^/]+/(?P<path>.+)$"
)


@dataclass(frozen=True)
class Finding:
    """A broken repository-owned documentation reference."""

    path: str
    target: str
    message: str


def _route_exists(content_root: Path, route: str) -> bool:
    route = route.strip("/")
    if not route:
        return (content_root / "index.mdx").is_file()
    candidates = (
        content_root / route / "index.mdx",
        content_root / route / "index.md",
        content_root / f"{route}.mdx",
        content_root / f"{route}.md",
    )
    return any(candidate.is_file() for candidate in candidates)


def validate(repo_root: Path) -> list[Finding]:
    """Return broken internal or repository-backed documentation links."""
    content_root = repo_root / "docs/astro-site/src/content/docs"
    findings: list[Finding] = []
    for document in sorted(content_root.rglob("*.md*")):
        relative = document.relative_to(repo_root).as_posix()
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = unquote(raw_target)
            parsed = urlparse(target)
            if target.startswith("/"):
                route = parsed.path
                if not _route_exists(content_root, route):
                    findings.append(
                        Finding(relative, target, "Astro route does not exist")
                    )
                continue
            github_match = GITHUB_BLOB.match(target)
            if github_match:
                local_path = repo_root / github_match.group("path")
                if not local_path.exists():
                    findings.append(
                        Finding(
                            relative,
                            target,
                            "GitHub-backed repository path does not exist",
                        )
                    )
    return findings


def main() -> int:
    """Validate the Astro documentation tree from the requested repository."""
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    findings = validate(args.repo.resolve())
    if findings:
        for finding in findings:
            print(
                f"{finding.path}: {finding.target}: {finding.message}",
                file=sys.stderr,
            )
        return 1
    print("Astro documentation links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
