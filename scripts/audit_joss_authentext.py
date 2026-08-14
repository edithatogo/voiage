#!/usr/bin/env python3
"""Apply the pinned Authentext core and academic pattern checks to paper.md."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from validate_joss import _body_without_front_matter

ROOT = Path(__file__).resolve().parents[1]
AUTHENTEXT = ROOT / ".repo-tools/authentext"

PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("critical", "collaborative-artifact", r"\b(?:i hope this helps|let me know if)\b"),
    ("critical", "speculative-gap", r"\bbased on available information\b"),
    ("high", "undue-significance", r"\b(?:stands as|a testament to|pivotal)\b"),
    ("high", "promotional", r"\b(?:groundbreaking|showcasing|breathtaking)\b"),
    (
        "high",
        "vague-academic-attribution",
        r"\b(?:studies have shown|research indicates|experts agree)\b",
    ),
    (
        "medium",
        "promotional-abstract",
        r"\b(?:novel methodology|significant contributions|valuable insights)\b",
    ),
    (
        "medium",
        "ai-vocabulary-cluster",
        r"\b(?:delve|intricate|meticulous|vibrant tapestry|evolving landscape)\b",
    ),
    (
        "medium",
        "generic-self-importance",
        r"\b(?:underscores? the importance|highlights? the significance)\b",
    ),
    (
        "low",
        "academic-filler",
        r"\b(?:it is important to note that|it is worth noting that)\b",
    ),
)


def _tool_commit() -> str:
    result = subprocess.run(  # noqa: S603 - fixed local Git inspection
        ["git", "-C", str(AUTHENTEXT), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _mask_literals(line: str) -> str:
    """Mask technical literals that Authentext requires editors to preserve."""
    line = re.sub(r"`[^`]+`", "`LITERAL`", line)
    line = re.sub(r"https?://\S+", "URL", line)
    return re.sub(r"\[@[^\]]+\]", "[CITATION]", line)


def audit(source: Path) -> dict[str, Any]:
    """Return deterministic Authentext findings for the canonical manuscript."""
    text = source.read_text(encoding="utf-8")
    body = _body_without_front_matter(text)
    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        candidate = _mask_literals(line)
        for severity, pattern_id, expression in PATTERNS:
            match = re.search(expression, candidate, re.IGNORECASE)
            if match:
                findings.append(
                    {
                        "severity": severity,
                        "pattern_id": pattern_id,
                        "line": line_number,
                        "match": match.group(0),
                    }
                )
    blocking = [
        finding
        for finding in findings
        if finding["severity"] in {"critical", "high", "medium"}
    ]
    return {
        "schema_version": "voiage.joss-authentext-audit.v1",
        "status": "pass" if not blocking else "blocked",
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(text.encode()).hexdigest(),
        "authentext_commit": _tool_commit(),
        "profiles_consulted": ["core-patterns", "academic"],
        "pattern_coverage": "selected deterministic blocking patterns",
        "interpretation": (
            "This script checks a selected deterministic subset of the consulted "
            "Authentext profiles; full human editorial review remains required."
        ),
        "findings": findings,
    }


def main() -> int:
    """Run the JOSS Authentext audit from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "paper.md")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/joss/authentext.json",
    )
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = audit(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"JOSS Authentext audit: {report['status']} "
        f"({len(report['findings'])} findings)"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
