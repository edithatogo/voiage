"""Validate the repository-owned JOSS paper package."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

REQUIRED_SECTIONS = (
    "Summary",
    "Statement of need",
    "State of the field",
    "Software design",
    "Research impact statement",
    "AI usage disclosure",
    "Acknowledgements",
    "References",
)
PLACEHOLDER_PATTERNS = (
    "must be updated before",
    "must be confirmed before",
    "remain subject to final",
    "todo",
    "tbd",
)
WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
CITATION_PATTERN = re.compile(r"@([A-Za-z0-9_:-]+)")
BIB_KEY_PATTERN = re.compile(r"@\w+\{\s*([^,\s]+)")


def _body_without_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    parts = text.split("---", maxsplit=2)
    return parts[2] if len(parts) == 3 else text


def validate_joss_package(root: Path) -> list[str]:
    """Return fail-closed findings for the JOSS manuscript and bibliography."""
    findings: list[str] = []
    paper_path = root / "paper.md"
    bibliography_path = root / "paper.bib"
    if not paper_path.is_file():
        return ["paper.md is missing"]
    if not bibliography_path.is_file():
        return ["paper.bib is missing"]

    paper = paper_path.read_text(encoding="utf-8")
    bibliography = bibliography_path.read_text(encoding="utf-8")
    if not paper.startswith("---\n"):
        findings.append("paper.md must begin with YAML metadata")
    findings.extend(
        f"paper.md metadata is missing {metadata_key}"
        for metadata_key in (
            "title:",
            "authors:",
            "affiliations:",
            "date:",
            "bibliography:",
            "repository:",
        )
        if metadata_key not in paper.split("---", maxsplit=2)[1]
    )

    body = _body_without_front_matter(paper)
    findings.extend(
        f"required JOSS section is missing: {section}"
        for section in REQUIRED_SECTIONS
        if not re.search(rf"^# {re.escape(section)}\s*$", body, re.MULTILINE)
    )

    words = WORD_PATTERN.findall(body)
    if not 750 <= len(words) <= 1750:
        findings.append(
            f"JOSS paper body has {len(words)} words; expected 750 through 1750"
        )

    lowered = body.casefold()
    findings.extend(
        f"submission placeholder remains: {placeholder}"
        for placeholder in PLACEHOLDER_PATTERNS
        if placeholder in lowered
    )

    if "Software Heritage" not in body and "doi.org/10." not in body:
        findings.append("paper must link to a permanent software archive")
    if "human author" not in lowered or "validated" not in lowered:
        findings.append("AI disclosure must record human review and validation")
    if re.search(r"\band\s+others\b", bibliography, re.IGNORECASE):
        findings.append(
            "paper.bib contains placeholder author lists; record complete authors"
        )

    cited = set(CITATION_PATTERN.findall(body))
    bibliography_keys = set(BIB_KEY_PATTERN.findall(bibliography))
    missing = sorted(cited - bibliography_keys)
    if missing:
        findings.append(
            "paper cites bibliography keys that are missing: " + ", ".join(missing)
        )
    uncited = sorted(bibliography_keys - cited)
    if uncited:
        findings.append("paper.bib contains uncited records: " + ", ".join(uncited))
    return findings


def main() -> int:
    """Run the JOSS package validator from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    findings = validate_joss_package(args.root.resolve())
    if findings:
        for finding in findings:
            print(f"JOSS readiness: {finding}")
        return 1
    print("JOSS readiness: paper package satisfies repository-owned checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
