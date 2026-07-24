"""Validate the repository-owned JOSS paper package."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
from typing import Any

REQUIRED_SECTIONS = (
    "Summary",
    "Statement of need",
    "State of the field",
    "Software design",
    "Research impact statement",
    "AI usage disclosure",
    "Acknowledgements",
    "Software and data availability",
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
CFF_SCALAR_PATTERN = re.compile(
    r'^(?P<key>[A-Za-z][A-Za-z0-9-]*):\s*(?:"(?P<quoted>[^"]*)"|(?P<plain>[^#\n]+?))\s*$',
    re.MULTILINE,
)
CFF_ORCID_PATTERN = re.compile(r"^\s+orcid:\s*(?P<orcid>\S+)\s*$", re.MULTILINE)


def _body_without_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    parts = text.split("---", maxsplit=2)
    return parts[2] if len(parts) == 3 else text


def _cff_scalars(text: str) -> dict[str, str]:
    """Extract the small stable scalar surface used from CITATION.cff."""
    return {
        match.group("key"): (match.group("quoted") or match.group("plain")).strip()
        for match in CFF_SCALAR_PATTERN.finditer(text)
    }


def _validate_discovery_metadata(root: Path) -> list[str]:
    """Validate CFF and CodeMeta identity fields needed by JOSS reviewers."""
    findings: list[str] = []
    cff_path = root / "CITATION.cff"
    codemeta_path = root / "codemeta.json"
    if not cff_path.is_file():
        findings.append("CITATION.cff is missing")
    if not codemeta_path.is_file():
        findings.append("codemeta.json is missing")
    if findings:
        return findings

    scalars = _cff_scalars(cff_path.read_text(encoding="utf-8"))
    required_cff_fields = (
        "title",
        "version",
        "date-released",
        "repository-code",
        "url",
        "license",
    )
    findings.extend(
        f"CITATION.cff is missing {field}"
        for field in required_cff_fields
        if field not in scalars
    )
    orcid_match = CFF_ORCID_PATTERN.search(cff_path.read_text(encoding="utf-8"))
    if orcid_match is None:
        findings.append("CITATION.cff is missing an author ORCID")
    try:
        date.fromisoformat(scalars.get("date-released", ""))
    except ValueError:
        findings.append("CITATION.cff date-released must use ISO-8601 format")

    try:
        codemeta: dict[str, Any] = json.loads(codemeta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [*findings, "codemeta.json is not valid JSON"]

    version = scalars.get("version")
    repository = scalars.get("repository-code")
    licence = scalars.get("license")
    if version and codemeta.get("version") != version:
        findings.append("codemeta.json version must match CITATION.cff version")
    if repository and codemeta.get("codeRepository") != repository:
        findings.append(
            "codemeta.json codeRepository must match CITATION.cff repository-code"
        )
    if scalars.get("url") and codemeta.get("url") != scalars["url"]:
        findings.append("codemeta.json url must match CITATION.cff url")
    if licence and licence not in str(codemeta.get("license", "")):
        findings.append("codemeta.json license must identify the CITATION.cff licence")
    if (
        version
        and codemeta.get("downloadUrl") != f"https://pypi.org/project/voiage/{version}/"
    ):
        findings.append(
            "codemeta.json downloadUrl must identify the CITATION.cff release"
        )
    if (
        version
        and codemeta.get("releaseNotes") != f"{repository}/releases/tag/v{version}"
    ):
        findings.append(
            "codemeta.json releaseNotes must identify the CITATION.cff release"
        )

    authors = codemeta.get("author")
    codemeta_orcid = (
        authors[0].get("@id")
        if isinstance(authors, list) and authors and isinstance(authors[0], dict)
        else None
    )
    if orcid_match and codemeta_orcid != orcid_match.group("orcid"):
        findings.append("codemeta.json author ORCID must match CITATION.cff")
    return findings


def validate_joss_package(root: Path) -> list[str]:
    """Return fail-closed findings for the JOSS manuscript and bibliography."""
    findings: list[str] = []
    paper_path = root / "paper.md"
    bibliography_path = root / "paper.bib"
    if not paper_path.is_file():
        return ["paper.md is missing"]
    if not bibliography_path.is_file():
        return ["paper.bib is missing"]

    findings.extend(_validate_discovery_metadata(root))

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
