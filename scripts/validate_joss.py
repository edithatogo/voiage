"""Validate the repository-owned JOSS paper package."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml

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
HEADING_PATTERN = re.compile(r"^# (?P<title>.+?)\s*$", re.MULTILINE)
CFF_SCALAR_PATTERN = re.compile(
    r'^(?P<key>[A-Za-z][A-Za-z0-9-]*):\s*(?:"(?P<quoted>[^"]*)"|(?P<plain>[^#\n]+?))\s*$',
    re.MULTILINE,
)
CFF_ORCID_PATTERN = re.compile(r"^\s+orcid:\s*(?P<orcid>\S+)\s*$", re.MULTILINE)
EXPECTED_AFFILIATION_RORS = {
    1: "0040r6f76",
    2: "01kpzv902",
    3: "01ej9dk98",
}


def _body_without_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    parts = text.split("---", maxsplit=2)
    return parts[2] if len(parts) == 3 else text


def _front_matter(text: str) -> dict[str, Any] | None:
    """Parse the initial JOSS YAML mapping without accepting later YAML blocks."""
    if not text.startswith("---\n"):
        return None
    parts = text.split("---", maxsplit=2)
    if len(parts) != 3:
        return None
    loaded: Any = yaml.safe_load(parts[1])
    return loaded if isinstance(loaded, dict) else None


def _validate_joss_metadata(paper: str) -> list[str]:
    """Validate author, affiliation, date, bibliography, and repository metadata."""
    findings: list[str] = []
    metadata = _front_matter(paper)
    if metadata is None:
        return ["paper.md YAML front matter must be a mapping"]
    title = metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        findings.append("paper.md metadata title must be non-empty")
    if metadata.get("bibliography") != "paper.bib":
        findings.append("paper.md bibliography must identify paper.bib")
    if metadata.get("repository") != "https://github.com/edithatogo/voiage":
        findings.append("paper.md repository must identify the submitted repository")
    publication_date = metadata.get("date")
    if not isinstance(publication_date, str):
        findings.append("paper.md date must use the form D Month YYYY")
    else:
        try:
            parsed_date = datetime.strptime(publication_date, "%d %B %Y")
        except ValueError:
            findings.append("paper.md date must use the form D Month YYYY")
        else:
            canonical_date = (
                f"{parsed_date.day} {parsed_date.strftime('%B')} {parsed_date.year}"
            )
            if canonical_date != publication_date:
                findings.append("paper.md date must use the form D Month YYYY")

    affiliations = metadata.get("affiliations")
    affiliation_indices: set[int] = set()
    if not isinstance(affiliations, list) or not affiliations:
        findings.append("paper.md affiliations must be a non-empty array")
    else:
        for affiliation in affiliations:
            if not isinstance(affiliation, dict):
                findings.append("paper.md affiliation entries must be mappings")
                continue
            index = affiliation.get("index")
            name = affiliation.get("name")
            if not isinstance(index, int) or index in affiliation_indices:
                findings.append("paper.md affiliation indices must be unique integers")
            else:
                affiliation_indices.add(index)
            if not isinstance(name, str) or not name.strip():
                findings.append("paper.md affiliation names must be non-empty")
            ror = affiliation.get("ror")
            if isinstance(index, int) and ror != EXPECTED_AFFILIATION_RORS.get(index):
                findings.append(
                    f"paper.md affiliation {index} ROR identifier is missing or incorrect"
                )

    authors = metadata.get("authors")
    if not isinstance(authors, list) or not authors:
        findings.append("paper.md authors must be a non-empty array")
    else:
        for author in authors:
            if not isinstance(author, dict):
                findings.append("paper.md author entries must be mappings")
                continue
            findings.extend(
                f"paper.md author {field} must be non-empty"
                for field in ("given-names", "surname")
                if not isinstance(author.get(field), str) or not author[field].strip()
            )
            affiliation = author.get("affiliation")
            if not isinstance(affiliation, str):
                findings.append("paper.md author affiliations must be an index list")
                continue
            try:
                author_indices = {
                    int(value.strip()) for value in affiliation.split(",")
                }
            except ValueError:
                findings.append("paper.md author affiliations must be an index list")
                continue
            missing = sorted(author_indices - affiliation_indices)
            if missing:
                findings.append(
                    "paper.md author affiliation indices do not resolve: "
                    + ", ".join(str(value) for value in missing)
                )
    return findings


def _normalise_prose(text: str) -> str:
    """Normalise authored prose for exact claim-to-manuscript reconciliation."""
    text = re.sub(r"\[[^\]]*@[^\]]+\]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _section_map(body: str) -> tuple[list[str], dict[str, str]]:
    """Return first-level section order and prose beneath each heading."""
    headings = list(HEADING_PATTERN.finditer(body))
    order = [heading.group("title") for heading in headings]
    sections: dict[str, str] = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        sections[heading.group("title")] = body[heading.end() : end].strip()
    return order, sections


def _load_json_object(
    path: Path, label: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load a required JSON object with a reader-facing diagnostic."""
    if not path.is_file():
        return None, [f"{label} is missing: {path.relative_to(path.parents[1])}"]
    try:
        loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, [f"{label} is not valid JSON"]
    if not isinstance(loaded, dict):
        return None, [f"{label} must be a JSON object"]
    return loaded, []


def _submodule_commit(root: Path, relative_path: str) -> str | None:
    """Read the pinned gitlink, falling back to an initialized tool checkout."""
    gitlink = subprocess.run(  # noqa: S603 - fixed local Git inspection
        [  # noqa: S607 - system Git is required for repository metadata
            "git",
            "-C",
            str(root),
            "ls-tree",
            "HEAD",
            "--",
            relative_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    fields = gitlink.stdout.split()
    if gitlink.returncode == 0 and len(fields) >= 3 and fields[0] == "160000":
        return fields[2]

    path = root / relative_path
    if not path.is_dir():
        return None
    top_level = subprocess.run(  # noqa: S603 - fixed local Git inspection
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        top_level.returncode != 0
        or Path(top_level.stdout.strip()).resolve() != path.resolve()
    ):
        return None
    checkout = subprocess.run(  # noqa: S603 - fixed local Git inspection
        ["git", "-C", str(path), "rev-parse", "HEAD"],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    return checkout.stdout.strip() if checkout.returncode == 0 else None


def _validate_article_contract(
    root: Path,
    body: str,
    bibliography: str,
) -> list[str]:
    """Validate the current JOSS article against its explicit 2026 contract."""
    findings: list[str] = []
    contract, contract_findings = _load_json_object(
        root / "paper/joss-article-contract.json",
        "JOSS article contract",
    )
    findings.extend(contract_findings)
    if contract is None:
        return findings

    if contract.get("schema_version") != "voiage.joss-article-contract.v1":
        findings.append("JOSS article contract schema_version is unsupported")
    if contract.get("canonical_source") != "paper.md":
        findings.append("JOSS article contract must identify paper.md as canonical")
    metadata_contract = contract.get("metadata")
    if not isinstance(metadata_contract, dict):
        findings.append("JOSS article contract metadata must be an object")
    else:
        if metadata_contract.get("bibliography") != "paper.bib":
            findings.append("JOSS article contract bibliography must be paper.bib")
        if (
            metadata_contract.get("repository")
            != "https://github.com/edithatogo/voiage"
        ):
            findings.append(
                "JOSS article contract repository must identify the submitted project"
            )

    count_contract = contract.get("word_count")
    if not isinstance(count_contract, dict):
        findings.append("JOSS article contract word_count must be an object")
    else:
        target = count_contract.get("target")
        tolerance = count_contract.get("tolerance_fraction")
        accepted_minimum = count_contract.get("accepted_minimum")
        accepted_maximum = count_contract.get("accepted_maximum")
        if target != 1600 or tolerance != 0.02:
            findings.append(
                "JOSS article target must remain 1600 words with ±2% tolerance"
            )
        if accepted_minimum != 1568 or accepted_maximum != 1632:
            findings.append("JOSS article accepted word band must be 1568 through 1632")
        word_count = len(WORD_PATTERN.findall(body))
        if not isinstance(accepted_minimum, int) or not isinstance(
            accepted_maximum, int
        ):
            findings.append("JOSS article accepted word bounds must be integers")
        elif not accepted_minimum <= word_count <= accepted_maximum:
            findings.append(
                f"JOSS article has {word_count} body words; "
                f"contract target is {target} with accepted band "
                f"{accepted_minimum} through {accepted_maximum}"
            )

    actual_order, sections = _section_map(body)
    expected_order = contract.get("section_order")
    if not isinstance(expected_order, list) or not all(
        isinstance(item, str) for item in expected_order
    ):
        findings.append("JOSS article contract section_order must be a string array")
        expected_order = []
    elif actual_order != expected_order:
        findings.append(
            "JOSS first-level section order differs from the article contract: "
            + ", ".join(actual_order)
        )

    section_contracts = contract.get("sections")
    if not isinstance(section_contracts, dict):
        findings.append("JOSS article contract sections must be an object")
    else:
        for title in expected_order:
            rules = section_contracts.get(title)
            if not isinstance(rules, dict):
                findings.append(f"JOSS article contract lacks rules for {title}")
                continue
            prose = sections.get(title)
            if prose is None:
                continue
            minimum = rules.get("minimum_words")
            maximum = rules.get("maximum_words")
            requirements = rules.get("requirements")
            if (
                not isinstance(minimum, int)
                or not isinstance(maximum, int)
                or minimum > maximum
            ):
                findings.append(f"JOSS section word bounds are invalid: {title}")
                continue
            section_words = len(WORD_PATTERN.findall(prose))
            if not minimum <= section_words <= maximum:
                findings.append(
                    f"JOSS section {title} has {section_words} words; "
                    f"contract requires {minimum} through {maximum}"
                )
            if (
                not isinstance(requirements, list)
                or not requirements
                or not all(
                    isinstance(requirement, str) and requirement.strip()
                    for requirement in requirements
                )
            ):
                findings.append(f"JOSS section requirements are incomplete: {title}")
    if r"\autoref{fig:health-example}" not in body:
        findings.append("JOSS worked-example figure must be cross-referenced in text")
    if "{#fig:health-example}" not in body:
        findings.append("JOSS worked-example figure must have a stable identifier")

    boundaries = contract.get("paper_boundaries")
    if not isinstance(boundaries, dict):
        findings.append("JOSS article contract paper_boundaries must be an object")
    else:
        artifacts = boundaries.get("required_supporting_artifacts")
        if not isinstance(artifacts, list):
            findings.append("JOSS required_supporting_artifacts must be an array")
        else:
            findings.extend(
                f"required JOSS supporting artifact is missing: {artifact}"
                for artifact in artifacts
                if not isinstance(artifact, str) or not (root / artifact).is_file()
            )

    evidence, evidence_findings = _load_json_object(
        root / "paper/joss-claim-evidence.json",
        "JOSS claim-evidence manifest",
    )
    findings.extend(evidence_findings)
    cited = set(CITATION_PATTERN.findall(body))
    bibliography_keys = set(BIB_KEY_PATTERN.findall(bibliography))
    if evidence is not None:
        claims = evidence.get("claims")
        if not isinstance(claims, list) or not claims:
            findings.append("JOSS claim-evidence manifest must contain claims")
        else:
            claim_ids: set[str] = set()
            for claim in claims:
                if not isinstance(claim, dict):
                    findings.append("JOSS claim-evidence entries must be objects")
                    continue
                claim_id = claim.get("id")
                section = claim.get("section")
                statement = claim.get("statement")
                status = claim.get("status")
                if not isinstance(claim_id, str) or not claim_id:
                    findings.append("JOSS claim-evidence entry lacks an id")
                    continue
                if claim_id in claim_ids:
                    findings.append(f"duplicate JOSS claim-evidence id: {claim_id}")
                claim_ids.add(claim_id)
                if status not in {"verified", "bounded"}:
                    findings.append(
                        f"JOSS claim {claim_id} has invalid status: {status}"
                    )
                if not isinstance(section, str) or section not in sections:
                    findings.append(
                        f"JOSS claim {claim_id} identifies an unknown section"
                    )
                    continue
                if not isinstance(statement, str) or _normalise_prose(
                    statement
                ) not in _normalise_prose(sections[section]):
                    findings.append(
                        f"JOSS claim {claim_id} no longer matches its manuscript statement"
                    )
                evidence_items = claim.get("evidence")
                if not isinstance(evidence_items, list) or not evidence_items:
                    findings.append(f"JOSS claim {claim_id} lacks evidence")
                else:
                    for item in evidence_items:
                        locator = (
                            item.get("locator") if isinstance(item, dict) else None
                        )
                        if not isinstance(locator, str) or not locator:
                            findings.append(
                                f"JOSS claim {claim_id} has an invalid evidence locator"
                            )
                        elif (
                            not locator.startswith(("https://", "http://"))
                            and not (root / locator).exists()
                        ):
                            findings.append(
                                f"JOSS claim {claim_id} evidence does not exist: {locator}"
                            )
                citation_keys = claim.get("citation_keys", [])
                if not isinstance(citation_keys, list):
                    findings.append(
                        f"JOSS claim {claim_id} citation_keys must be an array"
                    )
                else:
                    findings.extend(
                        f"JOSS claim {claim_id} citation is unresolved: {key}"
                        for key in citation_keys
                        if key not in cited or key not in bibliography_keys
                    )

    readiness, readiness_findings = _load_json_object(
        root / "paper/joss-readiness-manifest.json",
        "JOSS readiness manifest",
    )
    findings.extend(readiness_findings)
    assurance, assurance_findings = _load_json_object(
        root / "paper/joss-editorial-assurance.json",
        "JOSS editorial assurance manifest",
    )
    findings.extend(assurance_findings)
    if readiness is not None:
        source_assurance = readiness.get("source_assurance")
        if not isinstance(source_assurance, dict):
            findings.append("JOSS readiness source_assurance must be an object")
        else:
            for tool in ("sourceright", "authentext"):
                item = source_assurance.get(tool)
                if not isinstance(item, dict):
                    findings.append(f"JOSS readiness lacks {tool} assurance")
                    continue
                path = item.get("path")
                commit = item.get("commit")
                if not isinstance(path, str) or not isinstance(commit, str):
                    findings.append(f"JOSS readiness {tool} path or commit is invalid")
                    continue
                observed = _submodule_commit(root, path)
                if observed != commit:
                    findings.append(
                        f"JOSS readiness {tool} commit mismatch: "
                        f"expected {commit}, observed {observed or 'unavailable'}"
                    )
    if assurance is not None:
        tools = assurance.get("tools")
        if not isinstance(tools, dict):
            findings.append("JOSS editorial assurance tools must be an object")
        else:
            for tool in ("sourceright", "authentext"):
                item = tools.get(tool)
                status = item.get("status") if isinstance(item, dict) else None
                accepted_statuses = (
                    {"pass", "pass_with_warnings"}
                    if tool == "sourceright"
                    else {"pass"}
                )
                if status not in accepted_statuses:
                    findings.append(
                        f"JOSS editorial assurance {tool} status is not accepted: "
                        f"{status}"
                    )
                source_sha256 = (
                    item.get("source_sha256") if isinstance(item, dict) else None
                )
                observed_sha256 = sha256((root / "paper.md").read_bytes()).hexdigest()
                if source_sha256 != observed_sha256:
                    findings.append(
                        f"JOSS editorial assurance {tool} source hash is stale"
                    )
                if tool == "sourceright":
                    bibliography_sha256 = (
                        item.get("bibliography_sha256")
                        if isinstance(item, dict)
                        else None
                    )
                    observed_bibliography_sha256 = sha256(
                        (root / "paper.bib").read_bytes()
                    ).hexdigest()
                    if bibliography_sha256 != observed_bibliography_sha256:
                        findings.append(
                            "JOSS editorial assurance sourceright "
                            "bibliography hash is stale"
                        )
    return findings


def _journal_first_authorized(root: Path, readiness: dict[str, Any]) -> bool:
    """Require the bound maintainer decision before retiring the arXiv gate."""
    sequence = readiness.get("author_project_sequence", {})
    binding = sequence.get("maintainer_decision")
    receipt_path = (
        "conductor/tracks/v2_2_release_and_venue_submissions_20260830/"
        "maintainer-venue-decision-20260831.json"
    )
    if not isinstance(binding, dict) or binding.get("path") != receipt_path:
        return False
    path = root / receipt_path
    try:
        content = path.read_bytes()
        receipt = json.loads(content)
    except (OSError, ValueError):
        return False
    return (
        binding.get("sha256") == sha256(content).hexdigest()
        and isinstance(receipt, dict)
        and receipt.get("schema_version") == "voiage.maintainer-venue-decision.v1"
        and receipt.get("source") == "current_user_message"
        and receipt.get("candidate_version") == "2.2.0"
        and receipt.get("journal_first_authorized") is True
        and isinstance(receipt.get("user_statement"), str)
        and bool(receipt["user_statement"].strip())
    )


def _validate_submission_gates(root: Path, body: str) -> list[str]:
    """Return gates that must be observed before authenticated JOSS submission."""
    findings: list[str] = []
    readiness, readiness_findings = _load_json_object(
        root / "paper/joss-readiness-manifest.json",
        "JOSS readiness manifest",
    )
    findings.extend(readiness_findings)
    if readiness is None:
        return findings
    required_gates = {
        "manuscript_gates": {
            "article_contract": {"pass", "ready"},
            "citation_and_source_audit": {"pass", "ready"},
            "authentext_review": {"pass", "ready"},
            "independent_panel_review": {"complete_internal_review", "ready"},
            "official_pdf_build_and_visual_review": {"pass", "ready"},
        },
        "repository_gates": {
            "public_development_history": {"ready"},
            "license_documentation_tests_and_contribution_paths": {"ready"},
            "exact_reviewed_immutable_archive": {"ready"},
            "published_v2_2_0_release": {"ready"},
        },
    }
    for layer, required in required_gates.items():
        gates = readiness.get(layer)
        if not isinstance(gates, dict):
            findings.append(f"JOSS readiness {layer} must be an object")
            continue
        for gate in sorted(required.keys() | gates.keys()):
            status = gates.get(gate)
            if not isinstance(status, str) or status not in required.get(
                gate, {"ready"}
            ):
                findings.append(f"JOSS submission gate is not ready: {gate}={status}")
    route = readiness.get("submission_route")
    if not isinstance(route, dict):
        findings.append("JOSS readiness submission_route must be an object")
    elif (
        route.get("selected") != "pyopensci_first_then_joss_partner_fast_track"
        or route.get("pyopensci_acceptance") != "accepted"
    ):
        findings.append(
            "JOSS selected partner route requires observed pyOpenSci acceptance"
        )
    external = readiness.get("external_gates")
    if not isinstance(external, dict):
        findings.append("JOSS readiness external_gates must be an object")
    else:
        required_before_submission = {"demonstrated_research_use"}
        findings.extend(
            f"JOSS external submission gate is not ready: {gate}={external.get(gate)}"
            for gate in required_before_submission
            if external.get(gate) != "ready"
        )
    sequence = readiness.get("author_project_sequence")
    if not isinstance(sequence, dict):
        findings.append("JOSS readiness author_project_sequence must be an object")
    else:
        arxiv_gate = (
            "permanent_arxiv_identifier_and_announcement_before_joss_submission"
        )
        for gate in (arxiv_gate, "community_engagement_before_joss_submission"):
            status = sequence.get(gate)
            if status == "ready":
                continue
            if (
                gate == arxiv_gate
                and status == "not_required_by_maintainer"
                and _journal_first_authorized(root, readiness)
            ):
                continue
            findings.append(
                f"author-requested pre-JOSS sequence is not ready: {gate}={status}"
            )
    assurance, assurance_findings = _load_json_object(
        root / "paper/joss-editorial-assurance.json",
        "JOSS editorial assurance manifest",
    )
    findings.extend(assurance_findings)
    if assurance is not None:
        human_review = assurance.get("human_review")
        if not isinstance(human_review, dict):
            findings.append("JOSS editorial human_review must be an object")
        else:
            findings.extend(
                f"JOSS human submission gate is not confirmed: "
                f"{gate}={human_review.get(gate)}"
                for gate in (
                    "authorship_funding_conflicts",
                    "citation_source_check",
                    "all_retained_ai_outputs_reviewed_modified_and_validated",
                )
                if human_review.get(gate) != "confirmed"
            )
    prospective_phrases = (
        "requires a new release",
        "will cite a release",
        "before joss submission",
    )
    lowered = _normalise_prose(body)
    findings.extend(
        f"JOSS submission candidate retains prospective wording: {phrase}"
        for phrase in prospective_phrases
        if phrase in lowered
    )
    return findings


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
    findings.extend(_validate_joss_metadata(paper))

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

    lowered = _normalise_prose(body)
    findings.extend(
        f"submission placeholder remains: {placeholder}"
        for placeholder in PLACEHOLDER_PATTERNS
        if placeholder in lowered
    )

    if "Software Heritage" not in body and "doi.org/10." not in body:
        findings.append("paper must link to a permanent software archive")
    if not all(
        phrase in lowered
        for phrase in (
            "all retained ai-assisted outputs",
            "human author",
            "reviewed",
            "validated",
            "primary",
            "responsibility",
        )
    ):
        findings.append(
            "AI disclosure must record human primary decisions, review, "
            "validation, and responsibility"
        )
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
    findings.extend(_validate_article_contract(root, body, bibliography))
    return findings


def main() -> int:
    """Run the JOSS package validator from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    parser.add_argument(
        "--submission",
        action="store_true",
        help="also enforce observed repository, human, and external submission gates",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write a machine-readable article-contract report",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    findings = validate_joss_package(root)
    if args.submission and (root / "paper.md").is_file():
        body = _body_without_front_matter(
            (root / "paper.md").read_text(encoding="utf-8")
        )
        findings.extend(_validate_submission_gates(root, body))
    findings = list(dict.fromkeys(findings))
    if args.report:
        report_path = args.report if args.report.is_absolute() else root / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        body = (
            _body_without_front_matter((root / "paper.md").read_text(encoding="utf-8"))
            if (root / "paper.md").is_file()
            else ""
        )
        report = {
            "schema_version": "voiage.joss-article-contract-report.v1",
            "scope": "submission" if args.submission else "manuscript",
            "status": "pass" if not findings else "blocked",
            "canonical_source": "paper.md",
            "body_word_count": len(WORD_PATTERN.findall(body)),
            "target_word_count": 1600,
            "accepted_word_band": [1568, 1632],
            "findings": findings,
            "submission_performed": False,
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if findings:
        for finding in findings:
            print(f"JOSS readiness: {finding}")
        return 1
    print("JOSS readiness: paper package satisfies repository-owned checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
