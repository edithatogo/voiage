"""Contract tests for the repository-owned JOSS submission package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from scripts.audit_joss_sources import _reported_output_path, _sourceright_manuscript
from scripts.validate_joss import (
    _normalise_prose,
    _submodule_commit,
    validate_joss_package,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_submission_metadata(destination: Path) -> None:
    """Copy the checked-in contract surface into a temporary JOSS package."""
    for filename in ("CITATION.cff", "codemeta.json"):
        shutil.copy2(ROOT / filename, destination / filename)
    shutil.copytree(ROOT / "paper", destination / "paper")
    for directory in (
        ".repo-tools",
        "bindings",
        "docs",
        "r-package",
        "rust",
        "scripts",
        "tests",
        "voiage",
    ):
        (destination / directory).symlink_to(ROOT / directory, target_is_directory=True)


def test_current_joss_package_satisfies_repository_contract() -> None:
    """The checked-in JOSS package should pass all automatable preflight gates."""
    assert validate_joss_package(ROOT) == []


def test_joss_tool_revisions_come_from_pinned_gitlinks() -> None:
    """General CI can validate pins without initialized submodule worktrees."""
    assert (
        _submodule_commit(ROOT, ".repo-tools/sourceright")
        == "dde39b3bb334f79f12e395a5317b21e036336bdd"
    )
    assert (
        _submodule_commit(ROOT, ".repo-tools/authentext")
        == "7f70dad5b6deab1af92faf037ef2638e7f3aea05"
    )


def test_joss_source_audit_reports_temporary_outputs_portably(
    tmp_path: Path,
) -> None:
    """Independent read-only audits may retain evidence outside the checkout."""
    output_directory = tmp_path / "sourceright"
    output_directory.mkdir()
    report = output_directory / "citations.md"
    report.touch()

    assert _reported_output_path(report, output_directory) == "citations.md"


def test_joss_independent_validation_protocol_is_bounded() -> None:
    """Independent evidence must not be substituted with automated activity."""
    protocol = (ROOT / "docs/release/joss-independent-validation.md").read_text(
        encoding="utf-8"
    )
    readiness = (ROOT / "docs/release/joss-submission-readiness.md").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (ROOT / "paper/joss-readiness-manifest.json").read_text(encoding="utf-8")
    )

    assert "voiage==2.0.0" in protocol
    assert "EVPI: 0.667" in protocol
    assert "issue #471" in protocol
    assert "AI-agent run" in protocol
    assert (
        "The selected route is **pyOpenSci review first, followed by a JOSS partner"
        in readiness
    )
    assert "separate maintainer instruction" in readiness
    assert "automated accounts" in readiness
    assert "confirmed by the author on 24 July 2026" in readiness
    assert "No external funding and no competing interests" in readiness
    assert manifest["submission_route"] == {
        "selected": "pyopensci_first_then_joss_partner_fast_track",
        "selected_at": "2026-08-27",
        "pyopensci_inquiry_or_review": "not_started",
        "pyopensci_acceptance": "pending",
        "joss_partner_referral": "not_started",
        "authority": (
            "maintainer_for_authenticated_actions_and_external_venues_for_acceptance"
        ),
    }
    assert manifest["submission_performed"] is False


def test_paper_reproduction_manifest_matches_tracked_outputs() -> None:
    """The reviewer command should verify every generated paper artefact."""
    manifest = ROOT / "paper/reproduction.sha256"
    entries = [
        line.split(maxsplit=1)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(entries) == 6
    for expected_digest, relative_path in entries:
        artefact = ROOT / relative_path
        assert artefact.is_file()
        assert hashlib.sha256(artefact.read_bytes()).hexdigest() == expected_digest


def test_structured_reproduction_manifest_binds_inputs_and_outputs() -> None:
    """The reviewer record declares identity, environment, seeds, and inputs."""
    manifest = json.loads(
        (ROOT / "paper/reproduction-manifest.json").read_text(encoding="utf-8")
    )
    checksum_entries = {
        relative_path: expected_digest
        for expected_digest, relative_path in (
            line.split(maxsplit=1)
            for line in (ROOT / "paper/reproduction.sha256")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }

    assert manifest["schema_version"] == "voiage.paper.reproduction.v1"
    assert manifest["source_reference"] == "v2.0.0"
    assert manifest["synthetic_data"] is True
    assert manifest["seeds"] == {
        "probabilistic_sensitivity_analysis": 20260723,
        "bootstrap": 20260724,
    }
    assert (
        manifest["lockfile"]["sha256"]
        == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    )
    assert manifest["verification_command"].endswith("--verify-tracked")
    assert manifest["inputs"]["probabilistic_sensitivity_analysis_draws"] == 10_000
    assert manifest["inputs"]["evaluated_total_sample_sizes"] == [
        50,
        100,
        200,
        400,
        800,
        1_200,
    ]
    assert manifest["inputs"]["delayed_scenario_years"] == 2
    assert manifest["inputs"]["delayed_scenario_value_realisation"] == 0.6
    assert manifest["inputs"]["bootstrap_interval"] == {
        "method": "paired percentile bootstrap of PSA draws",
        "confidence_level": 0.95,
    }
    assert len(manifest["inputs"]["sensitivity_scenarios"]) == 9
    assert {
        item["path"]: item["sha256"] for item in manifest["outputs"]
    } == checksum_entries


def test_joss_workflow_uses_pinned_open_journals_builder() -> None:
    """Hosted rendering should use the official pinned JOSS toolchain."""
    workflow = (ROOT / ".github/workflows/joss-paper.yml").read_text(encoding="utf-8")

    assert "permissions: {}" in workflow
    assert (
        "openjournals/openjournals-draft-action@"
        "85a18372e48f551d8af9ddb7a747de685fbbb01c"
    ) in workflow
    assert "python scripts/validate_joss.py" in workflow
    assert "generate_paper_health_example.py --verify-tracked" in workflow
    assert "submodules: recursive" in workflow
    assert "scripts/audit_joss_sources.py" in workflow
    assert "scripts/audit_joss_authentext.py" in workflow
    assert "scripts/audit_joss_readability.py" in workflow
    assert "build/joss/article-contract-report.json" in workflow
    assert "requirements-joss.txt" in workflow
    assert "if-no-files-found: error" in workflow
    assert '"CITATION.cff"' in workflow
    assert '"codemeta.json"' in workflow


def test_joss_article_contract_has_exact_target_and_substantive_sections() -> None:
    """The checked-in contract fixes the internal budget and section substance."""
    contract = json.loads(
        (ROOT / "paper/joss-article-contract.json").read_text(encoding="utf-8")
    )

    assert contract["word_count"] == {
        "unit": (
            "body words after YAML front matter, including headings and "
            "figure descriptions"
        ),
        "official_minimum": 750,
        "official_maximum": 1750,
        "target": 1600,
        "tolerance_fraction": 0.02,
        "accepted_minimum": 1568,
        "accepted_maximum": 1632,
    }
    assert contract["section_order"] == list(contract["sections"])
    assert contract["metadata"]["affiliation_rors"] == {
        "1": "0040r6f76",
        "2": "01kpzv902",
        "3": "01ej9dk98",
    }
    assert contract["assessment_layers"]["external"] == [
        "demonstrated research use, at minimum by the developer",
        (
            "community engagement or collaborative input as a detailed-review "
            "criterion and strong positive pre-review signal"
        ),
        "JOSS editorial screening, review, and acceptance",
    ]
    assert all(
        rules["minimum_words"] <= rules["maximum_words"] and rules["requirements"]
        for rules in contract["sections"].values()
    )


def test_joss_readiness_distinguishes_use_gate_from_engagement_signal() -> None:
    """Submission readiness should preserve the current criteria distinction."""
    readiness = json.loads(
        (ROOT / "paper/joss-readiness-manifest.json").read_text(encoding="utf-8")
    )

    assert readiness["external_gates"]["demonstrated_research_use"] == "ready"
    developer_use = json.loads(
        (ROOT / "paper/joss-developer-research-use.json").read_text(encoding="utf-8")
    )
    assert developer_use["published_package"]["version"] == "2.0.0"
    assert developer_use["analysis"]["draws"] == 500
    assert (
        readiness["author_project_sequence"][
            "community_engagement_before_joss_submission"
        ]
        == "pending"
    )
    assert readiness["author_project_sequence"]["joss_requirement"] == {
        "permanent_arxiv_identifier": False,
        "community_engagement": (
            "detailed_review_criterion_and_strong_positive_pre_review_signal"
        ),
    }


def test_joss_validator_rejects_internal_word_budget_drift(tmp_path: Path) -> None:
    """The stricter 1,600 ±2% budget remains a fail-closed article gate."""
    source = (ROOT / "paper.md").read_text(encoding="utf-8")
    (tmp_path / "paper.md").write_text(
        source.replace(
            "The synthetic health example compares",
            "Briefly, the synthetic health example compares",
        )
        + "\nAdditional uncontracted prose " * 30,
        encoding="utf-8",
    )
    (tmp_path / "paper.bib").write_text(
        (ROOT / "paper.bib").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_submission_metadata(tmp_path)

    findings = validate_joss_package(tmp_path)

    assert any("contract target is 1600" in finding for finding in findings)


def test_joss_validator_rejects_section_order_drift(tmp_path: Path) -> None:
    """Required headings cannot pass when their article order changes."""
    source = (ROOT / "paper.md").read_text(encoding="utf-8")
    source = source.replace("# Statement of need", "# TEMPORARY")
    source = source.replace("# State of the field", "# Statement of need")
    source = source.replace("# TEMPORARY", "# State of the field")
    (tmp_path / "paper.md").write_text(source, encoding="utf-8")
    (tmp_path / "paper.bib").write_text(
        (ROOT / "paper.bib").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_submission_metadata(tmp_path)

    findings = validate_joss_package(tmp_path)

    assert any("section order differs" in finding for finding in findings)


def test_joss_validator_rejects_unresolved_author_affiliation(tmp_path: Path) -> None:
    """Structured YAML checks resolve every author affiliation by index."""
    source = (ROOT / "paper.md").read_text(encoding="utf-8")
    (tmp_path / "paper.md").write_text(
        source.replace('affiliation: "1, 2, 3"', 'affiliation: "1, 2, 4"'),
        encoding="utf-8",
    )
    (tmp_path / "paper.bib").write_text(
        (ROOT / "paper.bib").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_submission_metadata(tmp_path)

    findings = validate_joss_package(tmp_path)

    assert "paper.md author affiliation indices do not resolve: 4" in findings


def test_joss_validator_rejects_incorrect_affiliation_ror(tmp_path: Path) -> None:
    """Affiliation metadata remains linked to its authoritative ROR record."""
    source = (ROOT / "paper.md").read_text(encoding="utf-8")
    (tmp_path / "paper.md").write_text(
        source.replace('ror: "01kpzv902"', 'ror: "000000000"'),
        encoding="utf-8",
    )
    (tmp_path / "paper.bib").write_text(
        (ROOT / "paper.bib").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_submission_metadata(tmp_path)

    findings = validate_joss_package(tmp_path)

    assert "paper.md affiliation 2 ROR identifier is missing or incorrect" in findings


def test_sourceright_adapter_preserves_pandoc_citation_keys() -> None:
    """SourceRight receives exact citation identifiers rather than styled prose."""
    converted = _sourceright_manuscript("Evidence [@alpha2024; see @beta2025, p. 4].")

    assert converted == r"Evidence \cite{alpha2024,beta2025}."


def test_claim_reconciliation_ignores_markdown_link_destinations() -> None:
    """Evidence claims compare authored prose rather than Markdown destinations."""
    prose = "Derived from the [reproduction notes](paper/methods.md)."

    assert _normalise_prose(prose) == "derived from the reproduction notes."


def test_joss_validator_rejects_missing_required_section(tmp_path: Path) -> None:
    """Required JOSS sections remain fail-closed."""
    (tmp_path / "paper.md").write_text(
        "---\ntitle: Example\nbibliography: paper.bib\n---\n\n# Summary\n"
        + "word " * 800,
        encoding="utf-8",
    )
    (tmp_path / "paper.bib").write_text("@misc{example, title={Example}}\n")
    _write_submission_metadata(tmp_path)

    findings = validate_joss_package(tmp_path)

    assert any("Statement of need" in finding for finding in findings)


def test_joss_validator_rejects_placeholder_language(tmp_path: Path) -> None:
    """Workflow placeholders must not reach a submission draft."""
    source = (ROOT / "paper.md").read_text(encoding="utf-8")
    (tmp_path / "paper.md").write_text(
        source.replace(
            "# Summary",
            "# Summary\n\nThis statement must be updated before submission.",
        ),
        encoding="utf-8",
    )
    (tmp_path / "paper.bib").write_text(
        (ROOT / "paper.bib").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_submission_metadata(tmp_path)

    findings = validate_joss_package(tmp_path)

    assert any("placeholder" in finding for finding in findings)


def test_joss_validator_rejects_placeholder_bibliography_authors(
    tmp_path: Path,
) -> None:
    """Incomplete author lists cannot pass the submission preflight."""
    source = (ROOT / "paper.md").read_text(encoding="utf-8")
    bibliography = (ROOT / "paper.bib").read_text(encoding="utf-8")
    (tmp_path / "paper.md").write_text(source, encoding="utf-8")
    (tmp_path / "paper.bib").write_text(
        bibliography.replace(
            "author = {Ades, A. E. and Lu, G. and Claxton, Karl}",
            "author = {Ades, A. E. and Lu, G. and Claxton, Karl and others}",
        ),
        encoding="utf-8",
    )
    _write_submission_metadata(tmp_path)

    findings = validate_joss_package(tmp_path)

    assert (
        "paper.bib contains placeholder author lists; record complete authors"
        in findings
    )


def test_joss_validator_rejects_discovery_metadata_version_drift(
    tmp_path: Path,
) -> None:
    """Citation and discovery records must describe the released JOSS package."""
    (tmp_path / "paper.md").write_text(
        (ROOT / "paper.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "paper.bib").write_text(
        (ROOT / "paper.bib").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_submission_metadata(tmp_path)
    codemeta = tmp_path / "codemeta.json"
    codemeta.write_text(
        codemeta.read_text(encoding="utf-8").replace(
            '"version": "2.1.0"', '"version": "0.9.0"'
        ),
        encoding="utf-8",
    )

    findings = validate_joss_package(tmp_path)

    assert "codemeta.json version must match CITATION.cff version" in findings
