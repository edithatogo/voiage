"""Contract tests for the repository-owned JOSS submission package."""

from __future__ import annotations

from pathlib import Path
import shutil

from scripts.validate_joss import validate_joss_package

ROOT = Path(__file__).resolve().parents[1]


def _write_submission_metadata(destination: Path) -> None:
    """Copy the checked-in discovery metadata into a temporary JOSS package."""
    for filename in ("CITATION.cff", "codemeta.json"):
        shutil.copy2(ROOT / filename, destination / filename)


def test_current_joss_package_satisfies_repository_contract() -> None:
    """The checked-in JOSS package should pass all automatable preflight gates."""
    assert validate_joss_package(ROOT) == []


def test_joss_independent_validation_protocol_is_bounded() -> None:
    """Independent evidence must not be substituted with automated activity."""
    protocol = (ROOT / "docs/release/joss-independent-validation.md").read_text(
        encoding="utf-8"
    )
    readiness = (ROOT / "docs/release/joss-submission-readiness.md").read_text(
        encoding="utf-8"
    )

    assert "voiage==2.0.0" in protocol
    assert "EVPI: 0.667" in protocol
    assert "issue #471" in protocol
    assert "AI-agent run" in protocol
    assert "The selected route is **direct JOSS submission**" in readiness
    assert "automated accounts" in readiness
    assert "confirmed by the author on 24 July 2026" in readiness
    assert "No external funding and no competing interests" in readiness


def test_joss_workflow_uses_pinned_open_journals_builder() -> None:
    """Hosted rendering should use the official pinned JOSS toolchain."""
    workflow = (ROOT / ".github/workflows/joss-paper.yml").read_text(encoding="utf-8")

    assert "permissions: {}" in workflow
    assert (
        "openjournals/openjournals-draft-action@"
        "85a18372e48f551d8af9ddb7a747de685fbbb01c"
    ) in workflow
    assert "python3 scripts/validate_joss.py" in workflow
    assert "if-no-files-found: error" in workflow
    assert '"CITATION.cff"' in workflow
    assert '"codemeta.json"' in workflow


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

    assert findings == [
        "paper.bib contains placeholder author lists; record complete authors"
    ]


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
            '"version": "2.0.0"', '"version": "0.9.0"'
        ),
        encoding="utf-8",
    )

    findings = validate_joss_package(tmp_path)

    assert "codemeta.json version must match CITATION.cff version" in findings
