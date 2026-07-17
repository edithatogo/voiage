"""Contract tests for the repository-owned engineering harness."""

from pathlib import Path

from scripts.repo_harness import (
    check_context_contract,
    check_docs_platform,
    check_workflows,
    collect_findings,
)

ROOT = Path(__file__).parents[1]


def test_current_repository_has_no_harness_findings() -> None:
    """The checked-in repository satisfies the fail-closed harness."""

    assert collect_findings(ROOT) == []


def test_unpinned_action_is_rejected(tmp_path: Path) -> None:
    """A mutable action reference cannot enter the workflow surface."""

    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True)
    (workflow_root / "ci.yml").write_text(
        """name: CI\non: [push]\npermissions:\n  contents: read\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n""",
        encoding="utf-8",
    )
    findings = check_workflows(tmp_path)
    assert any("not pinned" in finding.message for finding in findings)


def test_sphinx_build_configuration_is_rejected(tmp_path: Path) -> None:
    """The harness prevents a second documentation toolchain from returning."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "conf.py").write_text("project = 'legacy'\n", encoding="utf-8")
    findings = check_docs_platform(tmp_path)
    assert any("Sphinx" in finding.message for finding in findings)


def test_missing_agent_context_section_is_rejected(tmp_path: Path) -> None:
    """The harness keeps agent-facing repository context explicit."""
    (tmp_path / "AGENTS.md").write_text(
        "# Agent Protocol\n\n## Context Loading Order\n", encoding="utf-8"
    )
    findings = check_context_contract(tmp_path)
    assert any("Repository Context Map" in finding.message for finding in findings)
