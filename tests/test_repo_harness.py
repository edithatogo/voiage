"""Contract tests for the repository-owned engineering harness."""

import json
from pathlib import Path

from scripts.repo_harness import (
    check_conflict_markers,
    check_context_contract,
    check_docs_platform,
    check_operational_assurance,
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


def test_stale_uv_version_is_rejected(tmp_path: Path) -> None:
    """All workflow lanes use the repository's reviewed uv frontier."""
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True)
    (workflow_root / "ci.yml").write_text(
        """name: CI
on: [push]
permissions: {}
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990
        with:
          version: "0.9.18"
""",
        encoding="utf-8",
    )

    findings = check_workflows(tmp_path)

    assert any("frontier version 0.11.29" in finding.message for finding in findings)


def test_private_assurance_dependencies_do_not_pollute_source_scan(
    tmp_path: Path,
) -> None:
    """Private retained environments are outside the tracked-source contract."""
    fixture = tmp_path / ".assurance" / "vendor" / "merge-conflict.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        "<<<<<<< third-party fixture\n>>>>>>> fixture\n", encoding="utf-8"
    )

    assert check_conflict_markers(tmp_path) == []


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


def test_missing_operational_assurance_gate_is_rejected(tmp_path: Path) -> None:
    """The harness detects removal of a promoted assurance gate."""
    findings = check_operational_assurance(tmp_path)
    assert any(
        finding.path == ".github/workflows/operational-assurance.yml"
        for finding in findings
    )


def test_weakened_coverage_policy_is_rejected(tmp_path: Path) -> None:
    """Every promoted coverage threshold is structurally pinned."""
    policy = tmp_path / ".github" / "coverage-policy.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        '{"schema_version":"1.0.0","aggregate_percent":90.0,'
        '"critical_modules":{},"changed_line_percent":0.0,'
        '"changed_branch_percent":90.0}',
        encoding="utf-8",
    )
    findings = check_operational_assurance(tmp_path)
    assert any("exactly match" in finding.message for finding in findings)


def test_self_asserted_mutation_approval_is_rejected(tmp_path: Path) -> None:
    """Repository content cannot declare its own governance approval."""
    baseline = tmp_path / ".github" / "mutation-baselines" / "voiage-cohort.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps(
            {
                "promotion_provenance": {
                    "review_state": "requires_external_anchor",
                    "human_approved": True,
                },
                "cohort": {
                    "tool_version": "3.6.0",
                    "lock_sha256": "0" * 64,
                    "configuration_sha256": "0" * 64,
                    "sources": [],
                },
                "universe": {"ids": [], "sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    findings = check_operational_assurance(tmp_path)
    assert any("external review anchor" in finding.message for finding in findings)
