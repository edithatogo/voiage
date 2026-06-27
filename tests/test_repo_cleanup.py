from __future__ import annotations

from pathlib import Path
from shutil import which
import subprocess


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _tracked_paths(*paths: str) -> list[str]:
    git = which("git")
    assert git is not None
    result = subprocess.run(
        [git, "ls-files", *paths],
        capture_output=True,
        check=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _tracked_files() -> list[str]:
    git = which("git")
    assert git is not None
    result = subprocess.run(
        [git, "ls-files"],
        capture_output=True,
        check=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_generated_artifacts_are_not_tracked() -> None:
    tracked = _tracked_paths(
        ".DS_Store",
        ".tmp",
        "clinical_trials_coverage.json",
        "coverage.json",
        "coverage.xml",
        "examples/metamodeling_validation.nbconvert.ipynb",
        "examples/validation_benchmarking.nbconvert.ipynb",
        "example_net_benefits.csv",
        "example_parameters.csv",
        "pyvoi.egg-info",
        "r-package/voiageR/voiageR.Rcheck",
        "tests/test_cli_coverage.py.bak",
        "tests/test_cli_simple_coverage.py.bak",
        "voiage/clinical_trials.py.bak",
    )

    assert tracked == []

    backup_suffixes = (".bak", ".backup", ".old", ".orig", ".rej", "~")
    tracked_backups = [
        path for path in _tracked_files() if path.endswith(backup_suffixes)
    ]
    assert tracked_backups == []


def test_root_test_artifacts_are_not_tracked() -> None:
    tracked_root_test_artifacts = [
        path
        for path in _tracked_files()
        if "/" not in path
        and (
            path.startswith("test_results_")
            or (path.startswith("test_") and path.endswith(".py"))
        )
    ]

    assert tracked_root_test_artifacts == []


def test_repo_local_ignore_rules_cover_generated_artifacts() -> None:
    gitignore = _read(".gitignore")

    for pattern in (
        ".DS_Store",
        "*.bak",
        "*.backup",
        "*.old",
        "*.orig",
        "*.rej",
        "*~",
        "/test_*.py",
        "/test_results_*",
        "/baseline_performance_results.json",
        "/demo_*.txt",
        "/example_net_benefits.csv",
        "/example_parameters.csv",
        "/jax_audit_results.json",
        "/test.csv",
        "examples/*.nbconvert.ipynb",
        ".tmp/",
        ".ruff_cache/",
        "coverage.json",
        "*_coverage.json",
        "coverage_html_report/",
        "r-package/**/*.Rcheck/",
    ):
        assert pattern in gitignore


def test_cli_example_uses_committed_fixtures_and_fails_fast() -> None:
    example = _read("examples/cli_example.py")

    assert "example_net_benefits.csv" not in example
    assert "example_parameters.csv" not in example
    assert "cli_samples" in example
    assert 'which("voiage")' in example
    assert "check_returncode()" in example


def test_stale_status_mirror_documents_are_not_tracked() -> None:
    tracked = _tracked_paths(
        ".qoder",
        "BRANCH_ARCHITECTURE.md",
        "CHANGES_SUMMARY.md",
        "CHANGELOG_CLI.md",
        "docs/IMPLEMENTATION_COMPLETION_REPORT.md",
        "IMPLEMENTATION_PLAN.md",
        "IMPLEMENTATION_PLAN_JAX_DP_v0.3.0.md",
        "MATURATION_TODO_LIST.md",
        "PHASE2_IMPLEMENTATION_SUMMARY.md",
        "REPOSITORY_STRUCTURE.md",
        "development_plan.md",
        "roadmap.v1.md",
        "roadmap_updated.md",
        "todo_updated.md",
        "todo.v1.md",
        "voiage/.qoder",
        "voiage/README_FOR_QODER_QUEST.md",
        "voiage/REPO_MAP.md",
        "voiage/REPO_MAP_CORRECTED.md",
        "voiage/REPO_MAP_UPDATED.md",
        "voiage/ROADMAP_STATUS.md",
        "voiage/ROADMAP_STATUS_CORRECTED.md",
        "voiage/ROADMAP_STATUS_UPDATED.md",
    )

    assert tracked == []


def test_stale_root_reports_and_one_off_artifacts_are_not_tracked() -> None:
    tracked = _tracked_paths(
        "CERTIFICATE_OF_COMPLETION.md",
        "COMPLETION_CERTIFICATE_FINAL.md",
        "COMPLETION_SUMMARY_WITH_FAULTHANDLER.md",
        "COVERAGE_ACHIEVEMENT_FINAL.md",
        "COVERAGE_ACHIEVEMENT_REPORT.md",
        "COVERAGE_IMPROVEMENT_MISSION_FINAL_REPORT.md",
        "DOCUMENTATION_DEPLOYMENT_VERIFICATION.md",
        "FINAL_ACHIEVEMENT_SUMMARY.md",
        "FINAL_COMPLETION_CERTIFICATE.md",
        "FINAL_COMPLETION_CERTIFICATION.md",
        "FINAL_COMPLETION_REPORT.md",
        "FINAL_COMPLETION_SUMMARY.md",
        "FINAL_COVERAGE_ACHIEVEMENT_SUMMARY.md",
        "FINAL_PROJECT_COMPLETION_CERTIFICATE.md",
        "FINAL_PROJECT_COMPLETION_REPORT.md",
        "HTA_COVERAGE_ACHIEVEMENT_93_PERCENT.md",
        "IMPLEMENTATION_SUMMARY.md",
        "MASTER_SUMMARY.md",
        "PHASE_1_1_COMPLETION_SUMMARY.md",
        "PHASE_1_2_COMPLETION_SUMMARY.md",
        "PHASE_1_3_COMPLETE_SUCCESS.md",
        "PHASE_1_4_COMPLETION_REPORT.md",
        "PHASE_1_COMPLETION_REPORT.md",
        "PROJECT_OFFICIAL_COMPLETION_CERTIFICATE.md",
        "TEST_COVERAGE_FINAL_RESULTS.md",
        "TEST_COVERAGE_IMPROVEMENT_RESULTS.md",
        "docs/FINAL_TEST_COVERAGE_STATUS.md",
        "docs/TEST_COVERAGE_IMPROVEMENT_REPORT.md",
        "paper/DOCUMENTATION_DEPLOYMENT_SETUP_COMPLETE.md",
        "JAX_DEVELOPMENT_GUIDE.md",
        "JAX_INTEGRATION_AUDIT_v0.3.0.md",
        "REVIEWS.md",
        "TESTING_APPROACH.md",
        "advanced_regression.py",
        "benchmark_phase1_3_performance.py",
        "cli_examples.sh",
        "coverage_gap_analysis.py",
        "demo_cli.py",
        "enhanced_performance_benchmark.py",
        "generate_homepage_examples.py",
        "implement_advanced_jax_features_phase1_3.py",
        "jax_audit.py",
        "jax_config.json",
        "jax_dev_setup.py",
        "jax_dev_utils.py",
        "performance_benchmark.py",
        "performance_optimizer.py",
        "simple_benchmark.py",
        "replace_evsi_implementation.py",
        "update_jax_evsi.py",
        "verification_completion.py",
        "baseline_performance_results.json",
        "demo_evpi.txt",
        "demo_evppi.txt",
        "jax_audit_results.json",
        "test.csv",
    )

    assert tracked == []


def test_taskipy_profile_target_exists() -> None:
    pyproject = _read("pyproject.toml")
    expected_task = 'profile = "uv run scalene --cli profile_scalene.py"'

    assert expected_task in pyproject
    assert Path("profile_scalene.py").is_file()


def test_unreleased_changelog_sections_are_unique() -> None:
    changelog = _read("changelog.md").splitlines()
    start = changelog.index("## [Unreleased]")
    end = next(
        index
        for index, line in enumerate(changelog[start + 1 :], start + 1)
        if line.startswith("## [")
    )
    headings = [
        line.removeprefix("### ")
        for line in changelog[start:end]
        if line.startswith("### ")
    ]

    assert headings in (
        ["Added", "Changed", "Fixed"],
        ["Removed", "Added", "Changed", "Fixed"],
    )


def test_tracked_text_files_do_not_embed_local_voiage_paths() -> None:
    searched_suffixes = (
        ".ipynb",
        ".json",
        ".md",
        ".py",
        ".rst",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    )
    local_markers = (
        "/" + "Users/doughnut/GitHub/voiage",
        "file:///" + "Users/edithatogo/GitHub/voiage",
    )
    offenders: list[str] = []

    for path in _tracked_files():
        if not path.endswith(searched_suffixes):
            continue
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in local_markers):
            offenders.append(path)

    assert offenders == []


def test_roadmap_statuses_match_completed_cleanup_state() -> None:
    roadmap = _read("roadmap.md")

    for expected in (
        "Current Status (As of June 2026)",
        "Phase 4: Ecosystem, Community & Future Ports ✅/🔄 **REPOSITORY COMPLETE, EXTERNAL GATES EXPLICIT**",
        "Phase 5: Spec, Fixtures & Polyglot Bindings ✅ **COMPLETE**",
        "Phase 6: Ecosystem Integrations ✅ **COMPLETE**",
        "Phase 7: SOTA VOI Frontier ✅/🔄 **IMPLEMENTED EXPERIMENTAL SURFACE, PARITY GATED**",
        "Phase 8: Rust Core Migration Program ✅/🔄 **FOUNDATION COMPLETE, EXPANSION EVIDENCE-GATED**",
        "external registry, hardware, and speedup evidence gates remain explicit",
    ):
        assert expected in roadmap

    for stale in (
        "Phase 5: Spec, Fixtures & Polyglot Bindings 📋 **PLANNED**",
        "Phase 6: Ecosystem Integrations 📋 **PLANNED**",
        "Phase 7: SOTA VOI Frontier 📋 **PLANNED**",
        "Phase 8: Rust Core Migration Program 📋 **PLANNED**",
        "Language-Agnostic API Specification:\n    *   **Status: `📋 Planned`**",
        "Planning for R/Julia Ports:\n    *   **Status: `📋 Planned`**",
    ):
        assert stale not in roadmap


def test_root_project_state_docs_match_current_protocol() -> None:
    product = _read("product.md")
    tech_stack = _read("tech-stack.md")
    workflow = _read("workflow.md")
    guidelines = _read("product-guidelines.md")
    combined = "\n".join((product, tech_stack, workflow, guidelines))

    for expected in (
        "Python 3.10-3.14",
        "coverage threshold at **>90%**",
        "AGENTS.md",
        "todo.md",
        "roadmap.md",
        "Sphinx remains in the local developer docs gate",
        "Starlight/Astro",
        "external gates",
    ):
        assert expected in combined

    for stale in (
        "Python >=3.8",
        "Policy Analystysts",
        "Financial Analystysts",
        "### In Development\n- Structural Uncertainty VOI",
        "The Plan is the Source of Truth",
        "Target: >80% code coverage",
        "Works correctly on mobile",
    ):
        assert stale not in combined
