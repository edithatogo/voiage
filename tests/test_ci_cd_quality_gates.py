"""Validation tests for CI/CD quality gates configuration.

This module validates that the CI/CD quality gates are properly configured
according to the strict CI/CD quality gates policy, including:
- Linting and formatting gates
- Type checking gates
- Coverage thresholds
- Property-based testing
- Mutation testing
- Security checks
- Documentation builds
- Binding language-native gates
"""

import os
import yaml
import pytest
from pathlib import Path

# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent
GITHUB_WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"
TOX_INI = PROJECT_ROOT / "tox.ini"
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"


class TestCICDQualityGatesConfiguration:
    """Validate CI/CD quality gates configuration."""

    def test_github_workflows_exist(self):
        """Test that required GitHub Actions workflows exist."""
        required_workflows = [
            "ci.yml",
            "bindings-ci.yml",
            "codeql.yml",
            "docs.yml",
            "release.yml",
            "sbom.yml",
        ]
        
        for workflow in required_workflows:
            workflow_path = GITHUB_WORKFLOWS_DIR / workflow
            assert workflow_path.exists(), f"Required workflow {workflow} not found"

    def test_ci_workflow_has_required_jobs(self):
        """Test that CI workflow has required quality gate jobs."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_config = yaml.safe_load(f)
        
        required_jobs = [
            "lint",
            "test-unit",
            "coverage-report",
            "test-integration",
            "test-e2e",
            "docs",
        ]
        
        existing_jobs = list(ci_config.get("jobs", {}).keys())
        for job in required_jobs:
            assert job in existing_jobs, f"Required CI job {job} not found in CI workflow"

    def test_coverage_threshold_is_90_percent(self):
        """Test that coverage threshold is set to 90%."""
        # Check in pyproject.toml
        with open(PYPROJECT_TOML) as f:
            pyproject_content = f.read()
        
        assert "cov-fail-under=90" in pyproject_content, "Coverage threshold not set to 90% in pyproject.toml"
        assert "fail_under = 90" in pyproject_content, "Coverage threshold not set to 90% in pyproject.toml [tool.coverage.report]"

    def test_tox_has_required_environments(self):
        """Test that tox.ini has required test environments."""
        with open(TOX_INI) as f:
            tox_content = f.read()
        
        required_envs = [
            "lint",
            "typecheck",
            "docs",
            "coverage_report",
            "frontier-contract",
            "version-sync",
        ]
        
        for env in required_envs:
            assert env in tox_content, f"Required tox environment {env} not found"

    def test_linting_rules_configured(self):
        """Test that linting rules are properly configured."""
        with open(PYPROJECT_TOML) as f:
            pyproject_content = f.read()
        
        # Check for Ruff configuration
        assert "[tool.ruff]" in pyproject_content, "Ruff configuration not found"
        assert "[tool.ruff.lint]" in pyproject_content, "Ruff lint configuration not found"
        
        # Check for Bandit security rules
        assert "bandit" in pyproject_content.lower(), "Bandit security checker not configured"

    def test_type_checking_configured(self):
        """Test that type checking is configured."""
        with open(TOX_INI) as f:
            tox_content = f.read()
        
        assert "typecheck" in tox_content, "Type checking environment not found in tox.ini"
        
        with open(PYPROJECT_TOML) as f:
            pyproject_content = f.read()
        
        assert "ty" in pyproject_content.lower(), "Type checker (ty) not found in dependencies"

    def test_property_based_tests_exist(self):
        """Test that property-based tests exist and are discoverable."""
        tests_dir = PROJECT_ROOT / "tests"
        
        # Check for property-based test files
        property_test_files = [
            "test_property_based.py",
            "test_property_invariants.py",
            "test_cli_invariants.py",
        ]
        
        for test_file in property_test_files:
            test_path = tests_dir / test_file
            assert test_path.exists(), f"Property-based test file {test_file} not found"

    def test_mutation_testing_configured(self):
        """Test that mutation testing is configured."""
        with open(PYPROJECT_TOML) as f:
            pyproject_content = f.read()
        
        assert "[tool.mutmut]" in pyproject_content, "Mutmut configuration not found"
        assert "mutmut" in pyproject_content.lower(), "Mutmut not in dependencies"

    def test_security_checks_configured(self):
        """Test that security checks are configured."""
        with open(PYPROJECT_TOML) as f:
            pyproject_content = f.read()
        
        # Check for Bandit
        assert "bandit" in pyproject_content.lower(), "Bandit security checker not configured"
        
        # Check for CodeQL workflow
        codeql_workflow = GITHUB_WORKFLOWS_DIR / "codeql.yml"
        assert codeql_workflow.exists(), "CodeQL workflow not found"

    def test_documentation_build_configured(self):
        """Test that documentation build is configured."""
        with open(TOX_INI) as f:
            tox_content = f.read()
        
        assert "docs" in tox_content, "Documentation build environment not found in tox.ini"
        
        # Check for docs workflow
        docs_workflow = GITHUB_WORKFLOWS_DIR / "docs.yml"
        assert docs_workflow.exists(), "Documentation workflow not found"

    def test_binding_ci_configured(self):
        """Test that binding CI is configured for all polyglot targets."""
        bindings_workflow = GITHUB_WORKFLOWS_DIR / "bindings-ci.yml"
        assert bindings_workflow.exists(), "Bindings CI workflow not found"
        
        with open(bindings_workflow) as f:
            bindings_config = yaml.safe_load(f)
        
        required_binding_jobs = [
            "typescript",
            "go",
            "rust",
            "julia",
            "dotnet",
            "r",
        ]
        
        existing_jobs = list(bindings_config.get("jobs", {}).keys())
        for job in required_binding_jobs:
            assert job in existing_jobs, f"Required binding job {job} not found"

    def test_python_version_matrix(self):
        """Test that Python version matrix is comprehensive."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_config = yaml.safe_load(f)
        
        test_unit_job = ci_config["jobs"]["test-unit"]
        python_versions = test_unit_job["strategy"]["matrix"]["python"]
        
        required_versions = ["3.10", "3.11", "3.12", "3.13", "3.14"]
        for version in required_versions:
            assert version in python_versions, f"Python version {version} not in test matrix"

    def test_weekly_expensive_gates_configured(self):
        """Test that expensive gates are configured for weekly scheduled runs."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_content = f.read()
        
        # Check for scheduled trigger in the raw YAML content
        assert "schedule:" in ci_content, "Scheduled trigger not found in CI workflow"
        assert "cron:" in ci_content, "Cron schedule not found in CI workflow"
        
        # Load YAML to check for expensive jobs
        with open(ci_workflow_path) as f:
            ci_config = yaml.safe_load(f)
        
        # Check for expensive jobs that run on schedule
        expensive_jobs = ["test-mutation", "profile"]
        for job in expensive_jobs:
            assert job in ci_config["jobs"], f"Expensive job {job} not found"
            job_config = ci_config["jobs"][job]
            # Check the raw content for the schedule condition (handle both quote styles)
            assert ("if: github.event_name == 'schedule'" in str(job_config) or 
                    "if: \"github.event_name == 'schedule'\"" in str(job_config) or
                    'if: "github.event_name == \'schedule\'"' in str(job_config) or
                    "github.event_name == 'schedule'" in str(job_config)), \
                f"Job {job} not configured to run only on schedule"

    def test_coverage_gate_preserves_90_percent_floor(self):
        """Test that coverage gate preserves the 90% floor."""
        with open(PYPROJECT_TOML) as f:
            pyproject_content = f.read()
        
        # Ensure coverage threshold is not weakened below 90%
        assert "cov-fail-under=90" in pyproject_content, "Coverage floor weakened below 90%"
        assert "fail_under = 90" in pyproject_content, "Coverage floor weakened below 90%"

    def test_integration_and_e2e_tests_configured(self):
        """Test that integration and E2E tests are configured."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_config = yaml.safe_load(f)
        
        assert "test-integration" in ci_config["jobs"], "Integration test job not found"
        assert "test-e2e" in ci_config["jobs"], "E2E test job not found"

    def test_benchmark_regression_checks_configured(self):
        """Test that benchmark regression checks are configured."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_config = yaml.safe_load(f)
        
        assert "benchmark-regression" in ci_config["jobs"], "Benchmark regression job not found"

    def test_frontier_contract_validation_configured(self):
        """Test that frontier contract validation is configured."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_config = yaml.safe_load(f)
        
        assert "frontier-contract" in ci_config["jobs"], "Frontier contract validation job not found"

    def test_version_sync_validation_configured(self):
        """Test that version synchronization validation is configured."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_config = yaml.safe_load(f)
        
        assert "version-sync" in ci_config["jobs"], "Version sync validation job not found"


class TestQualityGatePolicyCompliance:
    """Validate compliance with strict CI/CD quality gates policy."""

    def test_no_silent_omission_of_expensive_gates(self):
        """Test that expensive gates are documented rather than silently omitted."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_config = yaml.safe_load(f)
        
        # Check that expensive jobs have explicit conditions
        expensive_jobs = ["test-mutation", "profile"]
        for job in expensive_jobs:
            if job in ci_config["jobs"]:
                job_config = ci_config["jobs"][job]
                # Should have explicit schedule condition
                assert "if" in job_config, f"Expensive job {job} lacks explicit execution condition"

    def test_rust_and_binding_gates_preserved(self):
        """Test that Rust and binding language-native gates are preserved."""
        bindings_workflow = GITHUB_WORKFLOWS_DIR / "bindings-ci.yml"
        
        with open(bindings_workflow) as f:
            bindings_config = yaml.safe_load(f)
        
        # Check that each binding has appropriate quality gates
        rust_job = bindings_config["jobs"]["rust"]
        assert "cargo fmt --check" in str(rust_job), "Rust formatting check missing"
        assert "cargo clippy" in str(rust_job), "Rust clippy check missing"
        assert "cargo test" in str(rust_job), "Rust test check missing"

    def test_dependency_conflicts_prevented(self):
        """Test that base install avoids dependency conflicts."""
        with open(PYPROJECT_TOML) as f:
            pyproject_content = f.read()
        
        # Check that heavy dependencies are in optional groups
        if "[project.optional-dependencies]" in pyproject_content:
            # Find the dev section which should contain heavy dependencies
            if "dev = [" in pyproject_content:
                dev_section_start = pyproject_content.find("dev = [")
                dev_section_end = pyproject_content.find("]", dev_section_start)
                dev_section = pyproject_content[dev_section_start:dev_section_end]
                
                # Heavy/optional dependencies should be in dev group
                heavy_deps = ["mutmut", "scalene"]
                for dep in heavy_deps:
                    assert dep in dev_section.lower(), f"Heavy dependency {dep} should be in dev dependencies"
            else:
                # Fallback: check they're not in base dependencies
                base_deps_section = pyproject_content.split("dependencies = [")[1].split("]")[0]
                heavy_deps = ["mutmut", "scalene"]
                for dep in heavy_deps:
                    assert dep not in base_deps_section.lower(), f"Heavy dependency {dep} should not be in base dependencies"
        else:
            # If no optional dependencies section, heavy deps should not be in base dependencies
            base_deps_section = pyproject_content.split("dependencies = [")[1].split("]")[0]
            heavy_deps = ["mutmut", "scalene"]
            for dep in heavy_deps:
                assert dep not in base_deps_section.lower(), f"Heavy dependency {dep} should not be in base dependencies"

    def test_public_api_compatibility_preserved(self):
        """Test that changes preserve public API compatibility."""
        # This is a policy check - actual compatibility testing would be in integration tests
        # Here we validate that the test infrastructure exists to catch breaking changes
        tests_dir = PROJECT_ROOT / "tests"
        
        # Check for API contract tests
        contract_test_files = [
            "test_core_api_contract_validator.py",
            "test_core_api_fixture_runner_contract.py",
        ]
        
        for test_file in contract_test_files:
            test_path = tests_dir / test_file
            assert test_path.exists(), f"API contract test {test_file} not found"


class TestCICDDocumentation:
    """Validate CI/CD documentation and governance."""

    def test_ci_cd_docs_exist(self):
        """Test that CI/CD documentation exists."""
        # Check for contributing guidelines
        contributing_path = PROJECT_ROOT / "CONTRIBUTING.md"
        assert contributing_path.exists(), "CONTRIBUTING.md not found"
        
        with open(contributing_path) as f:
            contributing_content = f.read()
        
        # Should mention CI/CD or quality gates
        assert "ci" in contributing_content.lower() or "test" in contributing_content.lower(), \
            "CONTRIBUTING.md should document CI/CD process"

    def test_gate_matrix_distinguishes_triggers(self):
        """Test that gate matrix distinguishes PR, scheduled, release, and manual paths."""
        ci_workflow_path = GITHUB_WORKFLOWS_DIR / "ci.yml"
        
        with open(ci_workflow_path) as f:
            ci_content = f.read()
        
        # Check for different trigger types in the raw YAML content
        assert "pull_request:" in ci_content, "PR trigger not configured"
        assert "push:" in ci_content, "Push trigger not configured"
        assert "schedule:" in ci_content, "Schedule trigger not configured"