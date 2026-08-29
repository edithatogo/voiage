"""Fail-closed contract for the repository's polyglot CI/CD architecture."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.github_dependency_snapshot import snapshot

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _workflow(name: str) -> dict[str, object]:
    loaded = yaml.safe_load(_text(name))
    assert isinstance(loaded, dict)
    return loaded


def test_required_pr_workflows_support_merge_queue() -> None:
    """Required checks must run against GitHub's synthesized merge commit."""
    for name in (
        "ci.yml",
        "bindings-ci.yml",
        "codeql.yml",
        "dependency-review.yml",
        "operational-assurance.yml",
        "polyglot-assurance.yml",
    ):
        assert "merge_group:" in _text(name), name


def test_release_uses_pep740_and_exact_testpypi_promotion() -> None:
    """TestPyPI is an attested multi-version promotion gate, not a token smoke."""
    release = _text("release.yml")
    assert "attestations: false" not in release
    assert "Verify TestPyPI distribution attestations" in release
    assert "pypi-attestations" in release
    assert "pypi-attestations==0.0.30" in release
    assert "test.pypi.org/integrity/voiage/${PYTHON_VERSION}" in release
    assert "--provenance-file" in release
    assert '"reviewed-dist/$filename"' in release
    assert "verify pypi --staging" not in release
    assert 'python: ["3.12", "3.13", "3.14"]' in release
    assert "Download reviewed release payload" in release
    assert "Compare TestPyPI bytes with reviewed payload" in release
    assert release.index("Publish Reviewed Draft to TestPyPI") < release.index(
        "Publish to PyPI"
    )


def test_release_attests_build_and_polyglot_sbom_subjects() -> None:
    """Every promoted subject carries both provenance and its polyglot SBOM."""
    release = _text("release.yml")
    assert "workflow_call:" in release
    assert "actions/attest-build-provenance@" in release
    assert "actions/attest-sbom@" in release
    assert "polyglot.sbom.cdx.json" in release
    assert "release-manifest.json" in release
    assert "gh attestation verify" in release


def test_each_retained_language_has_native_assurance() -> None:
    """Python, Rust/C, R, Julia, and Astro/TypeScript remain explicit."""
    combined = "\n".join(
        _text(name)
        for name in (
            "ci.yml",
            "bindings-ci.yml",
            "ffi-sanitizers.yml",
            "rust-security.yml",
            "docs.yml",
        )
    )
    required = (
        'python: ["3.12", "3.13", "3.14"]',
        "cargo clippy --workspace --all-targets --all-features --locked",
        "cargo test --workspace --all-features --locked",
        "bash scripts/run_ffi_sanitizers.sh",
        "R CMD check --as-cran",
        "for attempt in 1 2 3",
        "rcmdcheck::rcmdcheck",
        "RETICULATE_PYTHON",
        "Aqua.test_all",
        "VOIAGE_RUN_JULIA_AQUA",
        "julia --project=.",
        "pnpm run check",
        "pnpm run build",
    )
    for marker in required:
        assert marker in combined, marker


def test_rust_publication_uses_short_lived_trusted_publishing() -> None:
    """crates.io publication must not depend on a stored registry token."""
    workflow = _text("rust-crates-release.yml")
    assert "rust-lang/crates-io-auth-action@" in workflow
    assert "id-token: write" in workflow
    assert "secrets.CARGO_REGISTRY_TOKEN" not in workflow


def test_binding_releases_consume_the_shared_release_identity() -> None:
    """R and Julia release validation is anchored to the common manifest."""
    workflow = _text("bindings-release.yml")
    assert "Verify shared immutable release manifest" in workflow
    assert "release-manifest.json" in workflow
    assert ".source_commit == $source_commit" in workflow
    assert ".metadata.component.version == $version" in workflow
    assert 'toolchain: "1.85"' in workflow
    assert "ubuntu-24.04" in workflow
    assert workflow.count("needs: verify-base-release") == 2
    assert "Publish immutable R source and manual release" in workflow


def test_shared_numerical_corpus_crosses_every_runtime_boundary() -> None:
    """The same reference corpus must reach every retained callable surface."""
    workflow = _text("bindings-ci.yml")
    for marker in (
        "Cross-language differential conformance",
        "specs/numerical-reference/v1/evpi-cases.json",
        "Python public API",
        "Rust crate API",
        "C ABI",
        "R installed package",
        "Julia installed package",
        "uv run --extra ci pytest tests/test_numerical_reference_cases.py",
    ):
        assert marker in workflow


def test_binding_matrix_isolates_julia_and_separates_r_test_contexts() -> None:
    """Hosted parity must not inherit local Julia or R development state."""
    workflow = _text("bindings-ci.yml")

    assert "JULIA_DEPOT_PATH" in workflow
    assert "Run R package-development checks" in workflow
    assert "Run installed R native and numerical-reference checks" in workflow
    assert "test-native-ffi.R" in workflow
    assert "test-zz-numerical-reference.R" in workflow


def test_arm64_is_observed_before_release_promotion() -> None:
    """Preview ARM64 coverage stays non-required until evidence supports it."""
    workflow = _text("polyglot-assurance.yml")
    assert "ubuntu-24.04-arm" in workflow
    assert "continue-on-error: true" in workflow
    assert "ARM64 observation" in workflow


def test_renovate_is_the_only_update_pr_producer() -> None:
    """GitHub alerts may remain inputs, but Dependabot must not open PRs."""
    assert not (ROOT / ".github" / "dependabot.yml").exists()
    renovate = json.loads((ROOT / "renovate.json").read_text(encoding="utf-8"))
    assert "config:best-practices" in renovate["extends"]
    assert ":dependencyDashboard" in renovate["extends"]
    assert renovate["osvVulnerabilityAlerts"] is True
    assert {"cargo", "github-actions", "npm", "pep621"} <= set(
        renovate["enabledManagers"]
    )
    assert "custom.regex" in renovate["enabledManagers"]
    custom_managers = renovate["customManagers"]
    assert any(
        "r-package/voiageR/DESCRIPTION" in manager["managerFilePatterns"][0]
        for manager in custom_managers
    )
    assert any(
        "bindings/julia/Project.toml" in manager["managerFilePatterns"][0]
        for manager in custom_managers
    )
    assert renovate["minimumReleaseAge"] == "14 days"


def test_dependency_graph_and_immutable_release_contracts_are_explicit() -> None:
    """Resolved dependencies and final assets have hosted integrity controls."""
    assurance = _text("polyglot-assurance.yml")
    release = _text("release.yml")
    quality = (
        ROOT
        / "docs/astro-site/src/content/docs/developer-guide/quality-and-security.mdx"
    ).read_text(encoding="utf-8")
    assert "Dependency submission" in assurance
    assert "contents: write" in assurance
    assert "immutable release" in release.lower()
    assert "immutable releases" in quality.lower()


def test_push_dependency_inventory_has_a_root_component() -> None:
    """The push-only SBOM composition path must produce metadata.component."""
    assurance = _text("polyglot-assurance.yml")
    assert "cyclonedx-py environment .venv" in assurance
    assert "--pyproject pyproject.toml --mc-type library" in assurance
    assert "compose_polyglot_sbom.py compose" in assurance


def test_ci_has_one_required_local_authority() -> None:
    """Tox and the harness are authoritative; nox remains a convenience."""
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "tox is the required local CI authority" in contributing
    assert "nox is an optional developer interface" in contributing


def test_dependency_snapshot_preserves_polyglot_package_urls() -> None:
    """Submitted snapshots retain every package ecosystem represented by purl."""
    document = {
        "components": [
            {"bom-ref": "numpy", "purl": "pkg:pypi/numpy@2.3.0"},
            {"bom-ref": "serde", "purl": "pkg:cargo/serde@1.0.0"},
            {"purl": "pkg:cran/testthat@3.2.0"},
            {"purl": "pkg:julia/JSON@0.21.4"},
            {"name": "component-without-purl"},
        ],
        "dependencies": [{"ref": "numpy", "dependsOn": ["serde"]}],
    }
    payload = snapshot(document, sha="a" * 40, ref="refs/heads/main")
    manifests = payload["manifests"]
    assert isinstance(manifests, dict)
    resolved = manifests["polyglot.sbom.cdx.json"]["resolved"]
    assert set(resolved) == {
        "pkg:pypi/numpy@2.3.0",
        "pkg:cargo/serde@1.0.0",
        "pkg:cran/testthat@3.2.0",
        "pkg:julia/JSON@0.21.4",
    }
    assert resolved["pkg:pypi/numpy@2.3.0"]["dependencies"] == ["pkg:cargo/serde@1.0.0"]
    assert resolved["pkg:cargo/serde@1.0.0"]["relationship"] == "indirect"
