"""Keep dependency automation and coverage delivery within reviewed boundaries."""

from fnmatch import fnmatchcase
import json
from pathlib import Path
import re

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_codecov_upload_uses_real_authoritative_report_and_oidc() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["coverage-report"]
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    steps = job["steps"]
    generation = next(
        i
        for i, step in enumerate(steps)
        if "tox -e coverage_report" in step.get("run", "")
    )
    upload_index = next(
        i
        for i, step in enumerate(steps)
        if step.get("uses", "").startswith("codecov/codecov-action@")
    )
    upload = steps[upload_index]
    assert generation < upload_index
    assert not upload.get("continue-on-error", False)
    assert "if" not in upload
    assert re.fullmatch(r"codecov/codecov-action@[0-9a-f]{40}", upload["uses"])
    assert upload["with"]["files"] == "./coverage.xml"
    assert upload["with"]["use_oidc"] is True
    assert upload["with"]["fail_ci_if_error"] is True
    assert upload["with"]["disable_search"] is True
    assert "token" not in upload["with"]


def test_renovate_inherits_owner_policy_without_changing_library_support() -> None:
    config = json.loads((ROOT / "renovate.json").read_text())
    assert config["extends"][0] == "github>edithatogo/renovate-config"
    assert config["rangeStrategy"] == "auto"
    rules = config["packageRules"]
    assert all(rule.get("automergeType") != "branch" for rule in rules)
    assert all("matchPackagePatterns" not in rule for rule in rules)
    library = next(
        rule
        for rule in rules
        if "Preserve published library" in rule.get("description", "")
    )
    assert {"cargo", "pep621", "pip_requirements", "pip_setup"} <= set(
        library["matchManagers"]
    )
    assert library["rangeStrategy"] == "auto"
    assert library["automerge"] is False
    final = rules[-1]
    assert set(config["enabledManagers"]) <= set(final["matchManagers"])
    assert final["automerge"] is False
    assert final["automergeType"] == "pr"
    engines = next(rule for rule in rules if rule.get("matchDepTypes") == ["engines"])
    assert engines["enabled"] is False
    interpreter = next(
        rule for rule in rules if "python-version" in rule.get("matchDatasources", [])
    )
    assert {"python-version", "node-version", "rust-version"} <= set(
        interpreter["matchDatasources"]
    )
    assert interpreter["matchUpdateTypes"] == ["pin"]
    assert interpreter["enabled"] is False


@pytest.mark.parametrize(
    "package", ["typer", "typing_extensions", "types-requests", "types-setuptools"]
)
def test_linting_group_does_not_capture_runtime_or_stub_packages(package: str) -> None:
    config = json.loads((ROOT / "renovate.json").read_text())
    lint = next(
        rule for rule in config["packageRules"] if rule.get("groupSlug") == "linting"
    )
    assert lint["matchPackageNames"] == ["ruff", "ty"]
    assert package not in lint["matchPackageNames"]


def test_renovate_uses_hosted_dashboard_schema() -> None:
    config = json.loads((ROOT / "renovate.json").read_text())
    # The hosted app still expects a boolean dashboard switch (issue #1305).
    assert isinstance(config.get("dependencyDashboard", True), bool)
    assert config["dependencyDashboardOSVVulnerabilitySummary"] == "unresolved"


def test_renovate_keeps_archival_environments_outside_live_dependency_updates() -> None:
    config = json.loads((ROOT / "renovate.json").read_text())
    # Renovate discovers package files before applying dependency package rules.
    patterns = config["pep621"]["ignorePaths"]
    for archived in (
        "paper/reproduction-environment/pyproject.toml",
        "paper/reproduction-environment/uv.lock",
        "paper/reproduction-environment/python315/pyproject.toml",
        "paper/reproduction-environment/python315/uv.lock",
        "specs/submission-readiness/dependency-frontier-environment-20260829/pyproject.toml",
    ):
        assert any(fnmatchcase(archived, pattern) for pattern in patterns)
    for live in (
        "pyproject.toml",
        "uv.lock",
        "requirements-joss.txt",
        "docs/astro-site/package.json",
    ):
        assert not any(fnmatchcase(live, pattern) for pattern in patterns)

    cargo_patterns = config["cargo"]["ignorePaths"]
    frozen_native = (
        "conductor/tracks/scientific_ecosystem_alignment_20261029/"
        "installed-adapters/native/Cargo.toml"
    )
    assert any(fnmatchcase(frozen_native, pattern) for pattern in cargo_patterns)
    live_manifests = [
        ROOT / "rust" / "Cargo.toml",
        ROOT / "rust" / "fuzz" / "Cargo.toml",
        ROOT / "r-package" / "voiageR" / "src" / "rust" / "Cargo.toml",
        *(ROOT / "rust" / "crates").glob("*/Cargo.toml"),
    ]
    assert len(live_manifests) > 3
    for manifest in live_manifests:
        assert manifest.is_file()
        relative_path = manifest.relative_to(ROOT).as_posix()
        assert not any(
            fnmatchcase(relative_path, pattern) for pattern in cargo_patterns
        )


def test_frozen_native_lock_retains_independent_advisory_coverage() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/rust-security.yml").read_text()
    )
    events = workflow.get("on", workflow.get(True))
    native = (
        "conductor/tracks/scientific_ecosystem_alignment_20261029/"
        "installed-adapters/native/"
    )
    for event in ("pull_request", "push"):
        paths = events[event]["paths"]
        assert native + "Cargo.toml" in paths
        assert native + "Cargo.lock" in paths
        assert "renovate.json" in paths
    assert events["schedule"]
    job = workflow["jobs"]["cargo-deny"]
    assert not job.get("continue-on-error", False)
    steps = job["steps"]
    python = next(
        step
        for step in steps
        if step.get("name") == "Set up qualified Python for retained lock audit"
    )
    assert re.fullmatch(r"actions/setup-python@[0-9a-f]{40}", python["uses"])
    assert python["with"]["python-version"] == "3.15.1"
    audit = next(
        step
        for step in steps
        if step.get("name") == "Audit retained native adapter lock"
    )
    command = audit["run"]
    assert "python scripts/audit_retained_native_lock.py prepare" in command
    assert command.count("python scripts/audit_retained_native_lock.py verify") == 2
    assert "cargo metadata --format-version 1" in command
    assert (
        'cargo deny --locked --manifest-path "$scratch/Cargo.toml" '
        "--config rust/deny.toml check -D warnings advisories"
    ) in command
    assert command.index("cargo metadata") < command.index(" verify ")
    assert command.index(" verify ") < command.index("cargo deny")
    assert "if" not in audit
    assert not audit.get("continue-on-error", False)
    assert "ignore = []" in (ROOT / "rust/deny.toml").read_text()


def test_renovate_keeps_proptest_within_the_declared_rust_floor() -> None:
    config = json.loads((ROOT / "renovate.json").read_text())
    rule = next(
        rule
        for rule in config["packageRules"]
        if rule.get("matchPackageNames") == ["proptest"]
    )
    assert rule["matchManagers"] == ["cargo"]
    assert rule["allowedVersions"] == "<1.12.0"
    assert 'rust-version = "1.85"' in (ROOT / "rust/Cargo.toml").read_text()
    for crate in ("voiage-domain", "voiage-numerics"):
        manifest = (ROOT / "rust/crates" / crate / "Cargo.toml").read_text()
        assert 'proptest = ">=1.7, <1.12"' in manifest


def test_governed_python_upgrades_require_separate_qualification() -> None:
    config = json.loads((ROOT / "renovate.json").read_text())
    rule = next(
        rule
        for rule in config["packageRules"]
        if rule.get("matchPackageNames") == ["scipy", "jax", "mutmut"]
    )
    assert rule["matchManagers"] == ["pep621"]
    assert rule["groupName"] is None
    assert rule["groupSlug"] is None
    assert rule["dependencyDashboardApproval"] is True
    assert rule["automerge"] is False


def test_renovate_holds_sha2_at_the_workspace_msrv_ceiling() -> None:
    config = json.loads((ROOT / "renovate.json").read_text())
    rule = next(
        rule
        for rule in config["packageRules"]
        if rule.get("matchPackageNames") == ["sha2"]
    )
    assert rule["matchManagers"] == ["cargo"]
    assert rule["allowedVersions"] == "<=0.11.0"
    # sha2 0.11.1 raises the toolchain floor to rustc 1.88, above the 1.85 MSRV.
    msrv = re.search(
        r'^rust-version\s*=\s*"([0-9.]+)"',
        (ROOT / "rust" / "Cargo.toml").read_text(),
        re.MULTILINE,
    )
    assert msrv is not None
    assert msrv.group(1) == "1.85"
    lock = (ROOT / "rust" / "Cargo.lock").read_text()
    locked = re.search(
        r'name = "sha2"\r?\nversion = "([0-9.]+)"',
        lock,
    )
    assert locked is not None
    assert locked.group(1) == "0.11.0"


def test_release_automation_policy_has_distinct_fail_closed_states() -> None:
    policy = json.loads(
        (
            ROOT
            / "conductor/tracks/agent_safe_engineering_20260907/automation-policy.json"
        ).read_text(encoding="utf-8")
    )
    assert policy["schema_version"] == "1.0"
    states = set(policy["states"])
    assert {"pending", "verified", "staged", "published", "partial", "failed"} <= states
    transitions = {tuple(edge) for edge in policy["allowed_transitions"]}
    assert ("staged", "published") in transitions
    assert ("staged", "partial") in transitions
    assert ("partial", "failed") in transitions
    assert ("pending", "published") not in transitions
    assert ("partial", "published") not in transitions
    assert len(policy["required_evidence"]) == len(set(policy["required_evidence"]))
    assert {
        "artifact_digest_mismatch",
        "signer_unverified",
        "exception_expired",
        "platform_receipt_missing",
        "publication_partial",
        "replay_artifact_mismatch",
    } <= set(policy["failure_reasons"])

    guards = policy["failure_guards"]
    assert ["staged", "published"] in guards["artifact_digest_mismatch"][
        "blocked_transitions"
    ]
    assert ["staged", "published"] in guards["signer_unverified"]["blocked_transitions"]


def test_release_automation_policy_binds_immutable_artifacts_and_no_publish_authority() -> (
    None
):
    policy = json.loads(
        (
            ROOT
            / "conductor/tracks/agent_safe_engineering_20260907/automation-policy.json"
        ).read_text(encoding="utf-8")
    )
    invariants = set(policy["invariants"])
    assert (
        "a release version is bound to one immutable source commit and artifact digest"
        in invariants
    )
    assert (
        "replay cannot publish different artifact bytes under an existing version"
        in invariants
    )
    assert any("never authorizes" in invariant for invariant in invariants)


def test_release_automation_policy_rejects_staged_publish_for_digest_and_signer_failures() -> (
    None
):
    policy = json.loads(
        (
            ROOT
            / "conductor/tracks/agent_safe_engineering_20260907/automation-policy.json"
        ).read_text(encoding="utf-8")
    )
    publish_edge = ["staged", "published"]
    for reason in ("artifact_digest_mismatch", "signer_unverified"):
        assert publish_edge in policy["failure_guards"][reason]["blocked_transitions"]
