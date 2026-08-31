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


def test_renovate_keeps_archival_environments_outside_live_dependency_updates() -> None:
    config = json.loads((ROOT / "renovate.json").read_text())
    archive_rule = next(
        rule
        for rule in config["packageRules"]
        if rule.get("matchFileNames")
        == [
            "paper/reproduction-environment/**",
            "specs/submission-readiness/dependency-frontier-environment-20260829/**",
        ]
    )
    assert archive_rule["enabled"] is False
    patterns = archive_rule["matchFileNames"]
    for archived in (
        "paper/reproduction-environment/pyproject.toml",
        "paper/reproduction-environment/uv.lock",
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
