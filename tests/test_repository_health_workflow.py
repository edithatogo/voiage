"""Contract test for the repository health workflow."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def test_repository_health_checkout_contract() -> None:
    """RepoHealth needs history, and both checks can run with read-only access."""

    workflow = ROOT / ".github" / "workflows" / "repository-health.yml"
    config = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    events = config[True]  # PyYAML interprets the unquoted "on" key as a boolean.
    assert config["permissions"] == {"contents": "read"}
    assert events["pull_request"]["branches"] == ["main"]
    assert "pull_request_target" not in events

    for job in ("alint", "repohealth"):
        checkout = next(
            step["with"]
            for step in config["jobs"][job]["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["persist-credentials"] is False
        if job == "repohealth":
            assert checkout["fetch-depth"] == 0


def test_local_credentials_are_git_ignored() -> None:
    """Local credential files and directories stay out of Git."""

    credential_paths = (
        ".env",
        "local/certificate.pem",
        "local/signing.key",
        "credentials/session",
    )
    git = shutil.which("git")
    assert git is not None
    result = subprocess.run(
        [git, "check-ignore", "--stdin"],
        cwd=ROOT,
        input="\n".join(credential_paths) + "\n",
        capture_output=True,
        check=True,
        text=True,
    )
    assert tuple(result.stdout.splitlines()) == credential_paths


@pytest.mark.parametrize(
    ("text", "has_stray_whitespace"),
    [
        ("ordinary text\n", False),
        ("hard break  \n", False),
        ("hard break  \r\n", False),
        ("single trailing space \n", True),
        ("three trailing spaces   \n", True),
        ("tab in trailing whitespace\t  \n", True),
        ("whitespace-only line\n  \n", True),
        ("single trailing space at EOF ", True),
    ],
)
def test_alint_markdown_whitespace(text: str, has_stray_whitespace: bool) -> None:
    """Intentional Markdown hard breaks preserve the trailing-space check."""

    config = yaml.safe_load((ROOT / ".alint.yml").read_text(encoding="utf-8"))
    rule = next(
        rule
        for rule in config["rules"]
        if rule["id"] == "markdown-no-stray-trailing-whitespace"
    )
    assert bool(re.search(rule["pattern"], text)) is has_stray_whitespace
