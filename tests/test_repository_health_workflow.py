"""Contract test for the repository health workflow."""

from pathlib import Path

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
