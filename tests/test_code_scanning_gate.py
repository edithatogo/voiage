"""Tests for the repository-owned code-scanning gate."""

from __future__ import annotations

from email.message import Message
from io import BytesIO
import json
from pathlib import Path
import sys
import urllib.error

import yaml

import scripts.check_code_scanning_alerts as gate

ROOT = Path(__file__).parents[1]


def test_scorecard_keeps_evidence_when_publication_fails() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/scorecard.yml").read_text())
    steps = workflow["jobs"]["analysis"]["steps"]
    action = next(step for step in steps if step.get("name") == "Run Scorecard")

    assert action["uses"] == (
        "ossf/scorecard-action@23f22cd43608bae7a1f1fa2603d7314dee9538a5"
    )
    assert action["with"] == {
        "results_file": "scorecard-results.json",
        "results_format": "sarif",
        "publish_results": True,
        "fail_on_publish_error": True,
    }
    for name in ("Upload Scorecard SARIF", "Upload Scorecard results"):
        upload = next(step for step in steps if step.get("name") == name)
        assert upload["if"] == "${{ always() }}"


def test_workflows_use_the_repository_owned_retrying_gate() -> None:
    github_context = "${{ github.token }}"
    for workflow in ("codeql.yml", "scorecard.yml"):
        content = (ROOT / ".github" / "workflows" / workflow).read_text()
        assert "scripts/check_code_scanning_alerts.py" in content
        assert "edithatogo/.github/.github/actions/code-scanning-gate" not in content
        jobs = yaml.safe_load(content)["jobs"].values()
        steps = [
            step
            for job in jobs
            for step in job.get("steps", [])
            if "scripts/check_code_scanning_alerts.py" in step.get("run", "")
        ]
        assert len(steps) == 1
        assert steps[0]["env"]["GITHUB_TOKEN"] == github_context
        assert "--token" not in steps[0]["run"].split()


def test_gate_reads_credential_from_environment(monkeypatch) -> None:
    marker = "fixture-github-token"
    monkeypatch.setenv("GITHUB_TOKEN", marker)
    monkeypatch.setattr(sys, "argv", ["check_code_scanning_alerts.py"])

    assert gate.parse_args().token == marker


def test_blocking_alerts_are_limited_to_current_commit_and_high_severity() -> None:
    alerts = [
        {
            "number": 1,
            "rule": {"id": "high-rule", "security_severity_level": "high"},
            "most_recent_instance": {
                "commit_sha": "current",
                "message": {"text": "blocking"},
            },
            "html_url": "https://example.test/1",
        },
        {
            "number": 2,
            "rule": {"id": "medium-rule", "security_severity_level": "medium"},
            "most_recent_instance": {"commit_sha": "current"},
        },
        {
            "number": 3,
            "rule": {"id": "critical-rule", "security_severity_level": "critical"},
            "most_recent_instance": {"commit_sha": "other"},
        },
    ]

    hits = gate.blocking_alerts(alerts, commit_sha="current")

    assert [(hit.number, hit.rule_id, hit.severity) for hit in hits] == [
        (1, "high-rule", "high")
    ]


def test_transient_api_failures_are_retried(monkeypatch) -> None:
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps([]).encode()

    def fake_urlopen(_request, timeout):
        nonlocal attempts
        assert timeout == 30
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(
                "https://example.test", 503, "temporary", Message(), BytesIO()
            )
        return Response()

    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gate.time, "sleep", lambda _seconds: None)

    assert (
        gate._fetch_page(
            "owner/repo",
            "token",
            1,
            100,
            deadline=gate.time.monotonic() + 5,
            retry_attempts=2,
        )
        == []
    )
    assert attempts == 2
