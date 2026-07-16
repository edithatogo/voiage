"""CLI contract tests for AI-assisted evidence triage."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage.cli import app


def test_ai_assisted_evidence_triage_cli(tmp_path: Path) -> None:
    input_file = tmp_path / "triage.json"
    input_file.write_text(
        json.dumps(
            {
                "relevance_labels": [1, 1, 0, 1, 0, 1],
                "triage_scores": [0.95, 0.4, 0.8, 0.7, 0.2, 0.6],
                "decision_impacts": [10, 8, 4, 6, 3, 5],
                "reviewer_time_minutes": [10, 8, 6, 7, 4, 5],
                "extraction_error_rates": [0.01, 0.1, 0.05, 0.02, 0.2, 0.03],
                "triage_threshold": 0.5,
                "audit_sample_rate": 0.5,
                "human_override_rate": 0.25,
                "model_drift": 0.1,
                "automation_cost": 2,
                "audit_cost_per_item": 0.5,
                "reviewer_cost_per_minute": 0.1,
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["--format", "json", "calculate-ai-assisted-evidence-triage", str(input_file)],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["analysis_type"] == "value_of_ai_assisted_evidence_triage"
    assert payload["method_maturity"] == "fixture-backed"
