"""Local Decision Studio and Business Reporting (#581).

This module provides a local-first, offline-capable Decision Studio for interactive
scenario exploration, expected opportunity loss (EOL), multi-criteria decision
analysis (MCDA), and reproducible business reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import html
import json
from pathlib import Path
from typing import Any

import jsonschema

from voiage.exceptions import raise_input_error

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCHEMA_PATH = (
    _ROOT
    / "specs"
    / "decision-studio"
    / "schemas"
    / "v1"
    / "decision-studio-session.schema.json"
)
_DEFAULT_FIXTURE_PATH = (
    _ROOT
    / "specs"
    / "decision-studio"
    / "fixtures"
    / "normative"
    / "studio-session-fixture.json"
)


def _current_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ScenarioResult:
    """Outcome of a specific scenario evaluation."""

    scenario_name: str
    optimal_choice: str
    expected_payoff: float
    parameter_adjustments: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionStudioSession:
    """Interactive Decision Studio session and reporting artifact.

    Attributes
    ----------
    session_id : str
        Unique session identifier.
    created_at : str
        ISO 8601 timestamp.
    decision_problem_id : str
        Referenced decision problem ID.
    decision_card_id : str
        Referenced decision card ID.
    title : str
        Human-readable title.
    scenarios : list[ScenarioResult]
        Evaluated decision scenarios.
    expected_losses : dict[str, float]
        Expected opportunity loss by alternative.
    voi_summary : dict[str, Any]
        VOI summary metrics (EVPI, EVPPI, choices).
    mcda_evaluation : dict[str, Any] | None
        Optional multi-criteria scoring summary.
    """

    session_id: str
    created_at: str
    decision_problem_id: str
    decision_card_id: str
    title: str
    scenarios: list[ScenarioResult]
    expected_losses: dict[str, float]
    voi_summary: dict[str, Any]
    mcda_evaluation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert session to serializable dictionary."""
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "decision_problem_id": self.decision_problem_id,
            "decision_card_id": self.decision_card_id,
            "title": self.title,
            "scenarios": [asdict(s) for s in self.scenarios],
            "expected_losses": dict(self.expected_losses),
            "voi_summary": dict(self.voi_summary),
        }
        if self.mcda_evaluation is not None:
            d["mcda_evaluation"] = dict(self.mcda_evaluation)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionStudioSession:
        """Instantiate session from dictionary."""
        if not isinstance(data, dict):
            raise_input_error("Session data must be a dictionary.")

        scenarios = [
            ScenarioResult(
                scenario_name=s["scenario_name"],
                optimal_choice=s["optimal_choice"],
                expected_payoff=float(s["expected_payoff"]),
                parameter_adjustments=dict(s.get("parameter_adjustments", {})),
            )
            for s in data.get("scenarios", [])
        ]

        return cls(
            session_id=str(data["session_id"]),
            created_at=str(data.get("created_at", _current_iso_timestamp())),
            decision_problem_id=str(data["decision_problem_id"]),
            decision_card_id=str(data["decision_card_id"]),
            title=str(data["title"]),
            scenarios=scenarios,
            expected_losses=dict(data.get("expected_losses", {})),
            voi_summary=dict(data.get("voi_summary", {})),
            mcda_evaluation=data.get("mcda_evaluation"),
        )

    def render_markdown_report(self) -> str:
        """Generate an executive Markdown summary report."""
        lines = [
            f"# Decision Studio Executive Report: {self.title}",
            f"**Session ID:** `{self.session_id}`  ",
            f"**Decision Problem:** `{self.decision_problem_id}` | **Decision Card:** `{self.decision_card_id}`  ",
            f"**Generated:** {self.created_at}",
            "",
            "## 1. Executive Summary & Recommendation",
            f"- **Optimal Uninformed Choice:** **{self.voi_summary.get('optimal_uninformed_choice', 'N/A')}**",
            f"- **Status Quo Choice:** {self.voi_summary.get('status_quo_choice', 'N/A')}",
            f"- **Expected Value of Perfect Information (EVPI):** ${self.voi_summary.get('evpi', 0.0):,.2f}",
            "",
            "## 2. Scenario Analysis & Robustness",
            "| Scenario | Optimal Choice | Expected Net Payoff | Parameter Adjustments |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for s in self.scenarios:
            params_str = (
                ", ".join(f"{k}={v}" for k, v in s.parameter_adjustments.items())
                or "None"
            )
            lines.append(
                f"| {s.scenario_name} | **{s.optimal_choice}** | ${s.expected_payoff:,.2f} | {params_str} |"
            )

        lines.extend(
            [
                "",
                "## 3. Expected Opportunity Loss (Regret)",
                "| Alternative | Expected Opportunity Loss |",
                "| :--- | :--- |",
            ]
        )

        for alt, loss in self.expected_losses.items():
            lines.append(f"| {alt} | ${loss:,.2f} |")

        if self.mcda_evaluation:
            lines.extend(
                [
                    "",
                    "## 4. Multi-Criteria Decision Analysis (MCDA) Scoring",
                    "| Alternative | Multi-Criteria Weighted Score |",
                    "| :--- | :--- |",
                ]
            )
            for alt, score in self.mcda_evaluation.get("scores", {}).items():
                lines.append(f"| {alt} | {score:.1f} / 100 |")

        return "\n".join(lines)

    def render_html_dashboard(self) -> str:
        """Generate a self-contained, offline-first interactive HTML dashboard."""
        scenarios_rows = "".join(
            f"<tr><td>{html.escape(s.scenario_name)}</td>"
            f"<td><strong>{html.escape(s.optimal_choice)}</strong></td>"
            f"<td>${s.expected_payoff:,.2f}</td></tr>"
            for s in self.scenarios
        )

        loss_rows = "".join(
            f"<tr><td>{html.escape(alt)}</td><td>${loss:,.2f}</td></tr>"
            for alt, loss in self.expected_losses.items()
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(self.title)} - Decision Studio</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }}
    .card {{ background: #fff; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
    h1 {{ color: #0f172a; }}
    h2 {{ color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #f1f5f9; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; background: #e0f2fe; color: #0369a1; }}
  </style>
</head>
<body>
  <div class="card">
    <span class="badge">Local Decision Studio</span>
    <h1>{html.escape(self.title)}</h1>
    <p><strong>Session ID:</strong> {html.escape(self.session_id)} | <strong>Created:</strong> {html.escape(self.created_at)}</p>
    <p><strong>EVPI (Decision Uncertainty Ceiling):</strong> ${self.voi_summary.get("evpi", 0.0):,.2f}</p>
  </div>
  <div class="card">
    <h2>Scenario Robustness Analysis</h2>
    <table>
      <thead><tr><th>Scenario</th><th>Optimal Choice</th><th>Expected Net Payoff</th></tr></thead>
      <tbody>{scenarios_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Expected Opportunity Loss</h2>
    <table>
      <thead><tr><th>Alternative</th><th>Expected Loss (Regret)</th></tr></thead>
      <tbody>{loss_rows}</tbody>
    </table>
  </div>
</body>
</html>"""


def compute_expected_loss(payoffs: dict[str, float]) -> dict[str, float]:
    """Compute expected opportunity loss (EOL / regret) for each alternative."""
    if not payoffs:
        return {}
    max_payoff = max(payoffs.values())
    return {alt: max(0.0, max_payoff - val) for alt, val in payoffs.items()}


def compute_mcda_scores(
    criteria_matrix: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, float]:
    """Compute normalized weighted MCDA scores for alternatives."""
    if not criteria_matrix or not weights:
        return {}
    scores: dict[str, float] = {}
    for alt, crit_vals in criteria_matrix.items():
        total = sum(crit_vals.get(c, 0.0) * w for c, w in weights.items())
        scores[alt] = round(total, 2)
    return scores


def create_decision_studio_session(
    session_id: str,
    title: str,
    decision_problem_id: str,
    decision_card_id: str,
    base_payoffs: dict[str, float],
    scenarios_adjustments: dict[str, dict[str, float]],
    evpi: float,
    status_quo_choice: str,
    evppi: dict[str, float] | None = None,
    mcda_criteria_matrix: dict[str, dict[str, float]] | None = None,
    mcda_weights: dict[str, float] | None = None,
) -> DecisionStudioSession:
    """Build and calculate a complete DecisionStudioSession."""
    if not base_payoffs:
        raise_input_error("base_payoffs dictionary cannot be empty.")

    optimal_uninformed = max(base_payoffs, key=base_payoffs.get)  # type: ignore[arg-type]
    losses = compute_expected_loss(base_payoffs)

    # Compute scenario results
    scenario_results: list[ScenarioResult] = []
    # Always include Base Case
    scenario_results.append(
        ScenarioResult(
            scenario_name="Base Case",
            optimal_choice=optimal_uninformed,
            expected_payoff=float(base_payoffs[optimal_uninformed]),
            parameter_adjustments={},
        )
    )

    for sc_name, adjustments in scenarios_adjustments.items():
        # Apply scenario multiplier to payoffs
        adj_payoffs = {}
        for alt, val in base_payoffs.items():
            multiplier = adjustments.get(alt, adjustments.get("global_multiplier", 1.0))
            adj_payoffs[alt] = val * multiplier
        sc_optimal = max(adj_payoffs, key=adj_payoffs.get)  # type: ignore[arg-type]
        scenario_results.append(
            ScenarioResult(
                scenario_name=sc_name,
                optimal_choice=sc_optimal,
                expected_payoff=float(adj_payoffs[sc_optimal]),
                parameter_adjustments=adjustments,
            )
        )

    voi_summary = {
        "evpi": float(evpi),
        "status_quo_choice": status_quo_choice,
        "optimal_uninformed_choice": optimal_uninformed,
    }
    if evppi:
        voi_summary["evppi"] = evppi

    mcda_eval = None
    if mcda_criteria_matrix and mcda_weights:
        scores = compute_mcda_scores(mcda_criteria_matrix, mcda_weights)
        mcda_eval = {
            "criteria": list(mcda_weights.keys()),
            "weights": mcda_weights,
            "scores": scores,
        }

    return DecisionStudioSession(
        session_id=session_id,
        created_at=_current_iso_timestamp(),
        decision_problem_id=decision_problem_id,
        decision_card_id=decision_card_id,
        title=title,
        scenarios=scenario_results,
        expected_losses=losses,
        voi_summary=voi_summary,
        mcda_evaluation=mcda_eval,
    )


def validate_decision_studio_session(
    session_dict: dict[str, Any], schema_path: Path | None = None
) -> bool:
    """Validate a dictionary against the Decision Studio session schema."""
    s_path = schema_path or _DEFAULT_SCHEMA_PATH
    if not s_path.is_file():
        raise_input_error(f"Schema not found at {s_path}")
    schema = json.loads(s_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=session_dict, schema=schema)
    return True


def load_decision_studio_fixture(
    fixture_path: Path | None = None,
) -> DecisionStudioSession:
    """Load and parse the normative Decision Studio session fixture."""
    f_path = fixture_path or _DEFAULT_FIXTURE_PATH
    if not f_path.is_file():
        raise_input_error(f"Fixture not found at {f_path}")
    data = json.loads(f_path.read_text(encoding="utf-8"))
    return DecisionStudioSession.from_dict(data)
