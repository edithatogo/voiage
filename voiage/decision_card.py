"""Decision Registry, Decision Cards, and Signed Result Bundles (#580).

This module turns VOI analyses into versioned, verifiable, auditable business
decision records, supporting lifecycle state, residual risk tracking, governance
and human sign-off, HTML/Markdown exports, and SHA-256 cryptographic bundle integrity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
import json
from pathlib import Path
from typing import Any

import jsonschema

from voiage.exceptions import raise_input_error

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCHEMA_PATH = (
    _ROOT / "specs" / "decision-cards" / "schemas" / "v1" / "decision-card.schema.json"
)


def _producer_code_version() -> str:
    """Identify a new card's installed producer without a release literal."""
    try:
        return f"v{package_version('voiage')}"
    except PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class DecisionProblemSnapshot:
    """Snapshot of the core decision problem context."""

    problem_id: str
    title: str
    alternatives: list[str]
    criterion: str


@dataclass(frozen=True)
class SelectedPolicy:
    """Details of the chosen action or operational policy."""

    name: str
    rationale: str
    expected_net_benefit: float
    baseline_comparison: str = ""


@dataclass(frozen=True)
class InformationValuation:
    """Summary of VOI metrics computed for the decision."""

    evpi: float
    evppi: dict[str, float] = field(default_factory=dict)
    evsi: dict[str, float] = field(default_factory=dict)
    enbs: dict[str, float] = field(default_factory=dict)
    recommended_information_action: str = ""


@dataclass(frozen=True)
class ResidualUncertainty:
    """Analysis of residual uncertainty and risk post-decision."""

    top_drivers: list[str] = field(default_factory=list)
    risk_quantiles: dict[str, float] = field(default_factory=dict)
    sensitivity_summary: str = ""


@dataclass(frozen=True)
class HumanApproval:
    """Record of formal human stakeholder review and sign-off."""

    approver: str
    approved_at: str
    rationale: str


@dataclass(frozen=True)
class Governance:
    """Ownership, review, expiry, and lifecycle governance."""

    owner: str
    reviewers: list[str]
    human_approval: HumanApproval | None = None
    expiry_date: str = ""
    refresh_cadence: str = ""


@dataclass(frozen=True)
class Lineage:
    """Cryptographic hashes, data versions, and code lineage."""

    model_version: str
    input_hash: str
    dataset_version: str = "v1.0.0"
    code_version: str = field(default_factory=_producer_code_version)
    bundle_hash: str = ""


@dataclass(frozen=True)
class DecisionCard:
    """Auditable Decision Card record."""

    decision_id: str
    version: str
    title: str
    status: str  # draft, proposed, approved, superseded, expired
    created_at: str
    decision_problem: DecisionProblemSnapshot
    selected_policy: SelectedPolicy
    information_valuation: InformationValuation
    governance: Governance
    lineage: Lineage
    residual_uncertainty: ResidualUncertainty = field(
        default_factory=ResidualUncertainty
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert DecisionCard to JSON-serializable dictionary."""
        governance_data: dict[str, Any] = {
            "owner": self.governance.owner,
            "reviewers": list(self.governance.reviewers),
            "expiry_date": self.governance.expiry_date,
            "refresh_cadence": self.governance.refresh_cadence,
        }
        if self.governance.human_approval is not None:
            governance_data["human_approval"] = asdict(self.governance.human_approval)
        data: dict[str, Any] = {
            "decision_id": self.decision_id,
            "version": self.version,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "decision_problem": asdict(self.decision_problem),
            "selected_policy": asdict(self.selected_policy),
            "information_valuation": asdict(self.information_valuation),
            "residual_uncertainty": asdict(self.residual_uncertainty),
            "governance": governance_data,
            "lineage": asdict(self.lineage),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionCard:
        """Construct DecisionCard from a dictionary."""
        if not isinstance(data, dict):
            raise_input_error("DecisionCard data must be a dictionary.")

        prob = data["decision_problem"]
        dp = DecisionProblemSnapshot(
            problem_id=str(prob["problem_id"]),
            title=str(prob["title"]),
            alternatives=list(prob["alternatives"]),
            criterion=str(prob["criterion"]),
        )

        pol = data["selected_policy"]
        sp = SelectedPolicy(
            name=str(pol["name"]),
            rationale=str(pol["rationale"]),
            expected_net_benefit=float(pol["expected_net_benefit"]),
            baseline_comparison=str(pol.get("baseline_comparison", "")),
        )

        inf = data["information_valuation"]
        iv = InformationValuation(
            evpi=float(inf["evpi"]),
            evppi={str(k): float(v) for k, v in inf.get("evppi", {}).items()},
            evsi={str(k): float(v) for k, v in inf.get("evsi", {}).items()},
            enbs={str(k): float(v) for k, v in inf.get("enbs", {}).items()},
            recommended_information_action=str(
                inf.get("recommended_information_action", "")
            ),
        )

        res = data.get("residual_uncertainty", {})
        ru = ResidualUncertainty(
            top_drivers=list(res.get("top_drivers", [])),
            risk_quantiles={
                str(k): float(v) for k, v in res.get("risk_quantiles", {}).items()
            },
            sensitivity_summary=str(res.get("sensitivity_summary", "")),
        )

        gov = data["governance"]
        ha_data = gov.get("human_approval")
        ha = (
            HumanApproval(
                approver=str(ha_data["approver"]),
                approved_at=str(ha_data["approved_at"]),
                rationale=str(ha_data["rationale"]),
            )
            if ha_data
            else None
        )
        gv = Governance(
            owner=str(gov["owner"]),
            reviewers=list(gov["reviewers"]),
            human_approval=ha,
            expiry_date=str(gov.get("expiry_date", "")),
            refresh_cadence=str(gov.get("refresh_cadence", "")),
        )

        lin = data["lineage"]
        lg = Lineage(
            model_version=str(lin["model_version"]),
            input_hash=str(lin["input_hash"]),
            dataset_version=str(lin.get("dataset_version", "v1.0.0")),
            # The reader's installed version cannot establish historical provenance.
            code_version=str(lin.get("code_version", "unknown")),
            bundle_hash=str(lin.get("bundle_hash", "")),
        )

        return cls(
            decision_id=str(data["decision_id"]),
            version=str(data["version"]),
            title=str(data["title"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            decision_problem=dp,
            selected_policy=sp,
            information_valuation=iv,
            residual_uncertainty=ru,
            governance=gv,
            lineage=lg,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize DecisionCard to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> DecisionCard:
        """Deserialize DecisionCard from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the decision card contents."""
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_markdown(self) -> str:
        """Render the DecisionCard as clean executive Markdown."""
        ha_str = (
            f"**Approved By:** {self.governance.human_approval.approver} ({self.governance.human_approval.approved_at})\n"
            f"> {self.governance.human_approval.rationale}"
            if self.governance.human_approval
            else "*Pending Formal Approval*"
        )
        return f"""# Decision Card: {self.title}

- **Decision ID:** `{self.decision_id}` (v{self.version})
- **Status:** `{self.status.upper()}`
- **Created:** {self.created_at}
- **Owner:** {self.governance.owner}

---

## 1. Selected Action / Policy
- **Chosen Alternative:** **{self.selected_policy.name}**
- **Expected Net Benefit:** `${self.selected_policy.expected_net_benefit:,.2f}`
- **Rationale:** {self.selected_policy.rationale}

## 2. Information Valuation (VOI)
- **EVPI (Value of Perfect Information):** `${self.information_valuation.evpi:,.2f}`
- **Recommended Data/Trial Action:** {self.information_valuation.recommended_information_action or "None (Proceed to deployment)"}

## 3. Governance & Sign-off
{ha_str}

## 4. Verification & Lineage
- **Model Version:** `{self.lineage.model_version}`
- **Input Hash:** `{self.lineage.input_hash}`
- **Card Hash:** `{self.compute_hash()}`
"""

    def to_html(self) -> str:
        """Render the DecisionCard as self-contained HTML."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Decision Card - {self.title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; color: #1f2937; line-height: 1.5; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 24px; max-width: 800px; margin: 0 auto; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .badge {{ display: inline-block; padding: 4px 8px; font-size: 12px; font-weight: bold; border-radius: 4px; background: #e0e7ff; color: #3730a3; }}
    h1 {{ margin-top: 0; font-size: 24px; }}
    h2 {{ font-size: 18px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; margin-top: 24px; }}
    .metric {{ font-size: 20px; font-weight: bold; color: #059669; }}
    .hash {{ font-family: monospace; font-size: 11px; background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="card">
    <span class="badge">{self.status.upper()}</span>
    <h1>{self.title}</h1>
    <p><strong>Decision ID:</strong> {self.decision_id} (v{self.version}) | <strong>Owner:</strong> {self.governance.owner}</p>

    <h2>Selected Policy</h2>
    <p><strong>{self.selected_policy.name}</strong></p>
    <p>Expected Net Benefit: <span class="metric">${self.selected_policy.expected_net_benefit:,.2f}</span></p>
    <p>{self.selected_policy.rationale}</p>

    <h2>Value of Information</h2>
    <p>Expected Value of Perfect Information (EVPI): <strong>${self.information_valuation.evpi:,.2f}</strong></p>
    <p>Recommended Next Step: <em>{self.information_valuation.recommended_information_action or "Execute policy"}</em></p>

    <h2>Lineage & Integrity</h2>
    <p>Card Hash: <span class="hash">{self.compute_hash()}</span></p>
  </div>
</body>
</html>"""


@dataclass
class DecisionBundle:
    """A signed, tamper-evident bundle combining a DecisionCard and inputs."""

    card: DecisionCard
    input_payload: dict[str, Any]
    bundle_hash: str = ""

    def __post_init__(self) -> None:
        """Compute initial bundle hash if not provided."""
        if not self.bundle_hash:
            self.bundle_hash = self.compute_bundle_hash()

    def compute_bundle_hash(self) -> str:
        """Calculate SHA-256 hash over card + input payload."""
        combined = {
            "card": self.card.to_dict(),
            "input_payload": self.input_payload,
        }
        serialized = json.dumps(combined, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that the bundle hash matches its contents."""
        return self.compute_bundle_hash() == self.bundle_hash


def create_decision_card(
    decision_id: str,
    title: str,
    decision_problem: DecisionProblemSnapshot,
    selected_policy: SelectedPolicy,
    information_valuation: InformationValuation,
    owner: str,
    reviewers: list[str],
    version: str = "1.0.0",
    status: str = "draft",
    residual_uncertainty: ResidualUncertainty | None = None,
    human_approval: HumanApproval | None = None,
    model_version: str = "1.0.0",
    input_hash: str = "",
) -> DecisionCard:
    """Create a new DecisionCard with ISO 8601 timestamps."""
    now_iso = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    gov = Governance(
        owner=owner,
        reviewers=reviewers,
        human_approval=human_approval,
    )
    lin = Lineage(
        model_version=model_version,
        input_hash=input_hash or hashlib.sha256(b"empty_input").hexdigest(),
    )
    return DecisionCard(
        decision_id=decision_id,
        version=version,
        title=title,
        status=status,
        created_at=now_iso,
        decision_problem=decision_problem,
        selected_policy=selected_policy,
        information_valuation=information_valuation,
        residual_uncertainty=residual_uncertainty or ResidualUncertainty(),
        governance=gov,
        lineage=lin,
    )


def validate_decision_card(
    card_dict: dict[str, Any], schema_path: Path | None = None
) -> bool:
    """Validate a raw dictionary against the Decision Card JSON schema."""
    s_path = schema_path or _DEFAULT_SCHEMA_PATH
    if not s_path.is_file():
        raise_input_error(f"Decision card schema not found at {s_path}")

    schema = json.loads(s_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=card_dict, schema=schema)
    return True
