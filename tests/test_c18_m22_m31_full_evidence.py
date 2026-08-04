"""Audit complete repository-owned evidence registration for C18 families."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
FAMILIES = {
    "risk-sensitive-constrained-voi": "tests/test_risk_sensitive_voi.py",
    "forecast-signal-information": "tests/test_forecast_signal_information.py",
    "information-source-portfolio": "tests/test_information_source_portfolio.py",
    "implementation-information": "tests/test_implementation_information.py",
    "uncertainty-modelling-value": "tests/test_uncertainty_modelling_value.py",
    "event-localized-information": "tests/test_event_localized_information.py",
    "belief-state-information": "tests/test_belief_state_information.py",
    "signed-social-information": "tests/test_signed_social_information.py",
    "heterogeneity-value": "tests/test_heterogeneity.py",
    "outcome-conditional-sample-information": "tests/test_outcome_conditional_sample_information.py",
}


def test_every_c18_family_registers_reference_property_and_pathology_evidence() -> None:
    for family, test_path in FAMILIES.items():
        manifest_path = ROOT / "specs/frontier" / family / "v1/fixtures/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["normative"], family
        assert all(item.get("reference") for item in manifest["normative"]), family
        assert manifest.get("pathologies"), family
        assert manifest.get("property_tests"), family
        assert (ROOT / test_path).is_file(), test_path
        for evidence_path in manifest["pathologies"] + manifest["property_tests"]:
            candidate = evidence_path.split("::", 1)[0]
            candidate_path = ROOT / candidate
            if not candidate_path.is_file() and not candidate.startswith("tests/"):
                candidate_path = ROOT / "specs/frontier" / family / "v1/fixtures" / candidate
            if "/" in candidate or candidate.startswith("tests/"):
                assert candidate_path.is_file(), (family, evidence_path)
