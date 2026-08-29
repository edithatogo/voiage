"""Keep C18 M22-M31 experimental until scientific and promotion gates pass."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
FAMILIES = {
    570: "risk-sensitive-constrained-voi",
    572: "forecast-signal-information",
    582: "information-source-portfolio",
    593: "implementation-information",
    594: "uncertainty-modelling-value",
    596: "event-localized-information",
    597: "belief-state-information",
    598: "signed-social-information",
    599: "heterogeneity-value",
    600: "outcome-conditional-sample-information",
}


def test_c18_families_are_experimental_until_both_gates_pass() -> None:
    contract_freeze = json.loads(
        (
            ROOT
            / "conductor/archive/supported_frontier_method_completion_20260723/contract-freeze.json"
        ).read_text()
    )
    contracts = {item["issue"]: item for item in contract_freeze["contracts"]}
    for issue, family in FAMILIES.items():
        assert contracts[issue]["classification_status"] in {
            "candidate",
            "candidate-census-checkpoint",
            "frozen-experimental",
        }
        capabilities = json.loads(
            (ROOT / "specs/frontier" / family / "v1/capabilities.json").read_text()
        )
        assert (
            capabilities.get("maturity", capabilities.get("method_maturity"))
            == "experimental"
        ), family
        assert capabilities.get("stable_claim_allowed", False) is False, family
