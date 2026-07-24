#!/usr/bin/env python3
"""Generate the method-level scientific evidence registry.

The mappings here are reviewable policy, while the checked-in JSON is the
language-neutral artifact consumed by documentation and binding tooling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
METHODS_PATH = ROOT / "specs" / "software-landscape" / "methods.json"
OUTPUT_PATH = ROOT / "specs" / "software-landscape" / "method-evidence.json"

SOURCES = {
    "ispor-voi-task-force": (
        "Value of Information Analytical Methods: Report 2 of the ISPOR "
        "Value of Information Analysis Emerging Good Practices Task Force",
        "guideline",
        "https://doi.org/10.1016/j.jval.2020.01.004",
        "doi:10.1016/j.jval.2020.01.004",
    ),
    "jackson-voi-overview": (
        "Value of Information Analysis in Models to Inform Health Policy",
        "review-paper",
        "https://doi.org/10.1146/annurev-statistics-040120-010730",
        "doi:10.1146/annurev-statistics-040120-010730",
    ),
    "heath-evppi-review": (
        "A Review of Methods for Analysis of the Expected Value of Information",
        "review-paper",
        "https://doi.org/10.1177/0272989X17697692",
        "doi:10.1177/0272989X17697692",
    ),
    "kunst-evsi-four-methods": (
        "Computing the Expected Value of Sample Information Efficiently",
        "review-paper",
        "https://doi.org/10.1016/j.jval.2020.02.010",
        "doi:10.1016/j.jval.2020.02.010",
    ),
    "fang-mlmc-evppi": (
        "Multilevel and Quasi Monte Carlo Methods for EVPPI",
        "primary-paper",
        "https://doi.org/10.1177/0272989X211026305",
        "doi:10.1177/0272989X211026305",
    ),
    "basu-individualized-care": (
        "Value of Information on Preference Heterogeneity and Individualized Care",
        "primary-paper",
        "https://doi.org/10.1177/0272989X06297393",
        "doi:10.1177/0272989X06297393",
    ),
    "andronis-implementation": (
        "Adjusting Estimates of Expected Value of Information for Implementation",
        "primary-paper",
        "https://doi.org/10.1177/0272989X15614814",
        "doi:10.1177/0272989X15614814",
    ),
    "sadatsafavi-validation-evsi": (
        "Expected Value of Sample Information Calculations for Risk Prediction Model Validation",
        "primary-paper",
        "https://doi.org/10.1177/0272989X251314010",
        "doi:10.1177/0272989X251314010",
    ),
    "zan-bickel-portfolio": (
        "Components of Portfolio Value of Information",
        "primary-paper",
        "https://doi.org/10.1287/deca.2013.0267",
        "doi:10.1287/deca.2013.0267",
    ),
    "lindley-experimental-design": (
        "On a Measure of the Information Provided by an Experiment",
        "primary-paper",
        "https://doi.org/10.1214/aoms/1177728069",
        "doi:10.1214/aoms/1177728069",
    ),
    "frazier-knowledge-gradient": (
        "A Tutorial on Bayesian Optimization",
        "review-paper",
        "https://doi.org/10.1287/educ.2018.0188",
        "doi:10.1287/educ.2018.0188",
    ),
    "go-isaac-robust-eig": (
        "Robust Expected Information Gain for Optimal Bayesian Experimental Design",
        "primary-paper",
        "https://proceedings.mlr.press/v180/go22a.html",
        "url:pmlr-v180-go22a",
    ),
    "r-voi-documentation": (
        "Value of Information Analysis with voi",
        "package-documentation",
        "https://stat.ethz.ch/CRAN/web/packages/voi/vignettes/voi.html",
        "url:r-voi-vignette",
    ),
    "bcea-reference": (
        "BCEA package reference manual",
        "package-documentation",
        "https://stat.ethz.ch/CRAN/web/packages/BCEA/refman/BCEA.html",
        "url:bcea-reference",
    ),
    "voi-web-tools-review": (
        "Web-based tools for value-of-information analysis",
        "review-paper",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7613968/",
        "url:pmc7613968",
    ),
    "real-options-sequential-decisions": (
        "Real Options in the Laboratory: Sequential Investment Decisions",
        "primary-paper",
        "https://doi.org/10.1016/j.jbef.2016.08.002",
        "doi:10.1016/j.jbef.2016.08.002",
    ),
    "nist-decision-pathways": (
        "Value of Information and Decision Pathways: Concepts and Case Studies",
        "review-paper",
        "https://doi.org/10.3389/fenvs.2022.805245",
        "doi:10.3389/fenvs.2022.805245",
    ),
    "pearl-bareinboim-transportability": (
        "External Validity: From Do-Calculus to Transportability Across Populations",
        "primary-paper",
        "https://doi.org/10.1214/14-STS486",
        "doi:10.1214/14-STS486",
    ),
    "hay-value-computation": (
        "Selecting Computations: Theory and Applications",
        "primary-paper",
        "https://arxiv.org/abs/1207.5879",
        "url:arxiv-1207.5879",
    ),
    "madras-learning-to-defer": (
        "Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer",
        "primary-paper",
        "https://arxiv.org/abs/1711.06664",
        "url:arxiv-1711.06664",
    ),
    "pyro-oed": (
        "Pyro Bayesian optimal experimental design documentation",
        "package-documentation",
        "https://docs.pyro.ai/en/stable/contrib.oed.html",
        "url:pyro-oed",
    ),
    "botorch-acquisition": (
        "BoTorch acquisition function documentation",
        "package-documentation",
        "https://botorch.readthedocs.io/en/stable/acquisition.html",
        "url:botorch-acquisition",
    ),
    "voiage-method-contracts": (
        "VOIAGE canonical and frontier method contracts",
        "repository-contract",
        "https://github.com/edithatogo/voiage/tree/main/specs",
        "repo:edithatogo/voiage@specs",
    ),
}

SOURCE_OVERRIDES = {
    "evppi-regression": ("heath-evppi-review",),
    "evppi-nested-mc": ("heath-evppi-review", "fang-mlmc-evppi"),
    "evsi-nested-mc": ("kunst-evsi-four-methods",),
    "evsi-regression": ("kunst-evsi-four-methods",),
    "evsi-moment-matching": ("kunst-evsi-four-methods",),
    "evsi-importance-sampling": ("kunst-evsi-four-methods",),
    "heterogeneity-voi": ("basu-individualized-care",),
    "preference-voi": ("basu-individualized-care",),
    "implementation-voi": ("andronis-implementation",),
    "validation-voi": ("sadatsafavi-validation-evsi",),
    "portfolio-voi": ("zan-bickel-portfolio",),
    "population-evpi": ("ispor-voi-task-force",),
    "optimal-study-design": ("ispor-voi-task-force",),
    "expected-loss-curve": ("bcea-reference",),
    "variance-sensitivity": ("r-voi-documentation",),
    "payer-burden": ("voi-web-tools-review",),
    "real-options-voi": ("real-options-sequential-decisions",),
    "value-of-computation": ("hay-value-computation",),
    "equity-voi": ("nist-decision-pathways", "voiage-method-contracts"),
    "causal-transportability-voi": (
        "pearl-bareinboim-transportability",
        "voiage-method-contracts",
    ),
    "data-quality-voi": ("nist-decision-pathways", "voiage-method-contracts"),
    "monitoring-voi": ("nist-decision-pathways", "voiage-method-contracts"),
    "data-valuation": ("nist-decision-pathways", "voiage-method-contracts"),
    "selective-prediction": ("madras-learning-to-defer", "voiage-method-contracts"),
    "expected-information-gain": ("lindley-experimental-design", "pyro-oed"),
    "robust-expected-information-gain": ("go-isaac-robust-eig",),
    "predictive-information-gain": ("lindley-experimental-design", "pyro-oed"),
    "bayesian-oed": ("lindley-experimental-design", "pyro-oed"),
    "knowledge-gradient": ("frazier-knowledge-gradient", "botorch-acquisition"),
    "entropy-search": ("botorch-acquisition",),
    "active-learning": ("botorch-acquisition",),
    "preference-acquisition": ("botorch-acquisition",),
    "multi-fidelity-acquisition": ("botorch-acquisition",),
}

VOP_METHODS = {
    "directional-evop",
    "pairwise-evop",
    "perspective-optima",
    "perspective-frontier",
    "perspective-switching",
    "perspective-perfect-information",
    "perspective-sample-information",
    "joint-parameter-perspective",
    "robust-perspective-decisions",
}
ML_METHODS = {
    "expected-information-gain",
    "robust-expected-information-gain",
    "predictive-information-gain",
    "bayesian-oed",
    "active-learning",
    "knowledge-gradient",
    "entropy-search",
    "preference-acquisition",
    "multi-fidelity-acquisition",
    "data-valuation",
    "selective-prediction",
}
APPLICATION_METHODS = {
    "llm-routing-voi",
    "rag-acquisition-voi",
    "agent-information-voi",
}
CONTRACT_SYNTHESIS_METHODS = {
    "sequential-voi",
    "equity-voi",
    "causal-transportability-voi",
    "data-quality-voi",
    "monitoring-voi",
    "data-valuation",
    "selective-prediction",
}
CORE_METHODS = {
    "net-benefit",
    "expected-loss",
    "evpi",
    "population-evpi",
    "evppi",
    "evsi",
    "enbs",
    "optimal-study-design",
    "ceac",
    "ceaf",
    "expected-loss-curve",
    "voi-curve",
    "dominance",
    "icer",
    "cea-plane",
    "variance-sensitivity",
    "payer-burden",
}

DISPOSITIONS = {
    "estimand": "canonical-estimand",
    "estimator": "canonical-estimator",
    "workflow": "supported-workflow",
    "visualization": "supported-visualization",
    "related-analysis": "related-analysis",
    "application": "application-contract",
}


def _family(method_id: str) -> str:
    if method_id in VOP_METHODS:
        return "value-of-perspective"
    if method_id in APPLICATION_METHODS:
        return "llm-rag-and-agent-applications"
    if method_id in ML_METHODS:
        return "information-theoretic-design-and-machine-learning"
    if method_id in CORE_METHODS or method_id.startswith(("evppi-", "evsi-")):
        return "stable-decision-voi-and-economic-evaluation"
    return "broader-decision-and-research-value"


def _evidence(method: dict[str, str]) -> dict[str, object]:
    method_id = method["id"]
    family = _family(method_id)
    source_ids = SOURCE_OVERRIDES.get(
        method_id, ("ispor-voi-task-force", "jackson-voi-overview")
    )
    if method_id in VOP_METHODS or method_id in APPLICATION_METHODS:
        source_ids = ("voiage-method-contracts",)
    elif family == "broader-decision-and-research-value":
        source_ids = SOURCE_OVERRIDES.get(
            method_id, ("jackson-voi-overview", "voiage-method-contracts")
        )

    if (
        method_id in VOP_METHODS
        or method_id in APPLICATION_METHODS
        or method_id in CONTRACT_SYNTHESIS_METHODS
    ):
        review_state = "contract-verified"
    elif method["maturity"] == "stable" or method_id in SOURCE_OVERRIDES:
        review_state = "primary-verified"
    else:
        review_state = "triage-required"

    if method_id in VOP_METHODS:
        boundary = "repository-defined-perspective-contract"
    elif method_id in ML_METHODS:
        boundary = "information-theoretic-only"
    elif method["class"] in {"visualization", "related-analysis"}:
        boundary = "derived-decision-output"
    else:
        boundary = "requires-decision-problem-v2"

    if method["class"] in {"visualization", "related-analysis"}:
        required_fields = ["alternatives", "objective", "provenance"]
    elif method_id in ML_METHODS:
        required_fields = [
            "uncertainty",
            "information_actions",
            "objective",
            "provenance",
        ]
    else:
        required_fields = [
            "alternatives",
            "uncertainty",
            "information_actions",
            "objective",
            "population",
            "time_horizon",
            "provenance",
        ]
    if method_id in VOP_METHODS:
        required_fields.insert(4, "perspectives")

    if method["maturity"] == "stable":
        promotion_gate = "none"
    elif method_id in VOP_METHODS:
        promotion_gate = "external-scientific-review"
    elif method_id in APPLICATION_METHODS or method_id in CONTRACT_SYNTHESIS_METHODS:
        promotion_gate = "method-contract-and-fixtures"
    elif review_state == "primary-verified":
        promotion_gate = "independent-replication"
    else:
        promotion_gate = "primary-literature-review"

    if method_id in VOP_METHODS:
        note = (
            "Repository-defined VOP contract. It is not an alias for preference "
            "heterogeneity, equity weighting, subgroup analysis, or ordinary "
            "scenario analysis, and requires independent scientific review "
            "before stable promotion."
        )
    elif method_id in APPLICATION_METHODS:
        note = (
            "Application contract only. A valid analysis must instantiate "
            "DecisionProblemV2 and report acquisition cost and net value; the "
            "application name does not create a new estimand."
        )
    elif method_id in ML_METHODS:
        note = (
            "Information-design or acquisition method. It is economic VOI only "
            "when embedded in DecisionProblemV2 with downstream utility or loss "
            "and acquisition cost."
        )
    elif method["class"] in {"visualization", "related-analysis"}:
        note = (
            "Decision output or adjacent analysis retained for library parity; "
            "it is not a distinct VOI estimand."
        )
    else:
        note = (
            "Decision-theoretic method requiring explicit alternatives, "
            "uncertainty, information action, objective, population, horizon, "
            "and provenance."
        )

    return {
        "method_id": method_id,
        "family": family,
        "source_ids": list(source_ids),
        "review_state": review_state,
        "disposition": DISPOSITIONS[method["class"]],
        "decision_boundary": boundary,
        "required_decision_fields": required_fields,
        "promotion_gate": promotion_gate,
        "note": note,
    }


def render() -> str:
    """Render the canonical JSON registry from reviewed mappings."""
    methods = json.loads(METHODS_PATH.read_text(encoding="utf-8"))["methods"]
    sources = [
        {
            "id": source_id,
            "title": values[0],
            "kind": values[1],
            "url": values[2],
            "verification": (
                "repository-verified"
                if values[1] == "repository-contract"
                else "primary-verified"
            ),
            "identifier": values[3],
        }
        for source_id, values in SOURCES.items()
    ]
    payload = {
        "schema_version": "1.1.0",
        "reviewed_on": "2026-07-24",
        "review_due": "2026-10-23",
        "sources": sources,
        "coverage": [_evidence(method) for method in methods],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    """Write the registry or verify the checked-in projection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        return 0 if OUTPUT_PATH.read_text(encoding="utf-8") == rendered else 1
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
