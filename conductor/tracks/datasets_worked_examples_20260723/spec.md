# Track Specification: Datasets And Executable Worked Examples

## Overview

Provide deterministic synthetic, pathological, and rights-cleared public
examples for every stable and supported method.

## Requirements

1. Record source, license, redistribution decision, retrieval time, raw and
   transformed hashes, dictionary, transformation, immutable offline snapshot,
   explicit refresh, expected results, and tolerances.
2. Cover core/design, structural/NMA, preference/heterogeneity/equity/
   implementation, multi-perspective, non-health, ML label acquisition,
   multi-fidelity evaluation, LLM routing, RAG stopping, human escalation,
   test-time compute, drift refresh, and agent tool/delegation.
3. Execute stable/supported examples in all supported languages.
4. Give experimental methods synthetic evidence immediately and public data
   before promotion.
5. Give every example a reproducibility card, assumptions, estimator
   uncertainty, sensitivity analysis, failure cases, accessible output, and
   deterministic offline instructions.

## Industry example and template workstreams

GitHub [#574](https://github.com/edithatogo/voiage/issues/574), record ID
`churn-retention-policy-voi-example`, is the flagship business example. It must
distinguish churn prediction from the value of obtaining or refreshing
information before a retention decision. The decision includes no-contact and
eligible offers/channels; heterogeneous treatment effect; margin and lifetime
value; offer/contact cost; capacity, fatigue, fairness and policy constraints;
implementation delay; calibration and drift; and current versus
post-information retention policies. It covers individual label/survey/feature
acquisition, a field experiment or pilot, model refresh, information-source
portfolio choice, break-even willingness to pay, policy change, regret, and
gross/net VOI without using accuracy or AUC as the value metric.

GitHub [#575](https://github.com/edithatogo/voiage/issues/575), record ID
`industry-domain-example-packs`, provides deterministic, executable packs for
consulting/business cases: marketing acquisition and campaign tests; pricing
and promotion; sales prioritization; finance, credit and collections; treasury
and hedging; demand forecasting and inventory; capacity and operations;
procurement and due diligence; maintenance; workforce and training; and
portfolio, programme or project selection. Every pack states the decision,
information action, utility, constraints, horizon, implementation path,
stakeholder perspectives, reproducibility and decision-maker interpretation.

GitHub [#577](https://github.com/edithatogo/voiage/issues/577), record ID
`domain-template-adapter-registry`, defines a versioned, community-extensible
registry of domain templates and optional adapters. Templates map familiar
industry fields to the canonical Decision Problem without owning numerical
estimators. Entries declare domain, decision pattern, method and capability
requirements, schema version, input/output mappings, units, privacy/rights,
supported languages, provenance, maturity, example IDs and validation status.
Untrusted third-party templates are isolated, validated, and never promoted by
download count alone.

## Rights, privacy, and failure policy

Use SourceRight review, public/redistributable data, open-weight models, or
committed offline outputs. Never silently replace snapshots or require
credentials/personal data. Zero hashes and metadata-only records are invalid
unless explicitly blocked.

## Acceptance criteria

Every stable/supported method has executable synthetic, edge, public, and
cross-language expected evidence from a clean offline installation. Issues
#574, #575 and #577 have schema-valid, rights-reviewed, accessible and
decision-complete evidence; public-data and practitioner review gates remain
explicit.
