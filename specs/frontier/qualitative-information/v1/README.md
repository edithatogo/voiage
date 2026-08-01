# Qualitative information assessment experimental contract v1

This directory defines the planned v1.3.0 portable contract for issue #558.
The workflow preserves human-entered ordinal judgements and produces
deterministic priority groups, recommendation classes, complete dissent and
incomplete or unverified states. The substantive criteria and ordinal classes
are accountable human judgements; the runtime validates and groups them but
does not derive a recommendation from a hidden rule or score. It does not
calculate EVPI, EVPPI, EVSI, utility, currency,
weighted scores or cardinal distance between classes.

## Contract files

- `schemas/qualitative-information-assessment.schema.json` validates the input.
- `schemas/qualitative-information-result.schema.json` validates the output.
- `schemas/qualitative-information-audit-event.schema.json` validates each
  chained history event.
- `schemas/qualitative-information-rendering.schema.json` validates a
  deterministic accessible text rendering receipt.
- `fixtures/normative/` contains the exact synthetic reference case.
- `fixtures/cases/` contains reproducible dissent, incomplete and adversarial
  mutations over the normative case.

## Human and AI boundary

AI judgements and audit events require provider, model-version and input
provenance. Human-verified AI contributions link to an accountable human review
event; an AI or system actor cannot approve an assessment. Only a final,
current-version accountable approval can make the workflow complete. Redacted
question text and source-linked rationales are replaced by stable markers in
results and renderings. Reviewer identities in these fixtures are synthetic
roles, not personal data.

Each audit event records the prior event identifier and digest, its own
canonical SHA-256 digest, and an assessment-content digest. The validator
recomputes the event chain and requires the final event to bind the current
assessment snapshot. This is tamper-evident repository evidence, not a digital
signature or an external append-only ledger.

## Language disposition

Python is the experimental reference executor. Rust, R and Julia are
unsupported until they validate and execute these exact fixtures. Mojo remains
external. Portable schema readability is not execution parity.

## Promotion boundary

The schemas and synthetic fixtures are repository evidence only. Practitioner
usability, accessibility conformance, privacy/ethics review, scientific naming,
hosted wheel evidence, stable promotion, release and issue closure are separate
gates.
