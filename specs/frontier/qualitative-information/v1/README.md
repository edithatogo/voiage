# Qualitative information assessment experimental contract v1

This directory defines the planned v1.3.0 portable contract for issue #558.
The workflow preserves human-entered ordinal judgements and produces
deterministic priority groups, recommendation classes, complete dissent and
incomplete states. It does not calculate EVPI, EVPPI, EVSI, utility, currency,
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
provenance. They remain unverified until a declared human action verifies them;
an AI actor cannot approve an assessment. Redacted and unavailable source
content is represented only by stable markers. Reviewer identities in these
fixtures are synthetic roles, not personal data.

## Language disposition

Python is the experimental reference executor. Rust, R and Julia are
unsupported until they validate and execute these exact fixtures. Mojo remains
external. Portable schema readability is not execution parity.

## Promotion boundary

The schemas and synthetic fixtures are repository evidence only. Practitioner
usability, accessibility conformance, privacy/ethics review, scientific naming,
hosted wheel evidence, stable promotion, release and issue closure are separate
gates.
