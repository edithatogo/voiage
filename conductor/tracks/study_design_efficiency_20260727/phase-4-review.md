# Phase 4 automated review

## Scope

Final automated review covered the complete #571 diff, with emphasis on result
deserialization, portfolio semantics, user surfaces, bindings, fixtures,
governance synchronization and repository assurance.

## Findings and remediation

The first pass found no Critical issues, three High issues and two Medium
issues. All were remediated before this checkpoint:

- EVSI/EVPI result deserialization now rejects non-zero EVSI at numerical-zero
  EVPI and materially out-of-bound values even when forged status labels,
  ratios and percentages are internally arithmetical.
- A complete COSS selection-probability map must sum to one within the declared
  result tolerance.
- The portfolio contract now requires primary/secondary metrics, guardrail
  identity and failures, heterogeneous/delayed/interference models,
  sequential monitoring, multiplicity, stopping rules, duration, opportunity
  cost, implementation delay and expected policy change. It reports gross and
  net EVSI/ENBS and selects on net signed ENBS.
- Portfolio deserialization re-derives totals, declared-resource use, binding
  constraints, selected policy changes and stopping rules.
- Static typing warnings were removed, the CLI registry and frontier-promotion
  registry were reconciled, and `changelog.md` now records the user-facing
  experimental family.

Focused remediation tests, Ruff and Basedpyright pass. Full repository,
Conductor, documentation, Rust and hosted evidence are recorded by S19; merge,
scientific promotion, release and issue closure remain separate gates.
