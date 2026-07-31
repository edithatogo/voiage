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

The exact-commit re-review found three further High issues and one Medium
issue. These were also remediated before S19 closure:

- every advanced model now has a provenance-bearing, fail-closed assurance
  declaring either no portfolio effect or prior incorporation in COSS;
- incremental opportunity and delay costs require a provenance-bearing
  declaration that COSS research cost excludes them;
- tolerance ties are constructed once against the fixed global maximum, with
  chain-tolerance and permutation assurance; and
- malformed candidate or constraint objects again raise the governed
  `InputError` boundary rather than leaking `AttributeError`.

The third pass reported no Critical or High findings. Its two Medium findings
were closed by rejecting non-finite allocator tolerances at the domain boundary
and by applying non-blank identifier constraints to assurance-provenance keys
and values. The allocator documentation now describes its actual total-cost
tie breaker.

Focused remediation tests, Ruff and Basedpyright pass. Full repository,
Conductor, documentation, Rust and hosted evidence are recorded by S19; merge,
scientific promotion, release and issue closure remain separate gates.
