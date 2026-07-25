# Round 7 editorial delta review

Date: 26 July 2026

This repository-owned review compares the current JOSS manuscript with the
official JOSS paper format, submission guidance, review criteria, checklist,
and AI-use policy. It is an internal editorial record, not independent peer
review or a JOSS decision.

## Improvements incorporated

- Added verified Research Organization Registry identifiers for all three
  affiliations and a regression check that rejects missing or incorrect
  identifiers.
- Replaced implementation-facing digest language in the Summary with a
  non-specialist description of the analysis record.
- Identified researchers and analysts using simulation models as the immediate
  audience without restricting the stated application domains.
- Explained the reported EVPI as the maximum expected gain from eliminating
  the modelled uncertainty and stated the simulated preference result directly.
- Replaced the abstract “value realisation” label with the assumed share of
  potential benefit reaching practice.
- Expanded the AI-use disclosure to state the nature of assistance and align
  the human-review affirmation with the current JOSS policy.

## Assurance outcome

The manuscript has SHA-256
`2acc713096caaee942fc620a683a027d952049e2b8e4d8275259a0a0ada6199d`
and a deterministic body count of 1,628 words. SourceRight reconciles all 19
citation occurrences; six software or web records retain missing-DOI warnings
for human source review. The selected deterministic Authentext patterns report
no findings. The article contract and focused tests pass locally.

The source change invalidates the earlier rendered-PDF attestation. A fresh
official Open Journals build, visual inspection, and review-only Textstat
record remain required before the current revision can be described as
rendered and reviewed.

## Unchanged external boundaries

This pass does not supply attributable non-author engagement, publish the exact
version 2 release, create an immutable archive for that release, assign the
permanent arXiv identifier, or perform a JOSS submission. Those states remain
separate evidence gates.
