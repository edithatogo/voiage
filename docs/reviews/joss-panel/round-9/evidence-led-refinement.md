# Round 9 JOSS evidence-led refinement

Date: 26 July 2026

This review used the current `paper.md`, official JOSS submission and review
criteria, the repository's JOSS article contract, SourceRight, Authentext, and
three independent roles: JOSS screening, value-of-information methodology, and
plain-language scientific editing. It is repository-owned analysis, not a JOSS
editorial decision.

## Findings incorporated

- Rewrote the summary around the decisions VOI supports before introducing
  implementation detail.
- Simplified the cross-language design account while retaining the separate
  Rust-core trade-off and current R and Julia limitations.
- Distinguished the generating expected net benefit from the fixed simulation
  mean. The generating value is zero; the 10,000-draw mean is -15.860 units.
- Defined EVPI, EVPPI, and EVSI values as applying to each eligible future
  person whose decision could use the evidence, not to each trial participant.
- Labelled the normal--normal study as an algebraic illustration that does not
  specify follow-up, missing data, treatment switching, or a proposed trial.
- Stated that evidence delay is fixed independently of sample size and that the
  discount rate applies to future decision opportunities.
- Clarified that linear EVPPI is correctly specified only for this independent,
  linear example and does not validate nonlinear or correlated applications.
- Replaced the suggestion that marginal EVPPI identifies a research priority
  with the narrower conclusion that it compares values of perfect resolution.
- Added implementation and independently derived test locators to the
  worked-example claim ledger.

## Findings retained as gates

- Demonstrated completed research use remains absent and cannot be repaired by
  rewriting the synthetic example.
- Non-author engagement, external use, or collaborative input remains absent.
- The exact reviewed v2 release and immutable archive do not yet exist.
- Human source-by-source citation confirmation and the complete JOSS AI-use
  attestation remain human gates.
- A permanent arXiv identifier remains the author's sequence preference rather
  than a JOSS requirement.

## Assurance result

The revised source contains 1,583 body words, within the repository's
1,568--1,632 contract and JOSS's 750--1,750 range. SourceRight reconciles all 18
citation occurrences with no citation issue and retains six bounded warnings
for software or web references without DOIs. Authentext reports no selected
pattern finding. A current-source Open Journals PDF, Textstat report, and visual
review are required before this round can be closed.
