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
pattern finding.

Hosted workflow run
[30166336066](https://github.com/edithatogo/voiage/actions/runs/30166336066)
built the source at commit `d8e3d006501be42f4416ddeec8946625f39fee98`
with the Open Journals toolchain. Artifact `8621698185` has GitHub digest
`sha256:086bd0e899ef9aa5dae0c260588e85a9d46771dc77d23b0e872b60f280dc1378`;
the six-page PDF has SHA-256
`c02b5f4442a9cadac08b0a25be2bacb3f4c217b48ae00e5d202f8115b34fe0c5`.
All six rendered pages were inspected without finding clipping, overlap,
missing content, broken page furniture, or an unreadable figure. Textstat
reported Flesch reading ease 36.4655 and Flesch--Kincaid grade 11.4712 as
review-only evidence, not a scientific-quality threshold. This closes the
repository-owned round-nine manuscript review; the external gates above remain.
