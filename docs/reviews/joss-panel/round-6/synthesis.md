# Round 6 JOSS review synthesis

This synthesis records the repository's AI-assisted internal review. It is
not a JOSS editorial decision, independent peer review, or evidence of
submission or acceptance.

## Disposition

The manuscript-level contract passes at 1,615 words, within the exact
1,600 ±2% target. All ten contracted sections are present in the required
order and satisfy their section budgets. The numerical reviewer independently
reproduced the reported EVPI, EVPPI, EVSI, ENBS, bootstrap, and sensitivity
results. SourceRight reconciles all 19 citation occurrences with 15
bibliography records, and the selected deterministic Authentext checks have no
blocking findings.

The article is ready for an official Open Journals build and author review.
The full submission package is not yet JOSS-ready because release, human, and
external screening gates remain open.

## Accepted priorities

1. Lead with the research decisions supported by EVPI, EVPPI, EVSI, and ENBS,
   then explain the software architecture through its consequences for users.
2. Describe the Rust-centred Python, R, and Julia boundaries exactly, including
   R's optional Python-backed methods and the additional native-library steps
   for R and Julia.
3. Specify the normal--normal study model and distinguish the repository's
   derivation from broader methodological sources.
4. Report the worked-example results, uncertainty method, conditional
   scenarios, estimator behaviour, and limitations without implying
   dominance, additive EVPPI components, or an optimised study size.
5. Add a stable figure identifier, tabular source data, structured
   reproduction inputs, and fail-closed regeneration.
6. Reconcile material claims to repository or external evidence; verify every
   reference record against authoritative sources; preserve missing-DOI
   warnings for legitimate software and web records.
7. State AI assistance in the manuscript and retain comprehensive human
   review, modification, validation, and accountability as an explicit author
   attestation rather than an automated inference.
8. Bound the context claim to the implemented versioned specification and
   digest-linked typed results; scalar convenience functions do not carry the
   full context.
9. State cross-platform evidence by language and distinguish scientific checks
   from the complete stable-promotion policy.

## Recommendations not adopted in full

- Cross-language scope remains in the Summary because one shared numerical
  implementation across research languages is part of the paper's research
  need, not incidental implementation detail.
- Software and web references without DOIs were not given invented
  identifiers.
- The arXiv permanent identifier was not made a JOSS eligibility condition.
  Waiting for it remains the author's chosen publication sequence.
- Panel C is presented as two conditional implementation scenarios, not as a
  probability interval or a claim that the evaluated sample sizes identify an
  optimum.

## Open gates

- Publish an exact v2 release containing the reviewed software, manuscript,
  tests, and reproduction evidence, then create and verify its immutable
  archive.
- Build and visually inspect the official Open Journals PDF from the exact
  reviewed commit.
- Obtain the author's final source-by-source confirmation and comprehensive
  JOSS AI-policy attestation.
- Obtain attributable non-author research use, community engagement, or
  collaborative input under issue #471. Automated review activity does not
  satisfy this gate.
- Complete the author's arXiv-first sequence when the submitted preprint
  receives its permanent identifier.
- Only then perform the authenticated JOSS submission and record its external
  state from authoritative evidence.

Scores in the individual reports describe review snapshots taken at different
points in the revision. They are diagnostic, not comparable acceptance scores,
and were not raised artificially after changes. The contract report, source
hashes, hosted PDF, and external evidence are the controlling records.
