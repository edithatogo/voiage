# Round-10 synthesis and action ledger

## Scope

Round 10 combined six independent editorial reports with dedicated
accessibility and sentence-level reviews. Every role read the complete initial
article before assessing sections, claims, citations, and sentences. Scores
ranged from 808 to 927 of 1,000 and therefore did not meet the requested
threshold. The low scores were driven principally by unbound assurance and
human/external evidence, not a disagreement about the worked calculations.

## Consensus findings

| Priority | Finding | Disposition |
| --- | --- | --- |
| P0 | Current-source citation/prose assurance, official JOSS build, and visual review were stale. | Re-run after wording is final; do not inherit old evidence. |
| P0 | The manuscript claimed a final AI-review attestation while the assurance record marked it pending. | Keep the record pending until explicit author confirmation; do not represent it as resolved. |
| P0 | Citation keys reconciled mechanically, but all source-by-source reviews remain queued. | Human-only gate; preserve the explicit boundary. |
| P0 | Completed research use and non-author engagement were not evidenced. | External JOSS eligibility gates; do not rewrite as adoption. |
| P1 | Definitions and design language were too technical for applied readers. | Revised with plain-language explanations and bounded examples. |
| P1 | The build-versus-contribute rationale was too categorical. | Revised to state the realistic dependency/compatibility trade-off. |
| P1 | The health example and figure needed clearer interpretation. | Revised prose, caption, and redundant figure encoding. |
| P2 | The availability section over-emphasised internal identifiers. | Shortened for readers while retaining repository evidence. |

## Evidence-based changes

The revision changes no calculated value, model parameter, source citation, or
scope boundary. It changes only how supported facts are expressed, with the
following effects:

- the study model remains a declared normal--normal computational illustration,
  not a proposed trial estimate;
- the uncertainty intervals remain finite-simulation intervals, not model
  uncertainty;
- Rust remains the shared EVPI core, with Python, R, and Julia interfaces of
  deliberately unequal current scope;
- the same-author exchange remains a verification case and not independent use.

## Exit criteria for Round 11

1. Contract, SourceRight, Authentext, health-example regeneration, and focused
   tests pass for the exact source digest.
2. A hosted official JOSS PDF is built from the committed revision and visually
   reviewed page by page.
3. Each panel role re-reviews the revised source and the synthesis records its
   score and any residual deduction.
4. The panel does not award credit for pending human citation/AI confirmation
   or external research-use/community evidence.

## Current boundary

Round 10 improves the manuscript but cannot produce the requested >995/1000
panel result while P0 human and external gates remain pending. The next loop
will nevertheless measure the revised manuscript quality separately from those
gates and retain the objective's fail-closed threshold.
