# Round 8 JOSS criteria reconciliation

Date: 26 July 2026

This repository-owned review compares the current manuscript and readiness
contract with the official JOSS submission, paper, review, checklist, and
editorial guidance. It records an internal assessment, not a JOSS decision.

## Material finding

The official guidance distinguishes two related questions. The submission and
editorial guides treat demonstrated research use as a hard pre-review gate and
describe non-author engagement as a strong positive signal rather than a hard
gate. The detailed review criteria still classify a single-author project with
no community engagement, external use, or collaborative input as not
acceptable. The previous readiness record collapsed these statements into one
pre-review gate.

## Changes incorporated

- Added demonstrated research use as its own pending pre-review gate.
- Retained issue #471 as the route to attributable engagement and as the
  author's selected prerequisite because the detailed review criterion remains
  a material acceptance risk.
- Recorded that the paper's synthetic demonstration and a fallback-only
  same-author adapter do not establish completed research-workflow use.
- Updated the manuscript to state that neither completed research-workflow use
  nor non-author engagement has yet been documented.
- Updated the article contract, fail-closed submission validator, Conductor
  track, roadmap, task list, tests, and readiness handoff with the same
  distinction.

## Assurance boundary

SourceRight reconciles all 19 citation occurrences with no citation issue and
retains six bounded missing-DOI warnings for software or web references.
Authentext reports no selected-pattern finding. Open Journals workflow run
[`30164270490`](https://github.com/edithatogo/voiage/actions/runs/30164270490)
built the exact source at commit
`c0f2fc623ffea36f600a6b82c6597d949ab32bc0`; the hosted article contract passed
at 1,629 words, and all six PDF pages passed visual review. Textstat remains
review-only evidence rather than an acceptance threshold.
