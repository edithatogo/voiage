# Track Specification: pyOpenSci Submission Staging

## Overview

Prepare a current, reviewable pyOpenSci submission packet without opening an
external issue, completing the pre-review survey, or representing the package
as submitted, under review, or accepted.

## Requirements

1. Pin the authoritative pyOpenSci submission template to the exact upstream
   commit inspected during staging and record its SHA-256 digest.
2. Compare the public `v2.0.0` and `v2.1.0` candidates against release,
   documentation, package, manuscript, and independent-validation evidence.
3. Select a candidate only when the repository evidence is internally
   consistent; otherwise retain an explicit maintainer decision gate.
4. Prepare a complete local submission draft with factual answers, links, and
   unchecked human attestations. The artifact must state that it is unposted.
5. Machine-check template provenance, candidate identity, required fields,
   human-only checkboxes, and the absence of submission or acceptance claims.
6. Rerun the repository-owned submission-readiness, package, documentation,
   security, and distribution-identity gates against the frozen candidate.

## Acceptance criteria

- The current template source, upstream commit, and content digest are recorded.
- Candidate selection or its unresolved decision is evidence-backed.
- The local draft contains no unresolved ordinary fields; human attestations,
  survey completion, authenticated posting, review, and acceptance remain
  visibly unperformed.
- Focused validators and the full project assurance required by `AGENTS.md`
  pass before the delivery pull request is considered merge-ready.
- No external pyOpenSci or JOSS issue, form, message, badge, DOI, or acceptance
  claim is created by this track.

## External gates

- A maintainer must personally confirm the Code of Conduct, maintenance,
  survey, submission-version, and authenticated-posting fields.
- Opening a pyOpenSci review issue requires a separate explicit instruction.
- pyOpenSci editorial decisions, reviewer findings, and acceptance remain
  external outcomes.
- JOSS referral remains separately authorized and cannot begin before the
  selected pyOpenSci route reaches the required external state.
- Genuine non-author engagement tracked by issue #471 cannot be supplied by an
  agent, automated account, or same-author repository.

## Out of scope

- Contacting potential validators, editors, reviewers, pyOpenSci, or JOSS.
- Completing the pyOpenSci pre-review survey.
- Publishing a release, archive, badge, DOI, or manuscript.
- Rewriting release-bound JOSS evidence merely to remove candidate drift.

## Authoritative inputs

- `AGENTS.md`
- `docs/release/pyopensci-readiness.md`
- `docs/release/joss-submission-readiness.md`
- `specs/submission-readiness/pyopensci-evidence.json`
- `https://github.com/pyOpenSci/software-submission/blob/a1f31b8aab21128faee96ee548d256d5cffc3ba9/.github/ISSUE_TEMPLATE/submit-software-for-review.md`
- `https://www.pyopensci.org/software-peer-review/how-to/author-guide.html`
- `https://www.pyopensci.org/software-peer-review/about/package-scope.html`
