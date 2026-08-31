# pyOpenSci software-review submission — UNPOSTED LOCAL DRAFT

Submission performed: **No**

This file is a repository-owned staging artifact based on the official
pyOpenSci template pinned in
`specs/submission-readiness/pyopensci-submission-template.json`. It is not a
GitHub issue, survey response, submission, review, or acceptance record.
The maintainer has authorized the selected submission route. Personal declarations were confirmed on 31 August 2026 and prior requests
were withdrawn. Survey completion and human-written submission text
remain gates before authenticated posting. This AI-assisted staging draft is
not a substitute for the maintainer's own review communication.

Submitting Author: @edithatogo
All current maintainers: @edithatogo
Package Name: voiage
One-Line Description of Package: Value-of-information analysis for decisions
about uncertainty and further research.
Repository Link: https://github.com/edithatogo/voiage
Version submitted: 2.2.0 (confirmed by maintainer; submission not performed)
EiC: Assigned by pyOpenSci after posting
Editor: Assigned by pyOpenSci after posting
Reviewer 1: Assigned by pyOpenSci after posting
Reviewer 2: Assigned by pyOpenSci after posting
Archive: A DOI-bearing reviewed-version archive has not been created
JOSS DOI: Not assigned; JOSS referral has not started
Version accepted: Not applicable before external acceptance
Date accepted: Not applicable before external acceptance

## Code of Conduct & Commitment to Maintain Package

- [x] I agree to abide by [pyOpenSci's Code of Conduct][PyOpenSciCodeOfConduct]
  during the review process and in future interactions in spaces supported by
  pyOpenSci should it be accepted. Confirmed by the maintainer on 31 August 2026.
- [x] I have read and will commit to package maintenance after the review as
  described by the [pyOpenSci policies][Commitment]. Confirmed by the maintainer on 31 August 2026;
  the repository records the approved best-effort maintenance policy and
  two-year post-acceptance commitment in `GOVERNANCE.md` and `SUPPORT.md`.

## Description

`voiage` is an open-source Value of Information (VOI) library for researchers
and analysts comparing decisions under uncertainty and assessing whether
additional evidence may be worth collecting. It provides Python APIs and a
command-line interface for EVPI, EVPPI, method-specific EVSI, ENBS, decision
context records, diagnostics, plotting, reporting, and explicitly marked
experimental frontier methods. Selected stable numerical policy is Rust-backed.
Python and Julia use the main core; the narrower R package bundles a separate
offline Rust kernel for EVPI and ENBS, checked against shared numerical fixtures.

## Associated Publication (Optional)

Publication Title: No peer-reviewed package publication exists. A JOSS-format
manuscript is prepared locally but has not been submitted or accepted.
Publication DOI: Not assigned
Journal/Venue: No associated peer-reviewed publication

## Scope

- [ ] Data retrieval
- [ ] Data extraction
- [x] Data processing/munging
- [ ] Data deposition
- [x] Data validation and testing
- [ ] Data visualization
- [x] Workflow automation
- [ ] Citation management and bibliometrics
- [ ] Scientific software wrappers
- [ ] Database interoperability

`voiage` validates labelled simulation and decision-context inputs, transforms
them into VOI results with retained units and provenance, and automates
reproducible research-prioritisation workflows. Its primary audience is
researchers, health economists, statisticians, and decision analysts using
probabilistic models to decide whether uncertainty could change a choice and
whether further evidence may justify its cost.

Established R tools such as `voi`, `BCEA`, and `dampack` provide broader
health-economic method and reporting surfaces, while SAVI provides web-based
EVPI and regression EVPPI. Python packages cover narrower signal-value or
domain-specific workflows. `voiage` differs by combining a Python VOI workflow,
versioned decision records, strict stable-versus-experimental maturity
boundaries, and selected Rust-backed calculations used by narrower R
and Julia bindings. It does not claim to replace those tools or to have complete
cross-language method parity.

No pyOpenSci pre-submission inquiry has been made. The scope categories above
are an evidence-backed draft classification and remain subject to pyOpenSci's
editorial scope decision.

## Domain Specific

- [ ] Geospatial
- [ ] Education

## Community Partnerships

- [ ] Astropy
- [ ] Pangeo

No domain-community affiliation is claimed in this draft.

## Technical checks

- [x] The package does not violate the Terms of Service of a service it
  interacts with; the core package is local-first and remote ingestion is
  fail-closed behind explicit source policy.
- [x] The package uses the OSI-approved Apache-2.0 licence.
- [x] The README and `CONTRIBUTING.md` document released and development
  installation.
- [x] The online documentation contains examples and API references for the
  supported public surface.
- [x] The documentation contains tutorials for essential EVPI, EVPPI, EVSI,
  and ENBS workflows.
- [x] The repository contains Python, Rust, binding, property, packaging,
  integration, and contract tests.
- [x] GitHub Actions runs CI, coverage, documentation, package-identity,
  security, supply-chain, and compatibility checks.

## Publication Options

- [x] Do you wish to automatically submit to the [Journal of Open Source
  Software][JournalOfOpenSourceSoftware]? The maintainer explicitly selected
  pyOpenSci first, then the eligible JOSS partner route. This selection does
  not claim pyOpenSci acceptance, JOSS eligibility or a completed submission.

### JOSS checks

- [x] The package has an obvious research application: prioritising research
  and further evidence for decisions under modelled uncertainty.
- [x] The package is not a minor utility; it contains stable analytical,
  schema, provenance, CLI, packaging, and cross-language infrastructure plus a
  separately governed experimental frontier.
- [x] A JOSS-format `paper.md` exists.
- [ ] A DOI-bearing long-term archive for the reviewed version exists. This
  remains an external acceptance-stage gate.

This draft and the corrected JOSS paper describe public `v2.2.0`. Its signed
tag, distribution hashes, provenance and immutable publication receipt are
bound in the candidate manifest. The historical developer-use record remains
v2.0.0; an automated two-environment replay verified v2.2.0 without adding human
use or independent adoption. JOSS remains blocked on partner eligibility,
current human review, the author-selected engagement prerequisite,
and reviewed-version archive evidence. No JOSS submission is claimed.

## Current policy: development history, AI and communication

This supplemental section follows the current policy page, which has disclosure
requirements beyond the unchanged pinned issue template.

- [x] I confirm sustained human-led development and the design decisions behind
  the public history. The repository has public history since July 2025;
  automated commit counts do not establish human design or review.
- [x] I have personally reviewed and understood all current submitted code and
  documentation, including AI-assisted changes. Confirmed by the maintainer on 31 August 2026.
- [x] I will write the review communication personally, using at most limited
  translation or grammar assistance as allowed by pyOpenSci. JOSS permits AI
  assistance in author/editor/reviewer conversations only for translation.
- [x] Generative AI tools were used in development and maintenance.
- [x] I have verified the AI scope and scale disclosure for this exact packet.

Recorded assistance includes Codex and Jules, with earlier Antigravity/Gemini
and Copilot entries in `AI_CONTRIBUTIONS.md`. Agentic assistance covers
substantial code and test generation, refactoring, CI, release evidence,
documentation and manuscript editing. The new research-handoff script and its
tests were agent-authored. A repository-wide percentage and complete historical
model inventory were not retained and are not inferred. The maintainer confirmed this disclosure on 31 August 2026. Later changes
require review against the packet actually submitted.
Project policy: `AI_USAGE.md`. Earlier sign-off records do not certify later work.

The maintainer prioritized voiage and authorized withdrawal of requests #271
and #272. Both were closed as not planned on 31 August 2026; no other open
submission by this account was observed. This does not claim editorial approval.
The decision and withdrawal receipts are retained in the active Conductor track.
The maintainer now directs journal submission before arXiv, following their
reported case-specific advice. No journal or arXiv submission is claimed.
Personal confirmation of the communication commitment does not make this
AI-assisted draft human-authored; the actual human-written body remains pending.

## Are you OK with Reviewers Submitting Issues and/or pull requests to your Repo Directly?

- [x] Reviewers may open
  requested changes as repository issues or pull requests and link them from
  the pyOpenSci review.

## Author-guide confirmations

- [x] I have read the pyOpenSci author guide. Confirmed by the maintainer on 31 August 2026.
- [x] I expect to maintain this package for at least two years and can help
  find a replacement maintainer if needed. Confirmed by the maintainer on 31 August 2026; the
  repository policy and the maintainer decision receipt record this commitment.

## Please fill out our survey

- [ ] Last but not least please fill out our pre-review survey. The survey has
  not been completed by this staging work.

## Editor and Review Templates

The current pyOpenSci editor and peer-review templates remain authoritative.
This repository draft does not populate editor-owned or reviewer-owned fields.

[Commitment]: https://www.pyopensci.org/software-peer-review/our-process/policies.html#after-acceptance-package-ownership-and-maintenance
[JournalOfOpenSourceSoftware]: https://joss.theoj.org/
[PyOpenSciCodeOfConduct]: https://www.pyopensci.org/handbook/CODE_OF_CONDUCT.html
