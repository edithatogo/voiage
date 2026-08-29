# pyOpenSci software-review submission — UNPOSTED LOCAL DRAFT

Submission performed: **No**

This file is a repository-owned staging artifact based on the official
pyOpenSci template pinned in
`specs/submission-readiness/pyopensci-submission-template.json`. It is not a
GitHub issue, survey response, submission, review, or acceptance record.
Unchecked human attestations must be completed personally, and authenticated
posting requires a separate maintainer instruction.

Submitting Author: @edithatogo  
All current maintainers: @edithatogo  
Package Name: voiage  
One-Line Description of Package: Value-of-information analysis for decisions
about uncertainty and further research.  
Repository Link: https://github.com/edithatogo/voiage  
Version submitted: 2.1.0 (recommended; maintainer confirmation pending)  
EiC: Assigned by pyOpenSci after posting  
Editor: Assigned by pyOpenSci after posting  
Reviewer 1: Assigned by pyOpenSci after posting  
Reviewer 2: Assigned by pyOpenSci after posting  
Archive: A DOI-bearing reviewed-version archive has not been created  
JOSS DOI: Not assigned; JOSS referral has not started  
Version accepted: Not applicable before external acceptance  
Date accepted: Not applicable before external acceptance

## Code of Conduct & Commitment to Maintain Package

- [ ] I agree to abide by [pyOpenSci's Code of Conduct][PyOpenSciCodeOfConduct]
  during the review process and in future interactions in spaces supported by
  pyOpenSci should it be accepted. Human attestation pending.
- [ ] I have read and will commit to package maintenance after the review as
  described by the [pyOpenSci policies][Commitment]. Human attestation pending;
  the repository records the approved best-effort maintenance policy and
  two-year post-acceptance commitment in `GOVERNANCE.md` and `SUPPORT.md`.

## Description

`voiage` is an open-source Value of Information (VOI) library for researchers
and analysts comparing decisions under uncertainty and assessing whether
additional evidence may be worth collecting. It provides Python APIs and a
command-line interface for EVPI, EVPPI, method-specific EVSI, ENBS, decision
context records, diagnostics, plotting, reporting, and explicitly marked
experimental frontier methods. Selected stable numerical policy is shared
through a Rust core, with narrower source bindings for R and Julia.

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
boundaries, and selected shared Rust-backed calculations used by narrower R
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

- [ ] Do you wish to automatically submit to the [Journal of Open Source
  Software][JournalOfOpenSourceSoftware]? Maintainer selection pending. The
  chosen project route is pyOpenSci first and a separately authorized JOSS
  partner handoff after pyOpenSci acceptance and JOSS scope confirmation.

### JOSS checks

- [x] The package has an obvious research application: prioritising research
  and further evidence for decisions under modelled uncertainty.
- [x] The package is not a minor utility; it contains stable analytical,
  schema, provenance, CLI, packaging, and cross-language infrastructure plus a
  separately governed experimental frontier.
- [x] A JOSS-format `paper.md` exists.
- [ ] A DOI-bearing long-term archive for the reviewed version exists. This
  remains an external acceptance-stage gate.

The JOSS paper and demonstrated developer-use record currently describe
`v2.0.0`, whereas this draft recommends the current `v2.1.0` release for
pyOpenSci. The later JOSS handoff remains blocked until that release boundary,
issue #471 human-engagement evidence, the selected arXiv sequence, and the
DOI-bearing archive are resolved. No JOSS submission is claimed.

## Are you OK with Reviewers Submitting Issues and/or pull requests to your Repo Directly?

- [ ] Maintainer confirmation pending. If confirmed, reviewers may open
  requested changes as repository issues or pull requests and link them from
  the pyOpenSci review.

## Author-guide confirmations

- [ ] I have read the pyOpenSci author guide. Human attestation pending.
- [ ] I expect to maintain this package for at least two years and can help
  find a replacement maintainer if needed. Human attestation pending; the
  repository policy is already documented but does not check this form box on
  the maintainer's behalf.

## Please fill out our survey

- [ ] Last but not least please fill out our pre-review survey. The survey has
  not been completed by this staging work.

## Editor and Review Templates

The current pyOpenSci editor and peer-review templates remain authoritative.
This repository draft does not populate editor-owned or reviewer-owned fields.

[Commitment]: https://www.pyopensci.org/software-peer-review/our-process/policies.html#after-acceptance-package-ownership-and-maintenance
[JournalOfOpenSourceSoftware]: https://joss.theoj.org/
[PyOpenSciCodeOfConduct]: https://www.pyopensci.org/handbook/CODE_OF_CONDUCT.html
