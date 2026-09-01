# Specification: v2.2 Release and Venue Submissions

## Overview

Release the repository-hardened source as v2.2.0, verify its immutable public
artifacts, then pursue the maintainer-selected venue sequence: pyOpenSci first,
JOSS through the pyOpenSci partner fast-track after eligibility is established,
and rOpenSci for the standalone R package. This track begins only after the
pre-submission comprehensive-hardening programme completed and was archived.

The maintainer authorized this sequence on 2026-08-30 by directing the agent to
proceed with the recommended release and submissions. That authorization does
not allow the agent to invent personal attestations or venue outcomes.

## Authoritative inputs

- `AGENTS.md`, including the solo-maintainer merge and manuscript boundaries.
- `.github/workflows/release.yml` and the binding/crate release workflows.
- `voiage/versioning.py` and the synchronized Python, Rust, R, Julia, ABI, and
  documentation version contracts.
- `specs/submission-readiness/final-candidate-binding-20260829.json`.
- `specs/submission-readiness/pyopensci-submission-staging.json` and its pinned
  upstream template revision.
- `specs/submission-readiness/ropensci-evidence.json` and
  `specs/submission-readiness/ropensci-standards-mapping.md`.
- `paper/main.tex`, `paper/metadata.json`, and the JOSS readiness contracts.
- Current official pyOpenSci, JOSS, and rOpenSci author guidance, rechecked
  before each authenticated external action.

## Requirements

1. Use v2.2.0 as the next backward-compatible minor release. Synchronize every
   package and binding version without changing the v2.1 C ABI contract.
2. Update release notes, submission packets, manuscript metadata, and candidate
   bindings so they describe the exact v2.2.0 source rather than historical
   v2.1.0 or v1.0.0 artifacts. Discovery citation records continue describing
   the last verified public release until publication receipts justify updating
   their version, release date, and download URLs.
3. Validate the version candidate locally and through a protected pull request.
   Merge only after all required checks and review threads are terminal.
4. Create a signed annotated `v2.2.0` tag on the exact merged candidate.
5. Let the release workflow build and attest a private draft first. Review and
   record the exact wheel and sdist SHA-256 values before invoking publication.
6. Verify the public GitHub release, PyPI/TestPyPI artifacts, provenance, SBOM,
   clean installation, and any triggered Rust or binding publication outcomes.
7. Refresh the pyOpenSci submission body against the live official template,
   obtain or record every required maintainer-only attestation, and create the
   authenticated submission only when no required answer is being inferred.
8. Preserve pyOpenSci review, acceptance, and partner-referral state as external
   evidence. Initiate the JOSS fast-track only when the current partner route
   permits it and all JOSS author declarations are explicitly evidenced.
9. Refresh and initiate rOpenSci review for the standalone `voiageR` package
   after the pyOpenSci-first ordering has been honored, without claiming CRAN,
   r-universe, review, acceptance, or indexing outcomes prematurely.
10. Keep repository readiness, release publication, submission creation, review,
    acceptance, DOI assignment, and indexing as distinct states.
11. Preserve the resolution of the pyOpenSci one-review-per-contact requirement
    against existing submissions. The later maintainer decision and withdrawal
    receipt verify that requests #271 and #272 were closed as not planned;
    preserve that evidence without treating closure as editorial approval.
    Avoid concurrent review at different venues unless the editors explicitly
    allow it, and obtain human review of venue communications.
12. Complete all currently known repository-owned release and venue-packet
    repairs before the first venue submission. Preparing JOSS and rOpenSci
    materials does not start their reviews or bypass later eligibility gates.

## Acceptance criteria

- **AC-01:** Track artifacts, registry state, and append-only evidence validate.
- **AC-02:** A reviewed, tree-equal PR places synchronized v2.2.0 metadata and
  refreshed release/submission records on `main` with full local and hosted gates.
- **AC-03:** The signed tag and public v2.2.0 release are bound to exact artifacts,
  hashes, provenance, SBOM, and successful clean-install evidence.
- **AC-04:** A pyOpenSci submission exists with an authoritative external URL and
  no fabricated maintainer attestation.
- **AC-05:** JOSS is submitted through the partner fast-track only after the
  pyOpenSci eligibility gate and required author declarations are satisfied.
- **AC-06:** rOpenSci submission evidence identifies the exact standalone R
  package candidate and preserves external review/acceptance states.
- **AC-07:** The track reports external waiting states honestly and is archived
  only when all acceptance criteria have authoritative evidence.

## Non-functional constraints

- Keep all release tags signed and immutable.
- Never expose credentials, environment values, or private draft payloads.
- Preserve the v2.1 ABI baseline and supported Python/R/Julia/Rust contracts.
- Use pull requests as the auditable source-change boundary.
- Do not select arXiv category or license, certify authorship/funding/conflicts,
  or answer personal venue declarations on the maintainer's behalf.

## External gates

- Maintainer-only venue attestations and survey answers.
- Private pre-review survey, human-written submission body, and authenticated
  pyOpenSci submission. Requests #271 and #272 are verified withdrawn and
  closed; no editorial scope or capacity approval is inferred from closure.
- GitHub environment protections and trusted-publishing authorization.
- pyOpenSci scope screening, review, acceptance, and partner referral.
- JOSS editorial screening, review, acceptance, and DOI publication.
- rOpenSci scope screening, review, acceptance, and downstream registry outcomes.

## Out of scope

- New analytical features or dependency promotions after the frozen candidate.
- Claiming acceptance, indexing, citation, or impact before external evidence.
- arXiv submission unless separately authorized with category and license choices.
