# Track Specification: SOTA Packaging Review Readiness

## Overview
This track maps `voiage` against the higher-bar review and packaging
expectations used by scientific Python and R communities, and then turns that
fit analysis into a concrete readiness plan. The goal is not to claim every
community as a perfect match. The goal is to state which communities are a
direct fit, which are a stretch fit, and which are not recommended unless the
repo changes materially.

The review targets to assess are:

- pyOpenSci
- rOpenSci
- JOSS
- scikit-learn-contrib
- NumFOCUS project/community alignment

## Functional Requirements
1. The track must produce a fit matrix for each target community, with each
   target classified as one of:
   - direct fit
   - stretch fit
   - not recommended
2. The track must document the repository changes needed for any direct-fit or
   stretch-fit target, including packaging, documentation, testing, metadata,
   citation, support, and maintenance expectations.
3. The track must distinguish between:
   - review-oriented communities such as pyOpenSci, rOpenSci, and JOSS
   - ecosystem or project-hosting communities such as NumFOCUS
   - compatibility-specific communities such as scikit-learn-contrib
4. The track must identify the minimum evidence needed for a realistic
   submission attempt, including:
   - installation clarity
   - citation metadata
   - community-facing documentation
   - tests and CI
   - release/versioning discipline
   - maintainership and support expectations
5. The track must record the recommended order of engagement so the repo can
   pursue the highest-fit communities first instead of treating all targets as
   equivalent.

## Non-Functional Requirements
1. The track must avoid overstating fit. If a target is only plausible after a
   larger API or ecosystem change, the track must say so explicitly.
2. The analysis must be specific enough that later implementation tracks can
   convert it into repo changes without reopening the fit question.
3. The output must remain usable even if the community target is a no-go;
   "not recommended" is a valid outcome.

## Acceptance Criteria
1. A fit matrix exists for all listed target communities.
2. The repo changes needed for each direct-fit or stretch-fit target are
   enumerated.
3. A recommended submission or engagement order is written down.
4. The deliverable makes clear which communities are true review targets and
   which are better treated as ecosystem or visibility channels.

## Out of Scope
1. Filing any external community submission.
2. Changing package code solely to satisfy an external target before the fit
   analysis is complete.
3. Treating scikit-learn-contrib as a target unless a genuine
   scikit-learn-compatible surface is proposed.
