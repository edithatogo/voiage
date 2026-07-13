# Track Specification: HPC Registry Readiness

## Overview

After language release automation is in place, the next step is to make
HPC-registry exposure explicit and reviewable. This track defines what counts
as "ready for HPC registry workflows" for Spack, EasyBuild, HPSF, and E4S.

This track does not claim that registry ingestion has completed. It creates the
contract, evidence package, and clear external handoff points required before the
project can be treated as registry-deployable in production HPC environments.

The outcome is expected to keep the boundary clear:

1. In-repo packaging/release workflows remain unchanged.
2. HPC packaging/curation paths are documented with explicit external/manual
   assumptions.
3. External actions are called out in `docs/release/binding-submission-checklist.md`
   and `docs/developer_guide/hpc_distribution_contract.rst`.

## Functional Requirements

1. Capture an explicit matrix for Spack, EasyBuild, HPSF, and E4S status with
   clear "ready"/"not ready"/"blocked" indicators.
2. Define the exact repository artifacts required for each HPC path
   (source tarballs, checksums, reproducible build flags, dependency manifests).
3. Document the external dependency and approval gate for each target.
4. Add a concrete handoff package in docs so reviewers can verify what has and has
   not been submitted for each registry ecosystem.
5. Update `roadmap.md` and `conductor/tracks.md` so this work appears in the
   strategic sequence and is not hidden behind language-specific completion notes.

## Non-Functional Requirements

1. Keep language bindings stable and avoid coupling runtime changes to registry work.
2. Prefer deterministic, reproducible build and manifest references over ad-hoc
   instructions.
3. Keep all claims conservative: no false assertion of live publication.

## Acceptance Criteria

1. The HPC registry contract is explicit in docs and linked from the
   roadmap and track registry.
2. `binding-submission-checklist.md` includes an explicit HPC registries row for
   Spack, EasyBuild, HPSF, and E4S.
3. A reviewed handoff note documents which steps are already done in-repo and
   which still need registry maintainer actions.
4. This track can be handed to a maintainer with a one-page submission
   readiness matrix.

## Out of Scope

1. Publishing final Spack/EasyBuild/HPSF/E4S registrations from this track.
2. Introducing speculative accelerator hardware work outside the existing
   abstraction contract.
3. Changing the core VOI APIs or benchmark contract.
