# Track Specification: Julia Release Submission

## Overview

This track closes the Julia release submission path. It keeps the package's
tag-driven release story aligned with GitHub Releases and TagBot while stating
clearly that Julia General registry approval remains external.

## Functional Requirements

1. Keep the Julia package version and `julia-v*` tag pattern aligned.
2. Keep the GitHub Release source-archive workflow intact.
3. Keep TagBot synchronization configured and documented.
4. Keep the Julia General registry step explicit as external approval /
   registration.

## Non-Functional Requirements

1. Preserve the thin adapter story for the Julia binding.
2. Avoid runtime behavior changes in this track.
3. Keep the release path reproducible and explicit.

## Acceptance Criteria

1. The Julia release docs and checklist agree on the tag and release flow.
2. The docs clearly state that Julia General registry approval remains
   external/manual.
3. The GitHub Release archive path remains documented.

## Out of Scope

1. Changing Julia runtime semantics.
2. Claiming General registry approval is complete without external proof.
3. Reworking the TagBot setup beyond documenting the submission path.
