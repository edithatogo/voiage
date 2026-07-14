# Track Specification: R Release Submission

## Overview

This track closes the R release submission path. It keeps the package release
story aligned with the documented GitHub Release source-archive flow while
stating clearly that CRAN and r-universe remain external registry targets.

## Functional Requirements

1. Keep the R package version and `r-v*` tag pattern aligned.
2. Keep the GitHub Release source-archive workflow intact.
3. Keep the CRAN and r-universe story explicit as external submission or
   indexing steps.
4. Keep the package docs/manual flow aligned with the shipped source tree.

## Non-Functional Requirements

1. Preserve the reticulate-bridge relationship to the Python façade unless the
   roadmap explicitly changes it.
2. Avoid runtime behavior changes in this track.
3. Keep the release path reproducible and transparent.

## Acceptance Criteria

1. The R release docs and checklist agree on the GitHub Release path.
2. The docs clearly state that CRAN submission remains external/manual.
3. The docs clearly state that r-universe indexing remains external/manual.

## Out of Scope

1. Changing R runtime semantics.
2. Claiming CRAN or r-universe submission is complete without external proof.
3. Reworking the package docs/manual pipeline.
