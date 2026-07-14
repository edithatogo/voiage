# Track Specification: Go Release Submission

## Overview

This track closes the Go release submission path. It keeps the semver tag
release flow aligned with the documented GitHub Release artifacts and the Go
module proxy indexing path.

## Functional Requirements

1. Keep the Go module version and `bindings/go/v*` tag pattern aligned.
2. Keep the module test and vet gates intact.
3. Keep the GitHub Release source-archive path documented.
4. Keep the module proxy indexing story explicit as the publication path.

## Non-Functional Requirements

1. Preserve the thin adapter story for the Go binding.
2. Avoid runtime behavior changes in this track.
3. Keep the release path reproducible and explicit.

## Acceptance Criteria

1. The Go release docs and checklist agree on the module-proxy path.
2. The docs clearly state that publication is driven by tagged modules and
   downstream indexing.
3. The release artifacts remain attached to GitHub Releases.

## Out of Scope

1. Changing Go runtime semantics.
2. Replacing the existing module release flow.
3. Claiming module proxy indexing has completed without external verification.
