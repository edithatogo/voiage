# Track Specification: TypeScript Release Submission

## Overview

This track closes the TypeScript release submission path. It keeps the npm
publish flow aligned with the documented tag pattern and release artifact
generation.

## Functional Requirements

1. Keep the TypeScript package version and `typescript-v*` tag pattern aligned.
2. Keep the npm publish workflow with provenance intact.
3. Keep the GitHub Release asset path documented and reproducible.
4. Keep the package submission path explicit about its in-repo automation.

## Non-Functional Requirements

1. Preserve the thin adapter story for the TypeScript binding.
2. Avoid runtime behavior changes in this track.
3. Keep the release path reproducible and explicit.

## Acceptance Criteria

1. The TypeScript release docs and checklist agree on the npm path.
2. The release assets still attach to GitHub Releases.
3. The docs clearly state that the npm publish flow is automated here.

## Out of Scope

1. Changing TypeScript runtime semantics.
2. Replacing the existing npm publishing automation.
3. Claiming registry submission succeeded without verification.
