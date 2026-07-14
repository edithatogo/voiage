# Track Specification: .NET Release Submission

## Overview

This track closes the .NET release submission path. It keeps the NuGet publish
flow aligned with the documented `dotnet-v*` tag pattern and GitHub Release
artifact generation.

## Functional Requirements

1. Keep the .NET package version and `dotnet-v*` tag pattern aligned.
2. Keep the build, test, and pack gates intact.
3. Keep the NuGet publish step explicit about its API-key requirement.
4. Keep the GitHub Release source-archive and nupkg artifact path documented.

## Non-Functional Requirements

1. Preserve the thin adapter story for the .NET binding.
2. Avoid runtime behavior changes in this track.
3. Keep the release path reproducible and explicit.

## Acceptance Criteria

1. The .NET release docs and checklist agree on the NuGet path.
2. The docs clearly state that publication happens on `dotnet-v*` tags when
   credentials are available.
3. The release artifacts remain attached to GitHub Releases.

## Out of Scope

1. Changing .NET runtime semantics.
2. Replacing the existing NuGet publish flow.
3. Claiming registry submission succeeded without verification.
