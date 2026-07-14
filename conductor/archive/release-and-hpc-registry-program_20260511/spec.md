# Track Specification: Release And HPC Registry Program

## Overview

This umbrella track coordinates the remaining release work that is still
described as "ready to submit" rather than "live registry verified" in the
repo docs. It does not publish packages itself. Instead, it sequences the
language-specific release submission tracks and the HPC distribution contract
track so they can be executed without overlapping scope.

This program is the prerequisite for the HPC native acceleration roadmap. The
release-to-registry pathway should be completed and documented before the repo
moves into the next HPC requirements stage.

The program covers:

1. HPC distribution contract and registry handoff
2. Python release submission
3. R release submission
4. Julia release submission
5. TypeScript release submission
6. Go release submission
7. Rust release submission
8. .NET release submission

## Launch Order

The child tracks should be implemented in this order:

1. HPC distribution contract
2. Python release submission
3. R release submission
4. Julia release submission
5. TypeScript release submission
6. Go release submission
7. Rust release submission
8. .NET release submission

## Functional Requirements

1. Define the child-track boundaries so each release path has a single owner
   and a single documented submission story.
2. Keep the HPC contract work separate from the language release work.
3. Preserve the existing release automation and versioning policy while the
   remaining submission work is organized.
4. Record the launch order for the child tracks so implementation can proceed
   deterministically.
5. Update the Conductor registry so the program and its child tracks are easy
   to discover.

## Non-Functional Requirements

1. Avoid any runtime code changes in this umbrella track.
2. Keep the child-track scopes non-overlapping.
3. Prefer explicit registry and release documentation over implied status.

## Acceptance Criteria

1. The umbrella track exists with spec, plan, metadata, and index files.
2. The spec names all release and HPC child tracks.
3. The program order is clear enough for autonomous implementation.
4. The registry entry points to the umbrella track.

## Out of Scope

1. Publishing any package to a live registry from this track.
2. Implementing HPC kernels or changing runtime behavior.
3. Rewriting the release automation already documented in the repo.
