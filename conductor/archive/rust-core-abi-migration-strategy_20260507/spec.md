# Track Specification: Rust Core ABI And Migration Strategy

## Overview
This track defines how `voiage` should move toward a Rust execution core
without breaking the current public API. It also decides whether a narrow ABI
layer is worth adopting and, if so, where that layer belongs.

The intended long-term architecture is:

- Rust as the canonical execution core
- Python as the primary façade
- R, Julia, TypeScript, Go, and .NET as thin adapters
- a narrow ABI only if a native bridge is clearly useful

## Functional Requirements
1. The track must evaluate ABI relevance with a clear yes/no recommendation and
   the reasons for it.
2. The track must define an API-preserving migration plan from the current
   implementation to the Rust-core target architecture.
3. The track must identify the minimum stable boundary that should remain
   unchanged across languages:
   - `ValueArray`
   - `ParameterSet`
   - `TrialDesign`
   - diagnostics and reporting envelopes
   - method result shapes
4. The track must document the adapter strategy per language:
   - Python
   - R
   - Julia
   - TypeScript
   - Go
   - .NET
5. The track must decide whether any future ABI surface should be:
   - C ABI
   - WASM / N-API for TypeScript
   - language-native FFI only
6. The track must define the compatibility and versioning rules needed so
   migration does not break existing user-facing APIs.

## Non-Functional Requirements
1. The track must prefer stable contracts over wide low-level ABI exposure.
2. The track must distinguish internal Rust modularization from external ABI
   publication.
3. The track must make a no-breaking-change path explicit for the existing
   language APIs.

## Acceptance Criteria
1. The repo has a written ABI recommendation.
2. The repo has a migration path that preserves the current public APIs.
3. The repo has an adapter policy for each language target.
4. The repo has compatibility and versioning rules for a Rust-core migration.

## Out of Scope
1. Implementing a new ABI layer.
2. Breaking the current Python or binding APIs.
3. Changing language bindings into a second primary source of truth.
