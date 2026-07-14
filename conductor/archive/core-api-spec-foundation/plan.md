# Track Implementation Plan: Core API Spec Foundation

## Phase 1: Create the Spec Artifact Scaffold [checkpoint: ]

- [x] Task: Create the project-level artifact folder for the cross-language API contract under `specs/core-api/`.
- [x] Task: Create a primary foundation document that explains the purpose of the contract, its intended audience, and how later schema and fixture artifacts will relate to it.
- [x] Task: Create a compact decision-record document that captures the chosen contract format, target runtimes, and the rule that the written contract is authoritative over incidental Python behavior.

## Phase 2: Write the Normative v1 Foundation [checkpoint: ]

- [x] Task: Define the canonical vocabulary and conceptual model for the stable v1 contract.
  - [x] Record the net-benefit-first interpretation explicitly.
  - [x] Record study-design and research-decision objects as first-class concepts.
  - [x] Define which method families are stable in v1 versus deferred.
- [x] Task: Record the backend and interchange principles that later tracks must respect.
  - [x] xarray-labeled arrays as the Python in-memory reference model.
  - [x] NumPy as the reference execution baseline.
  - [x] JAX as optional acceleration, not public contract.
  - [x] Arrow/Parquet as cross-language interchange boundary.
  - [x] Polars as adapter-only tabular tooling, not canonical compute model.
- [x] Task: Convert the competitor feature scan into a capability-baseline section that states what `voiage` must match or exceed to be credible.

## Phase 3: Prepare the Downstream Handoff [checkpoint: ]

- [x] Task: Enumerate the exact outputs expected from the next three tracks:
  - [x] canonical schemas and examples
  - [x] numerics, diagnostics, and extension rules
  - [x] conformance fixtures and Python cleanup
- [x] Task: Add a small-model execution guide that spells out the decisions later tracks must not re-litigate implicitly.
- [x] Task: Validate that all new spec artifacts exist, are internally cross-linked, and contain no unresolved placeholders that would block the next track.

## Execution Notes

- Phases are intentionally narrow so `conductor-implement` and `conductor-review` can run autonomously with a smaller model.
- The output of this track is written contract material, not Python refactoring.
