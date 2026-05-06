# Track Specification: CLI Completion and Integration Testing

## Overview
The repository needs a complete, reliable command-line surface and a trustworthy integration-test layer so end-to-end workflows can be exercised outside direct Python imports. This track closes CLI parity gaps and restores realistic integration coverage.

## Functional Requirements
1. Every supported public method family selected for CLI exposure must have a documented command path.
2. Integration tests must cover representative end-to-end CLI and library workflows using realistic fixture data.
3. CLI errors must be actionable and consistent with the public API contract.
4. Example commands and fixtures must reflect the actual supported inputs and outputs.

## Non-Functional Requirements
1. The CLI must remain scriptable and non-interactive by default.
2. Integration tests should be stable enough for CI and not depend on ad hoc local state.
3. The command surface should avoid premature exposure of unstable experimental methods.

## Acceptance Criteria
1. CLI coverage matches the intended public surface for stable methods.
2. Integration tests that were previously disabled or stale are either repaired or explicitly removed with rationale.
3. The relevant CLI help text and examples align with implemented behavior.
4. Required lint, type, and test checks pass.

## Out of Scope
1. Reworking the core scientific algorithms beyond what is necessary for CLI compatibility.
2. Cross-language bindings.
