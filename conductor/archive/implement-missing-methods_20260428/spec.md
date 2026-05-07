# Track Specification: Replace Placeholders and Implement Missing Methods

## Overview
Some advertised capabilities are placeholders, degraded fallbacks, or incomplete implementations. This track replaces those gaps with real behavior and closes method-level mismatches between the repository's claims and its actual scientific functionality.

## Functional Requirements
1. Placeholder implementations must be replaced with real computations or explicitly downgraded in the public contract.
2. Missing or incomplete method families called for by the active roadmap and track plan must be implemented behind stable interfaces.
3. New method behavior must be validated numerically against known formulas, reference calculations, or published results where feasible.
4. Silent fallback behavior that hides missing functionality must be removed or surfaced via explicit warnings.

## Non-Functional Requirements
1. Implementations must remain type-safe, test-covered, and readable.
2. Method-level diagnostics must be strong enough to support future conformance fixtures and external bindings.
3. Added complexity should be justified by scientific value rather than speculative feature breadth.

## Acceptance Criteria
1. Planned method gaps are either implemented or explicitly re-scoped in the spec and docs.
2. Placeholder logic is no longer reachable on supported execution paths.
3. New and updated methods have targeted tests plus broader regression coverage where shared surfaces change.
4. Validation, linting, and type checks pass for the affected modules.

## Out of Scope
1. CLI parity and documentation polish beyond what is required to land method implementations safely.
2. Cross-language fixture generation.
