# Plan for Cross-Language Conformance Fixtures

## [ ] Phase 1: Fixture Catalog and Layout

1. Create a versioned fixture directory structure under `specs/core-api/fixtures/v1/`.
2. Add `README.md` files that describe the normative and illustrative fixture collections.
3. Ensure the directory layout is stable and easy for future language implementations to mirror.

## [ ] Phase 2: Canonical Outputs and Tolerance Envelopes

1. Seed the normative fixtures for the key core API outputs.
2. Add illustrative fixtures that demonstrate expected shape without enforcing exact values.
3. Define tolerance rules for numeric comparisons where exact equality is not appropriate.

## [ ] Phase 3: Runner Contract

1. Implement a shared validation contract for fixture runners.
2. Document the expected file naming, manifest structure, and example conventions.
3. Add tests that lock the contract so future implementations remain compatible.
