# Track Specification: Cross-Language Conformance Fixtures

## Overview
This track creates the canonical fixtures and tolerance envelopes that future Python, R, and Julia implementations must pass. It turns the API contract into executable evidence and gives the project a durable basis for regression testing and future bindings.

## Functional Requirements
1. The track must create deterministic fixture inputs and canonical expected outputs for the stable v1 method families.
2. Fixtures must cover the agreed core scope:
   - EVPI
   - EVPPI
   - EVSI
   - ENBS
   - population VOI and sample-size optimization if included in the stable contract
3. Fixture artifacts must use interchange formats suitable for multiple languages, with Arrow/Parquet used where tabular interchange is beneficial.
4. The track must define the fixture manifest, naming conventions, and tolerance envelopes that bindings must enforce.
5. Fixtures must record the provenance and deterministic execution mode required to reproduce the canonical outputs.
6. The fixture runner contract must be designed for downstream package CI in Python, R, Julia, TypeScript, Go, Rust, and .NET 11 so every published binding can validate against the same canonical cases before registry release.

## Non-Functional Requirements
1. Fixtures must stay small enough for CI and for lower-capability models to regenerate or inspect safely.
2. The catalog must be explicit about which fixtures are normative versus illustrative.
3. The structure must support incremental growth without rewriting all existing bindings.

## Acceptance Criteria
1. A fixture manifest exists and points to all normative v1 fixtures.
2. Each stable core method family has at least one deterministic conformance case.
3. Tolerance envelopes are documented and machine-usable.
4. The artifacts can be consumed by future Python, R, and Julia runners without Python-specific assumptions.
5. The runner guide documents how future binding CI jobs should invoke conformance checks before package publishing to PyPI/Conda-forge, CRAN/r-universe, Julia General, npm, the Go module ecosystem, crates.io, and NuGet.

## Out of Scope
1. Full binding implementations.
2. Broad benchmark or parity studies against every external package.
