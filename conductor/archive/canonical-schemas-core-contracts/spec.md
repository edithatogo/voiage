# Track Specification: Canonical Schemas and Core Method Contracts

## Overview
This track turns the written API foundation into machine-readable contract artifacts. It defines the canonical schemas and semantic contracts for the core decision-analysis objects and method results that Python, R, and Julia implementations must share.

## Functional Requirements
1. The track must create canonical JSON Schemas for the stable v1 entities and payloads.
2. The scope must include:
   - decision problem and intervention identity
   - parameter draws / uncertainty inputs
   - net benefit result payloads
   - EVPI, EVPPI, EVSI, and ENBS result contracts
   - population VOI and sample-size optimization contracts where these are included in v1
3. Each schema must have a paired semantic description covering units, invariants, missing-data rules, and interpretation of outputs.
4. Contracts must distinguish stable result fields from optional capability-dependent fields.
5. Experimental and deferred methods must not be smuggled into stable schemas; they should be reserved through explicit extension hooks only where needed.

## Non-Functional Requirements
1. Schemas must be versioned and suitable for use in automated fixture validation.
2. Contracts must avoid Python-specific types or assumptions in the public layer.
3. Example payloads must be small, deterministic, and human-reviewable.

## Acceptance Criteria
1. Stable v1 schemas exist for the agreed core objects and method outputs.
2. Every schema has at least one canonical example document.
3. The contracts are precise enough to support cross-language fixture generation without hidden Python assumptions.
4. The output clearly separates required, optional, and extension fields.

## Out of Scope
1. Tolerance policies and reproducibility semantics beyond what is strictly required to reference them.
2. Binding-specific adapters.
