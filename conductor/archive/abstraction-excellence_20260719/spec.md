# Specification: Domain Abstraction Excellence

## Overview

Introduce a canonical Pydantic v2 contract boundary for machine-readable
concerns, parameters, numerical execution, kernels, backends and results while
retaining xarray as the numerical in-memory representation and preserving all
supported VOIAGE APIs.

## Requirements

- Frozen, strict and extra-forbidding concern/evidence and analysis models.
- Generic calculation-kernel protocol and explicit backend capabilities.
- Result envelopes containing method identity, maturity, diagnostics,
  provenance, run context, numerical policy and interchange identity.
- Compatibility adapters for existing dataclasses, arrays and method results.
- Deterministic JSON Schema and VOP-pinned Arrow/Parquet/IPC conformance.
- Opt-in experimental capabilities with governed fallback disclosure.
- Privacy-safe GitHub issue/Project synchronization payloads.

## Acceptance criteria

- New public contracts contain no untyped dictionaries or forwarding kwargs.
- Existing method calls continue to pass compatibility and numerical-parity tests.
- Backend selection is capability based and fail-closed when requirements cannot be met.
- Cross-repository fixtures validate in fresh processes and across PyArrow/Polars.
- CI, typing, security, profiling, mutation, documentation and package gates pass.

## Out of scope

- Immediate replacement of all established xarray/dataclass internals.
- Stable claims for experimental hardware or methods without promotion evidence.
- Publication of credentials or local/private evidence.

