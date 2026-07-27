# Track Specification: Polyglot ABI And Binding Parity

## Overview

Expose common maturity-tiered VOI contracts through Rust, Python, R, Julia,
and Mojo.

## Requirements

1. Publish a supported Rust facade and a versioned typed C ABI for
   specifications, arrays, designs, results, diagnostics, errors,
   serialization, capabilities, and resource ownership.
2. Keep Python ergonomic but Rust-backed for stable numerics.
3. Make R direct-ABI with typed/data-frame/Arrow interfaces and executable
   vignettes; keep Python bridging optional.
4. Package Julia with Artifacts/JLL and Tables/Arrow integration.
5. Package Mojo C-ABI wrappers with owned buffers and structured errors; label
   CLI/JSON as fallback until direct Rust interop qualifies.
6. Generate capability surfaces from the canonical registry.
7. Serialize the canonical Decision Problem and estimator-assurance envelopes
   consistently in Arrow, JSON, the C ABI, and language-native types.
8. Fail conformance when code, packages, capability manifests, or
   documentation advertise an unsupported method or maturity.
9. Deliver GitHub [#579](https://github.com/edithatogo/voiage/issues/579),
   record ID `industry-decision-contract-binding-parity`: preserve the complete
   industry Decision Problem across Rust, the C ABI, Python, R, Julia, and
   Mojo. Portable fields include alternatives and policies; uncertainty and
   dependence; information actions and source portfolios; outcomes, utility,
   risk measures and constraints; populations, segments and perspectives;
   costs, time, implementation and flexibility; provenance, units, missingness,
   identifiers, privacy and rights; estimator assurance; and decision/audit
   results.
10. Generate language-native builders and validators from the canonical schema
    and keep BI, warehouse, notebook, decision-service, and workflow adapters
    outside the numerical core. Round trips must preserve unknown additive
    fields, declared unsupported capabilities, categorical identifiers and
    units without silently coercing them.
11. For residual method issues #593--#600, expose only the reviewed canonical
    disposition from the supported-frontier and method-census tracks.
    Quantitative methods accepted for implementation require Rust-authoritative
    typed results and conformance through Rust, Python, R, Julia, and Mojo;
    schema or qualitative components may be contract-only when scientifically
    appropriate. Planned records must remain unavailable in capability
    discovery.
12. Shared fixtures cover implementation/perfection decompositions,
    EVIU/VSS, utility-equivalent prices, event/density integrals, belief-state
    policies, signed agent/social ledgers, static/dynamic subgroup value, and
    outcome-conditional VSI distributions. Each binding preserves policies,
    units, signs, thresholds, decompositions, diagnostics, and unsupported
    states.

## Compatibility and failure policy

ABI v1 evolves additively, negotiates version/capability, and defines ownership.
Panics, allocator mismatch, use-after-free, GC lifetime failures, symbol/header
drift, and silent fallback block release.

## Acceptance criteria

Clean installed packages call every advertised method and pass golden and
randomized differential fixtures on supported platforms. Issue #579 passes
cross-language Decision Problem, result, diagnostics and assurance round trips,
including pathological and unsupported cases. Unsupported methods fail
explicitly. Issues #593--#600 have matching installed, adapter, contract-only,
unsupported, or upstream-blocked dispositions across all capability surfaces.
