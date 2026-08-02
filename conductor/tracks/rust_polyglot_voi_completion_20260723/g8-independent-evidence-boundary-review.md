# G8 independent evidence and boundary review

## Scope and method

This review evaluates the Rust/polyglot parity contracts, installed-run packet,
registry/release closure plan and Conductor evidence chain. It is a repository
engineering/boundary review, not external scientific approval or registry
acceptance. The panel compared the artifacts with `specs/v1/binding-matrix.json`,
`specs/rust/migration_matrix.json`, `specs/v1/compatibility-policy.json`, the
Rust ABI tests and the documented release checklist.

## Findings

| Severity | Finding | Disposition |
|---|---|---|
| High | R and Julia installed parity is not demonstrated; the local run is blocked by an uninstalled R package and unavailable Julia FFI library. | Keep promotion blocked; rerun in clean installed environments. |
| High | Rust PyO3 test target requires a matching `libpython3.13.dylib` unavailable in the current environment. | Keep Rust core evidence separate from Python bridge evidence; rerun on a matching CI runner. |
| Medium | Several migration-matrix methods are fixture-backed or Python-only. | Preserve explicit maturity and do not advertise polyglot parity. |
| Medium | Registry, signing, publication and parent-issue closure are external/maintainer gates. | Use the state machine and receipt rules; no inferred completion. |
| Low | Fixture manifest currently enumerates stable kernel families rather than every experimental frontier method. | Extend only when a method contract and fixture are approved. |

## Decision

**Conditional pass for repository preparation; promotion blocked.** The schema,
fixture manifest, diagnostics/provenance requirements and fail-closed boundary
are adequate for implementation. No Critical finding was identified. Stable
promotion requires clean installed Rust/Python/R/Julia runs against the same
fixture hash, panel review of the resulting packet, maintainer approval, and
separate release/registry gates.

## Required follow-up

1. Build the matching Rust FFI/PyO3 artifacts and record the toolchain/runtime.
2. Install `voiageR` in a clean R 4.3 environment and run the shared fixtures.
3. Expose `libvoiage_ffi` to Julia 1.10/1.11 and run the shared fixtures.
4. Append output/diagnostic hashes, tolerance results and unsupported
   capability dispositions to the evidence packet.
5. Re-run this review after the packet is complete; retain external scientific,
   maintainer, signing and registry gates as separate decisions.
