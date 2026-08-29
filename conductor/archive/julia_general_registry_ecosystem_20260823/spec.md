# Track Specification: Julia General Registry & Ecosystem Integration

## Scope & Purpose
Establish first-class Julia distribution for `Voiage.jl` via the official Julia General Registry and connect with the JuliaHealth scientific community.

## Requirements
1. **Package Integrity**: `Project.toml` adheres to Julia Pkg standards (pinned UUID, compatibility bounds, test targets).
2. **JLL Resolution**: Support dynamic resolution of `voiage_ffi_jll` with clean fallback to `VOIAGE_FFI_LIBRARY`.
3. **Quality & Formatting**: Zero Aqua defects (`Aqua.test_all(Voiage)`).
4. **Domain Documentation**: Worked example using Julia dataframes and health-economic decision matrices.
5. **Staged Registration**: Defer Registrator execution until maintainer review.
