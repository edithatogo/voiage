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

## Compatibility and failure policy

ABI v1 evolves additively, negotiates version/capability, and defines ownership.
Panics, allocator mismatch, use-after-free, GC lifetime failures, symbol/header
drift, and silent fallback block release.

## Acceptance criteria

Clean installed packages call every advertised method and pass golden and
randomized differential fixtures on supported platforms. Unsupported methods
fail explicitly.

