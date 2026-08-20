# Installed parity assessment — 2026-08-21

This packet is bound to exact main commit `d9e01333bc47b3be07c075a54656ddfb5bd94e15` and the immutable shared-fixture manifest hash recorded in the adjacent JSON artifact. It updates the earlier 2026-08-03 disposition; it does not promote any method.

## Local evidence

* Rust workspace and release FFI build pass, including contract, ABI, numerical, property, differential, fixture and Python-bridge tests.
* Python repository-owned C18 contract/evidence suites pass under Python 3.14.
* A clean temporary R installation passes native FFI and numerical-reference tests. Four tests that exercise `testthat::with_mocked_bindings()` fail after installation because they require a `pkgload` development context; these are recorded as harness-context failures, not numerical or ABI failures.
* Julia 1.12.7 was attempted with the release FFI path, but package precompilation remained locked by a stale worker. The run is blocked and supplies no parity evidence.
* Mojo remains an unsupported external toolchain boundary.

## Disposition

The repository-owned result is **partial installed parity; retain all methods experimental**. The panel recommendation is conditional experimental acceptance only. Stable promotion, a broad cross-language parity claim, external registry acceptance, publication, and issue closure remain separate gates. A clean hosted Julia run and a clarified installed-R harness are the next executable evidence steps.

## Panel and maintainer boundary

The separated advisory panel reports remain authoritative for limitations and dissent. The maintainer/scientist may approve repository-owned experimental status, but no accountable decision is inferred for external registries, journals, package indexes, or publication services.
