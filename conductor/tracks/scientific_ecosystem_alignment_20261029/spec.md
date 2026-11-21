# voiage delivery contract

Approved by user on 29 October 2026. This bounded track implements the
[coordinated programme](https://github.com/edithatogo/repository-standards/tree/7874f05132447c77f681fbca7ad5394066564aaf/conductor/tracks/scientific_ecosystem_alignment_20261029); it does not replace existing native
acceptance or other active track owners.

- Reconcile accepted provider and capacity packets through the existing integrator.
- Review each Rust/LLVM source bootstrap against an explicit packaging requirement.
- Qualify Innovate/MARS adapters and tobac/VOP installed numerical contracts; retain full HPC gates.

Use frontier-first qualified candidates and explicit retained-version exceptions.
Separate core, binding, backend, build, test-oracle and tooling dependencies.
Acceptance requires exact-source numerical and consumer tests, measured builds,
agent correctness/security review, fixes and repository integration. Planning or
static source checks are not native acceptance. Existing promises are retained
until a versioned migration passes; no paid execution or public release implied.

## Serde usage and substitution policy — 18 November 2026

The Rust workspace already isolates serialization as `voiage-serialization`, whose `Cargo.toml` includes `serde` derive and `serde_json`. Its validated domain/numerical contracts remain in `voiage-domain`/`voiage-numerics`, with Python/R/FFI adapters as separate consumers. This is a sound use of Serde; replacing Polars/Arrow, scientific algorithms, numerical types or native bindings with it would be a regression.

- [ ] Within scientific ecosystem alignment, compare actual versioned numerical results and typed exchange DTOs across Rust, Python and R with frozen fixtures, all special/finite float values, vectors/shapes and diagnostic codes. Preserve missingness and nonfinite-value handling explicitly.
- [ ] Inspect the separate `serde_json_canonicalizer` use in `voiage-python` against the existing `voiage-serialization` canonical contract; only consolidate a repeated implementation if **byte-level** canonical/hash parity, RFC 8785 semantics, precision and existing consumer compatibility are proven. Do not assume `serde_json::to_string` is canonical.
- [ ] Reuse the serialization crate for new stable wire contracts; keep extra/extension fields dynamic if downstream versions require forward compatibility. Do not spread Serde derives into pure numeric kernels purely for convenience.
- [ ] Any implementation is separately owned by the existing dependency/simulation/consumer tracks with exact-head locked/native tests, independent scientific oracles and installed-consumer receipts; this note does not alter released capability or evidence status.
