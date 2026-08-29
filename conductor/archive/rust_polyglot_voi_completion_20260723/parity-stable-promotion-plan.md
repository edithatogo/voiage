# Rust/R/Julia parity and stable-promotion plan

## Objective

Make parity and stable promotion auditable without treating a local build,
source inspection, or registry listing as proof. Rust remains the numerical
authority; Python, R and Julia are consumer façades; Mojo remains an external
upstream boundary.

## Ordered gates

1. **Contract freeze:** bind each promoted method to a versioned API/schema,
   numerical tolerances, diagnostics, error semantics, seed policy and fixture
   manifest. No contract means no parity claim.
2. **Rust reference:** run the canonical Rust workspace against every fixture;
   record toolchain, target, command, output hashes and benchmark baseline.
3. **Python consumer:** run the installed wheel, not only the checkout, against
   the same fixtures and compare canonical outputs within the contract
   tolerance.
4. **R consumer:** install `r-package/voiageR` in a clean R 4.3 environment,
   load the shared C ABI, run the same fixtures and record native symbol,
   platform and output hashes. CRAN/r-universe review remains external.
5. **Julia consumer:** instantiate `bindings/julia` in clean Julia 1.10/1.11
   environments, run the same C ABI fixtures and record platform/library
   metadata. Julia General/JLL review remains external.
6. **Negative and compatibility checks:** verify unsupported methods fail with
   typed diagnostics, ABI/layout symbols match, and old v1 fixtures remain
   byte- or tolerance-compatible.
7. **Promotion review:** a panel reviews the complete packet. Only then may a
   method move from `fixture_backed` to `verified` or from experimental to
   stable. Scientific validity, maintainer approval, signing, registry and
   publication remain separate gates.

## Required evidence packet

For each language/method record: immutable commit/tag, fixture-manifest SHA,
toolchain/runtime, OS/architecture, install command, test command, output and
diagnostic hashes, tolerance result, ABI symbol/layout result, unsupported
capabilities, benchmark result, reviewer and timestamp. Never record secrets.

## Current disposition

The repository currently has verified parity for the methods explicitly marked
`verified` in `specs/rust/migration_matrix.json`. Fixture-backed and Python-only
methods remain non-stable. R and Julia installed parity is not implied by their
source packages; Mojo has no local executable contract. Stable promotion is
blocked until this packet is complete and reviewed.
