# Engineering acceptance reconciliation — 13 November 2026

Verification started from hosted main
`2161126cb0d4bb8c1a000eb623abab0c5dab2daf` (tree
`e4f49ced07c35b99501a02e6378200f82ffb53c0`). The local probes used CPython
3.15.1 and the explicitly selected Rust toolchains below. The only initial
tracked edit was the ownership handoff. Raw command logs and resource reports
are retained in the sibling `voiage-engineering-acceptance-runs` directory on
the external SSD. Historical evidence remains in the existing records.

## G02.4 — Rust feature and MSRV qualification

The current workspace inventories 23 feature lanes across seven crates,
including both `otel` and `otel-http` on diagnostics. With Cargo and rustc
1.85.0 explicitly selected, `python scripts/rust_feature_matrix.py . --verify`
passed every locked lane on macOS arm64. The matrix output SHA-256 is
`cdd447e21360159b392516a897d682126b34fe20fad14d1740ebebede47690bc`.
`scripts/verify_msrv_rejection.py` passed the feature-off and compatible
controls and rejected the backend requiring Rust 1.86. An initial invocation
found Homebrew Cargo 1.99.1 and correctly rejected it before testing.
[Hosted MSRV proof on PR #1273](https://github.com/edithatogo/voiage/actions/runs/42212645440/job/127874799579)
covers the workspace all-targets, all-features check and executable witness on
Linux. The Rust workspace, lockfile, MSRV workflow and witness have identical
tracked content at the merged PR and the tested main revision. This
qualification covers the declared matrix and creates no dependency promotion.

## G03.4 — Provider and binding compatibility

The existing provider, C signature and C linkage witnesses remain applicable to
their pinned source inputs. The [hosted binding matrix for PR #1273](https://github.com/edithatogo/voiage/actions/runs/42212645440)
passed the R installed smokes and Julia source-binding tests on Linux, macOS
and Windows. A fresh isolated CPython 3.15.1 environment installed the
published PyPI 3.1.0 wheel and returned EVPI 0.5 for the symmetric two-action
case. The verified R-universe 3.1.0 source archive was built from the tested
main revision and installed into an isolated R library using Rust 1.85.0.
R 4.3.0 returned the expected EVPI 0.5 and ENBS 9.5 at 1e-12 absolute
tolerance. The archive SHA-256 matched the registry:
`021c0135bfac8f2e31b5d40878fb4ebb25f05cdc1e58400ff6d10cc26940cebd`.
[PyPI release](https://pypi.org/project/voiage/3.1.0/).
[R-universe build](https://github.com/r-universe/edithatogo/actions/runs/43390814299).

Julia General has no Voiage entry in the observed registry tree. The upstream
[Yggdrasil PR #14292](https://github.com/JuliaPackaging/Yggdrasil/pull/14292)
is open and no generated Voiage JLL release was available. The existing
[Voiage issue #1023](https://github.com/edithatogo/voiage/issues/1023)
owns the generated JLL, clean-depot validation and General registration.
The earlier local Git installation and hosted source tests remain valid for
their respective routes. Registry acceptance stays open.

## G04.4 — Coverage aggregation, cancellation and replay

On the same main SHA, [run 43406478582](https://github.com/edithatogo/voiage/actions/runs/43406478582)
passed both shards and applied the existing policy to combined native data.
Aggregate branch coverage was 94.8770816812054%, above the 90% floor. The
downloaded policy receipt SHA-256 is
`94ae2d0a7acc0ec078b19626e9c9892e36adf376a3c81a513e97640ee5104d59`.
In the dedicated [cancellation run 43406545152](https://github.com/edithatogo/voiage/actions/runs/43406545152),
both running shards were cancelled. The always-run aggregate saw
`MATRIX_RESULT=cancelled` and failed its first check with exit 1. No
aggregate or shard artifact was published. A prior [ordinary failure run](https://github.com/edithatogo/voiage/actions/runs/43385388173)
also rejected the unsuccessful matrix. The
[dependency preview](https://github.com/edithatogo/voiage/actions/runs/43407457128)
passed the nextest archive and replay steps on this SHA.

The same 30 focused identity and coverage controls passed twice under the
resource guard. The pytest-cache-cold and pytest-cache-warm runs recorded
21.09 s and 14.05 s of pytest time, with the same lock, coverage config,
interpreter, source and test selection. Guarded wall times were 84.946 s and
16.979 s. Other workloads were active, so the two measurements describe the
observed runs and support no CI savings estimate. Local controls exercise
mismatched source, lock and interpreter, failed subprocesses, missing and
tampered reports, and failed or cancelled shard status.

## G05.4 — Live numerical register

On the current main SHA, 127 tests passed for the locked `voiage-numerics`
and `voiage-test-support` packages with Cargo, rustc and rustdoc 1.99.1.
Both the compiled-test inventory and malformed/missing-target tests passed.
The first invocation mixed Rustup rustc with Homebrew rustdoc and failed
during doctest setup. Selecting rustdoc from the same Rustup toolchain passed.
The SSD run log SHA-256 is
`9a5538857bcbd558aa4e9ece792e2d34320a395163859fb063b193746795a70d`.
[PR #1273](https://github.com/edithatogo/voiage/pull/1273) retains extracted
archive evidence and successful hosted Rust tests. The fixture, numerical
sources, Rust lock and binding workflow match the tested main revision.
The standalone archive carries its analytic fixtures. The repository register
continues to validate live targets in the checkout.
