# Julia binary candidate for v2.2.0

Issue #555 still requires repository work as well as upstream acceptance. The
open Yggdrasil PR #14292 and its 15-platform Buildkite run target v2.1.0. The
published v2.2.0 release is commit
`7af563c8cb373057d30662650b3f332f39e05b83`. Its FFI adds the R-compatible
`voiage_v1_enbs_r` wrapper; the existing Julia EVPI and ENBS entrypoints and ABI
1.1 remain compatible. The older candidate is internally consistent but is
not a v2.2.0 build receipt.

The prepared replacement recipe is
`packaging/yggdrasil/candidates/v2.2.0/build_tarballs.jl`. It changes only the
version and pinned Git source. The original
`packaging/yggdrasil/V/voiage_ffi/build_tarballs.jl` remains the exact historical
recipe bound to the existing platform contract. Tests reject extra changes
to compiler, product, target filters or build flags in this bounded refresh.

The dated JSON receipt binds both recipes and keeps all new hosted-build,
product, runtime and registry fields empty. Local preparation and Julia syntax
validation do not establish BinaryBuilder execution. No upstream PR was
updated and no registry application was sent.

After protected repository delivery and separate upstream authorization, use
the prepared bytes for `V/voiage_ffi/build_tarballs.jl` in the existing upstream
PR. Re-run the platform matrix and verify actual products and supported host
execution at that new head; do not reuse the old 15 passing jobs as new evidence.
Only after the JLL is generated and accepted should the Julia package consume
its real UUID and version with bounded compatibility, verify clean-depot
installation without Rust, and proceed to General registration. No UUID has
been guessed and none is added by this change.

Sources: [published release](https://github.com/edithatogo/voiage/releases/tag/v2.2.0),
[current upstream candidate](https://github.com/JuliaPackaging/Yggdrasil/pull/14292),
and [historical build](https://buildkite.com/julialang/yggdrasil/builds/31972).
