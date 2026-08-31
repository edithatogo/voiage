# Voiage.jl

Julia binding for the stable EVPI and ENBS surfaces of the voiage Rust core.

## Setup

The package currently expects a locally built `voiage-ffi` library. From the
repository root:

```bash
cargo build --manifest-path rust/Cargo.toml --release --locked --package voiage-ffi
VOIAGE_FFI_LIBRARY="$PWD/rust/target/release/libvoiage_ffi.dylib" \
  julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Use `libvoiage_ffi.so` on Linux and `voiage_ffi.dll` on Windows.

## Binary artifact and registry status

The repository-owned BinaryBuilder recipe is
`packaging/yggdrasil/V/voiage_ffi/build_tarballs.jl`. It is submitted upstream
as [Yggdrasil PR #14292](https://github.com/JuliaPackaging/Yggdrasil/pull/14292).
That historical PR targets the signed v2.1.0 Rust source. Its recorded platform
matrix is retained with the historical build evidence. The merged v2.2.0
candidate is `packaging/yggdrasil/candidates/v2.2.0/build_tarballs.jl`. It has
not been built or submitted upstream; the v2.1.0 results do not validate it.
JLL registration by the BinaryBuilder automation is the first external gate.

After the generated `voiage_ffi_jll` exists in General, the package will depend
on that JLL and use its `libvoiage_ffi` product by default. The environment
variable above remains a development override. A clean-depot installation must
then pass on every supported platform before the source package is submitted
with:

```text
@JuliaRegistrator register subdir=bindings/julia
```

The metadata, subdirectory command, and TagBot configuration form the local
Registrator staging contract. Staging does not authorize or execute that
command: retain maintainer review after the JLL and clean-depot prerequisites.

The resulting General registry merge is the second external gate. The package
is not described as registered or independently installable until both gates
have evidence.

## First workflow

```julia
using Voiage

net_benefits = [10.0 1.0; 2.0 8.0]
evpi_value = evpi(net_benefits)
enbs_value = enbs(2.5, 1.0)

@show evpi_value
@show enbs_value
```

This example returns `3.0` for the simple two-strategy matrix and `1.5` for
the signed net research value.

## Release and scope

The release workflow verifies that `Project.toml` matches the release tag,
builds the FFI library, and runs `Pkg.test()`. TagBot is configured for the
`bindings/julia` subpackage and will create collision-free `julia-v*` tags
after Registrator accepts a version. The binding intentionally exposes only
the stable EVPI and ENBS contracts currently available through the shared Rust
ABI.

The experimental expected-utility information-pricing family and its VoC
presentation are not exposed by this Julia package because the stable C ABI
has no corresponding symbol. The family capability record therefore marks
Julia as `unsupported`; this is an explicit boundary, not a parity claim.
