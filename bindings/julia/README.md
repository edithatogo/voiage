# Voiage.jl

Julia binding for the stable EVPI surface of the voiage Rust core.

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
That PR builds the signed v2.0.0 Rust source for 64-bit glibc and musl Linux,
macOS, and Windows. JLL registration by the BinaryBuilder automation is the
first external gate.

After the generated `voiage_ffi_jll` exists in General, the package will depend
on that JLL and use its `libvoiage_ffi` product by default. The environment
variable above remains a development override. A clean-depot installation must
then pass on every supported platform before the source package is submitted
with:

```text
@JuliaRegistrator register subdir=bindings/julia
```

The resulting General registry merge is the second external gate. The package
is not described as registered or independently installable until both gates
have evidence.

## First workflow

```julia
using Voiage

net_benefits = [10.0 1.0; 2.0 8.0]
evpi_value = evpi(net_benefits)

@show evpi_value
```

This example returns `3.0` for the simple two-strategy matrix above.

## Release and scope

The release workflow verifies that `Project.toml` matches the release tag,
builds the FFI library, and runs `Pkg.test()`. TagBot is configured for the
`bindings/julia` subpackage and will create collision-free `julia-v*` tags
after Registrator accepts a version. The binding intentionally exposes only
the stable EVPI contract currently available through the shared Rust ABI.
