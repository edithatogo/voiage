# Voiage.jl

Julia package scaffold for the voiage core API contract.

## Setup

The package currently expects a locally built `voiage-ffi` library. From the
repository root:

```bash
cargo build --manifest-path rust/Cargo.toml --release --locked --package voiage-ffi
VOIAGE_FFI_LIBRARY="$PWD/rust/target/release/libvoiage_ffi.dylib" \
  julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Use `libvoiage_ffi.so` on Linux and `voiage_ffi.dll` on Windows. A
Julia-native binary-artifact package is still required before the package can
be installed from General without a separate Rust build.

## First workflow

```julia
using Voiage

net_benefits = [10.0 1.0; 2.0 8.0]
evpi_value = evpi(net_benefits)

@show evpi_value
```

This example returns `3.0` for the simple two-strategy matrix above.

## Release and caveats

The release workflow verifies that `Project.toml` matches the release tag,
builds the FFI library, runs `Pkg.test()`, and attaches a source archive to the
GitHub release. General registration remains blocked until platform-specific
libraries are delivered through Julia's artifact system. After that packaging
work, registry updates should use Registrator and TagBot. The walkthrough is a
thin adapter story because the Rust core remains the semantic authority.
