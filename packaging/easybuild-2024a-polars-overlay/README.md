# Polars 1.42.1 provider candidate for foss 2024a

This overlay resolves the split Polars Python package and native
`polars-runtime-32` provider for the Voiage foss 2024a graph. EasyBuild 5.4.0
successfully parsed and resolved the recipes against catalogue commit
`58e8b5a48767cbed1bf5669675d9638580d7259f`.

Polars 1.42.1 is a small Python package that requires the exact
`polars-runtime-32==1.42.1` distribution. The runtime source pins
`nightly-2026-04-01`; this overlay therefore gives that compiler a distinct
`Rust-nightly` module. The maturin provider remains built with stable Rust
1.96.0 from the adjacent stable-Rust overlay. The runtime recipe binds `RUSTC`
and `CARGO` to the dated-nightly module and cannot silently select stable Rust.

The two verified PyPI source archives, the runtime Cargo lock, all 530 registry
crate archives, and three commit-pinned Git source archives are declared as
checksum-bound recipe inputs. A strict patch redirects the Git dependencies to
the extracted local archives, and Cargo runs offline. The
dated compiler source identity reuses the exact source audit already bound by
the Spack overlay; this does not reuse any native build result.

Run the dependency-resolution check with EasyBuild 5.4.0, Environment Modules
and the pinned catalogue:

```console
eb packaging/easybuild-2024a-polars-overlay/2024a/polars-1.42.1-GCCcore-13.3.0.eb \
  --dry-run \
  --robot=packaging/easybuild-2024a-polars-overlay/2024a:packaging/easybuild-2024a-rust-overlay/2024a:packaging/easybuild-overlay/2024a:PINNED_EASYCONFIGS
```

The retained robot output is dependency-resolution evidence only. No Rust
compiler, crate, Python extension or EasyBuild module was built or loaded.
Native installation, full Voiage graph qualification, upstream submission,
x86-64 behavior and production-cluster performance remain false or
unverified.
