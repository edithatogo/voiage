# Stable Rust and validation providers for foss 2024a

This two-generation follow-on adds Rust 1.96.0, Maturin 1.13.1, setuptools-rust 1.12.0,
Pydantic 2.13.4 with exactly pydantic-core 2.46.4, and JSON Schema 4.26.0.
It extends the separate 2024a foundation without changing its recipe bytes. Source, Cargo-lock, backend-import, and Rust-bootstrap evidence is byte-identical to the reviewed 2023a source qualification because the selected upstream inputs are identical; the EasyBuild recipes and robot resolution are generation-specific.
These eight recipes do not complete the Voiage dependency graph.

The actual EasyBuild 5.4 dependency dry run passes for Pydantic and JSON Schema
with Python 3.12.14 and one stable Rust compiler. All 23 selected Python source
archives, the Rust source archive and 551 unique Cargo archives were downloaded
and hashed. The three Cargo lockfiles resolve with vendored sources and offline
mode. That metadata check used host Cargo 1.98, not a built Rust 1.96 compiler.
No native compiler, Python extension or EasyBuild installation was built here.

Use both repository overlays before the pinned EasyBuild catalogue:

```sh
eb packaging/easybuild-2024a-rust-overlay/2024a/pydantic-2.13.4-GCCcore-13.3.0.eb \
  packaging/easybuild-2024a-rust-overlay/2024a/jsonschema-4.26.0-GCCcore-13.3.0.eb \
  --robot=packaging/easybuild-2024a-rust-overlay/2024a:packaging/easybuild-overlay/2024a:PINNED_EASYCONFIGS \
  --dry-run
```

Replace `PINNED_EASYCONFIGS` with the easyconfigs directory at catalogue commit
`58e8b5a48767cbed1bf5669675d9638580d7259f`. A dry run proves dependency resolution,
not source compilation, installed module behavior or numerical correctness.

## Build tools and bootstrap boundaries

Maturin requires setuptools at least 77; typing-inspection and JSON Schema
specifications require hatchling at least 1.27. The separate build-support
module supplies setuptools 84, hatchling 1.29 and packaging 26.3 with their
source-built helpers. Consumers list these tools only as build dependencies.
The EasyBuild module generator excludes build-only edges from runtime modules.
The older foundation hatchling remains a prerequisite for building foundation
support, not a runtime dependency of the new packages.

Eleven backend distributions built from sdists in a private Python prefix.
A real EnvironmentModules load/unload test selected the new setuptools and
hatchling, then restored versions 70 and 1.24.2. This used a test module and
host Python, not an installed EasyBuild module. Its initial pip, flit-core,
setuptools-scm, wheel and old backend tools came from wheels in an isolated
environment; the receipt records that bootstrap exception.

Maturin's bootstrap cannot install an undeclared compiler:
`MATURIN_NO_INSTALL_RUST=1` is set for its installation. CargoPythonPackage
uses the declared Rust compiler, a private Cargo home, vendored crate checksums
and offline Cargo operation. Rust, Maturin and setuptools-rust are build-only
consumer dependencies; none is added to the Pydantic or JSON Schema runtime.
Polars' separately required dated nightly compiler is not substituted here.

The verified Rust source requires prebuilt Rust 1.95 bootstrap components dated
16 April 2026. Their checksums are recorded from the verified source, but those
binary archives were not downloaded or executed here. The Rust recipe configures compiler
and LLVM source builds with vendoring enabled and compiler/LLVM downloads disabled;
bootstrap components remain an explicit exception. A fully offline Rust build
has not been demonstrated. The retained sysroot patch required only a refresh
of unchanged surrounding context; its added and removed lines are identical.
The refreshed patch passes a zero-fuzz dry run against the actual source.

## Remaining evidence

Native Linux builds must confirm bootstrap acquisition, compiler behavior,
linked libraries, generated runtime modules and source-built extension imports.
The system OpenSSL wrapper still requires vendor security provenance. Native
Arrow/PyArrow, Polars, the remaining scientific consumers and the full Voiage
root remain separate work. Nothing here establishes upstream acceptance,
complete catalogue security qualification or completion of issue #1025.
