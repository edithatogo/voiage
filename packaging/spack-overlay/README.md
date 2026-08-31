# Local HPC catalogue overlay

These are source-verified recipe candidates, not installed HPC packages.
The namespace is `voiage_hpc_overlay`; the fallback catalogue is
`spack/spack-packages` at
`d4f7c711a6a42f1c4d551c8fd10fce9a11340a81`.
The selected voiage recipe is the reviewed 2.2.0 candidate from PR #1058.

The overlay preserves the copied Spack copyright notices, licence files,
patches and historical recipe versions. `manifest.json` binds every recipe and the retained architecture patch.
`source-audit.json` and `transitive-source-audit.json` record the downloaded,
hash-verified Python source archives and their build requirements.
`rust-source-audit.json` verifies the downloaded dated compiler source against
its official checksum and binds its embedded stage0 to the dated beta manifest.
Bootstrap binary digests are official metadata; those binaries were not
downloaded or installed. `rust-backend-source-audit.json` verifies the two
unchanged Python build-tool sources.
`arrow-source-audit.json` binds the Apache Arrow C++ archive to the official
SHA-512 sidecar. Its release signature has not been verified.

## Dependency changes

- Typing extensions 4.16.0 and Click 8.5.0 use Flit Core 3.11 through 3.x.
- Typer 0.27.2 uses PDM and the new annotated-doc dependency. Historical
  Click caps do not apply to this version's source metadata.
- Pydantic 2.13.4 requires pydantic-core 2.46.4. The native core needs Rust
  1.88 or later and Maturin 1.10 through 1.x. The old exact core dependency
  is restricted to the historical Pydantic range.
- Polars 1.42.1 separates its Python package from polars-runtime-32 at the
  same version. The runtime source pins `nightly-2026-04-01`. The recipe
  retains that requirement; it does not substitute stable or unpinned Rust.
- PyArrow 25 uses a custom scikit-build-core backend and requires Cython
  3.1 or later, LibCST 1.8.6 or later, and CMake 3.25 or later. Its Arrow
  C++ dependency is version matched and requests CSV, dataset, filesystem and
  Parquet support for voiage ingestion and interchange. Arrow 25 requires C++20. The copied
  C++ recipe still needs native build review against the new release.
- LibCST adds a native Rust extension and conditional YAML dependencies.
  Its Python 3.13 path requires the separate pyyaml-ft package; unsupported
  catalogue entries must be added rather than silently removing markers.

## Isolated solver audit

Run from the repository root. Choose a new evidence directory and retain it.
This configuration does not change the installed system catalogue.

```sh
mkdir -p .conductor/local/hpc-overlay-run
export SPACK_USER_CONFIG_PATH="$PWD/.conductor/local/hpc-overlay-run/config"
export SPACK_USER_CACHE_PATH="$PWD/.conductor/local/hpc-overlay-run/cache"
spack repo add --scope user "$PWD/packaging/spack-overlay"
spack config --scope user add repos:builtin:commit:d4f7c711a6a42f1c4d551c8fd10fce9a11340a81
spack config --scope user add "bootstrap:root:$PWD/.conductor/local/hpc-overlay-run/bootstrap"
cp packaging/spack-overlay/concretizer.yaml "$SPACK_USER_CONFIG_PATH/concretizer.yaml"
spack spec py-typer@0.27.2
spack spec py-pydantic@2.13.4
spack spec py-voiage@2.2.0
```

Spack may bootstrap its solver in that isolated bootstrap directory; reading
an existing download cache does not establish that dependencies were built.
Individual dependency graphs can resolve while the full voiage graph fails.
Retain each exit status and complete solver output separately.

No native dependency build, voiage installation, Linux foss stack, module-load
transcript or upstream submission is established by these recipes. Resolve
the complete graph, review the compiler and backend changes, then build and
test on a verified Linux host before advancing issue #1025.

## Recorded result

Spack 1.2.2 concretized Typer 0.27.2, Pydantic 2.13.4 with its new native
core, and PyArrow 25 with Arrow 25 separately. Each command returned zero;
`solver-receipt.json` records their log paths, hashes and the exact recipe
manifest. The complete solver outputs in `solver-logs/` contain no local
filesystem paths and allow independent inspection of the dependency graphs.

The dated-toolchain follow-up resolves the complete voiage graph on the recorded
macOS arm64 host. `solver-logs/voiage.json` retains the full concrete DAG.
It contains stable Rust 1.96.0 for Pydantic Core and a distinct exact
`nightly-2026-04-01` for Polars. Polars explicitly binds `RUSTC` and `CARGO`
to its direct dependency's prefix. The dated compiler uses beta stage0 from
5 March 2026 and retains the nightly release channel during source builds.

`concretizer.yaml` keeps unification enabled and minimal duplication. Only
Rust, rust-bootstrap, Maturin and setuptools-rust gain a maximum of two nodes.
Maturin's build/run Rust edge and setuptools-rust's run Rust edge require
separate backend instances; preserving those edges prevents a backend from
silently selecting the other compiler. Pydantic Core keeps its Rust 1.88
minimum and explicitly selects the stable 1.x range, because Spack otherwise
allows a named nightly to satisfy an open numeric lower bound. No dependency
minimum or upstream dependency type is removed.

The earlier missing-catalogue failure remains under
`history/pre-dated-rust-69873911/`. Failed intermediate separation attempts
are retained with their diagnostic receipt. The initial single-nightly solve
was not accepted as evidence of separate stable and nightly toolchains.

A concrete graph is not a native build. Rust source compilation, bootstrap
binary verification and execution, all native Python dependencies, installed
Arrow/numerical smoke tests, Linux foss stacks and module loading remain
unverified. No upstream submission or completion of issue #1025 is claimed.
