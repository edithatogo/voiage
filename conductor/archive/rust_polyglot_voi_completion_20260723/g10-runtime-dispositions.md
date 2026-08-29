# G10 runtime dispositions and installed-evidence protocol

This packet records the repository-owned disposition for every advertised
runtime. A `blocked` result is evidence of an unmet environment prerequisite,
not a parity failure or a promotion approval. The shared fixture manifest and
the evidence schema remain authoritative.

| Runtime | Surface | Runner class | Command | Current result | Required next action |
|---|---|---|---|---|---|
| Rust core | `rust/` numerical workspace and C ABI | macOS arm64 local/hosted Rust 1.85+ | `cargo test --manifest-path rust/Cargo.toml --workspace --all-targets` | `partial_pass`: core, ABI, numerics, properties and benchmarks pass; PyO3 `_core` needs matching `libpython3.13.dylib` | Run on a Python-matched CI runner and record toolchain, symbols, output and diagnostic hashes. |
| Python | PyO3 façade and public package | clean installed wheel, Python 3.12–3.14 | `uv run pytest tests/test_perspective_conformance_v1.py tests/test_perspective_promotion_evidence.py -q` | `passed` for the recorded checkout conformance slice; installed-wheel evidence is still required for promotion | Repeat from the built wheel against the immutable fixture manifest and capture output/diagnostic hashes. |
| R | `r-package/voiageR` C ABI consumer | clean R 4.3 environment | `Rscript -e 'testthat::test_dir("r-package/voiageR/tests/testthat", reporter="summary")'` | `blocked`: package is not installed in the current environment | Install the package and native library in a clean R 4.3 runner, then record ABI symbols/layout and shared-fixture hashes. |
| Julia | `bindings/julia` C ABI consumer | clean Julia 1.10/1.11 environment | `julia --project=bindings/julia -e 'using Pkg; Pkg.test()'` | `blocked`: `libvoiage_ffi.dylib` is unavailable to the loader | Build/expose the native library, rerun both supported Julia lines, and record loader/platform metadata plus fixture hashes. |
| Mojo | upstream interop boundary | external upstream toolchain | no local command | `not_applicable` locally; no executable or binding is claimed | Reopen only when an upstream Mojo toolchain and an approved Rust interop contract exist. |

## Completion rule

G10 remains in progress until Rust, Python, R and Julia have clean installed
runs over the same fixture-manifest hash, including canonical outputs,
diagnostics, tolerance results, ABI/layout checks and unsupported-capability
cases. A complete packet then requires panel review and maintainer promotion;
scientific validity, signing, registry publication and parent closure remain
separate gates.
