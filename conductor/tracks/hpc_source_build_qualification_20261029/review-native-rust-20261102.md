# Review Report: Native Rust/LLVM qualification

## Summary
The bounded compiler execution path and fail-closed controls are implemented;
native acceptance remains pending the two cold source-build runs.

## Verification Checks
- [x] **Plan Compliance**: Partial - P2 observation controls and P3 compiler
  execution; downstream compiler consumption, profiling and P5 remain open.
- [x] **Style Compliance**: Pass - Ruff, workflow audit and repository harness.
- [x] **New Tests**: Yes - source configuration, installed identity, failed
  commands, detached children, sampler loss, SIGTERM and cleanup races.
- [x] **Test Coverage**: Partial - 45 Linux controls passed. Installed package
  coverage passed at 95%; this is not complete qualification-helper coverage.
- [x] **Test Results**: Local tox environments passed across the full attempt
  and the affected docs/coverage rerun. The initial shallow checkout lacked
  historical evidence objects and the pinned documentation submodule. Restoring
  those inputs yielded 5,791 passed and 27 skipped in the installed coverage gate.
  Earlier runtime-selection and disk stops remain failed receipts.
- [ ] **Native Results**: Stable and dated-nightly cold builds are running.

## Findings and applied fixes

- Rust bootstrap rejects an LLVM build requested at stage0. The first native
  attempt retains its terminal failure; use stage1 for LLVM, then stage2 for
  rustc/std/Cargo/rustdoc. No binary compiler or LLVM substitution is enabled.
- A transient `/proc` I/O permission gap during exec must not disable RSS safety.
  Retain the gap and continue the independent RSS measurement; actual sampler
  failure still stops the stage. A regression exercises the memory stop.
- The Linux subreaper and pidfds own detached/orphaned children. Added real
  interruption, prelaunch-stop and fork-during-cleanup controls. Cleanup rescans
  up to three times and retains the original stop reason; residual children fail.
- Security review required an explicit self-repository reusable-workflow
  reference and explicit exception classes. The harness accepts only an existing
  workflow under `.github/workflows/`; third-party actions remain SHA-pinned.
- Tiny control fixtures use a 1 MiB test reserve so an unrelated filesystem
  condition cannot mask their intended timeout/cleanup assertion. Production
  native builds retain the 8 GiB free-space and 12 GiB sampled RSS limits.
- Workflow path filters include the reused qualification and observation helpers.
- Tox now selects the qualified Python 3.15.1 patch explicitly; the runtime
  assertion remains mandatory and no 3.15.2 promotion is claimed.
- Contributor instructions now require complete evidence history/tags and the
  pinned documentation tool, matching the successful local rerun.
- The nightly run completed source LLVM, then failed on a missing locked `rand`
  version in the separate standard-library workspace. Added `library/Cargo.toml`
  to both compiler vendor profiles after two failing regressions. The source
  archive identities, upstream locks, offline policy and compiler settings stay
  fixed. The corrected nightly run is `41415754392` at `059544da`.

## Evidence and remaining scope

The [receipt manifest](evidence/native-rust-llvm-20261102/manifest.json) retains
failed native/bootstrap and observer evidence, final controls and local gates.
Compiler source archives, bootstrap configuration and driver are identical
between `fd002a0` and the reviewed candidate. The compiler vendor profile now
also verifies the standard-library workspace; its earlier omission is retained
as a native failure. Later observer changes strengthen interruption/cleanup
handling and have separate Linux proof.
Native receipts must still demonstrate terminal success and installed identity,
compile/link EVPI and linkage checks before this PR can merge.

No source-built compiler binaries are retained by this checkpoint. Full Spack
and both EasyBuild installations, downstream use of these compilers, generation
isolation, cgroup/per-directory measurements and hotspot attribution stay open.
Bootstrap SDKs remain authenticated exceptions, never source-built outputs.
