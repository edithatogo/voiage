# Review Report: Native Rust/LLVM qualification

## Summary
Both selected cold compiler profiles built LLVM and stage2 Rust from source,
installed successfully and passed installed Cargo EVPI/linkage probes.
The full installed HPC matrix remains open.

## Verification Checks
- [x] **Plan Compliance**: Partial - P2 observation controls and P3 compiler
  execution; downstream compiler consumption, profiling and P5 remain open.
- [x] **Style Compliance**: Pass - Ruff, workflow audit and repository harness.
- [x] **New Tests**: Yes - source configuration, installed identity, failed
  commands, detached children, sampler loss, SIGTERM and cleanup races.
- [x] **Test Coverage**: Partial - 51 Linux controls passed. Installed package
  coverage passed at 95%; this is not complete qualification-helper coverage.
- [x] **Test Results**: Local tox environments passed across the full attempt
  and the affected docs/coverage rerun. The initial shallow checkout lacked
  historical evidence objects and the pinned documentation submodule. Restoring
  those inputs yielded 5,791 passed and 27 skipped in the installed coverage gate.
  Earlier runtime-selection and disk stops remain failed receipts.
- [x] **Native Results**: Passed - stable `41436471133` and dated-nightly
  `41436475314`, tested source `24ae6cb6`. All 12 stages per profile completed
  with successful owned-process cleanup. Installed Cargo compiled and ran
  `EVPI=5` using the installed source-built compiler; linkage resolved.

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
  fixed. Intermediate nightly run `41415754392` used this repair at `059544da`.
- Both standard-library fixes completed LLVM and stage2 Rust, including std,
  Cargo and rustdoc. Installation then exposed copyright generation's complete
  upstream workspace closure. The qualifier now checks the authenticated
  `default_paths_to_vendor` source contract and includes all 13 manifests.
- Cargo's offline `locate-project --workspace` resolves the 12 unique original
  locks, including `opt-dist`'s shared parent lock. Missing/external locks and
  contract omissions fail before compilation. Four focused regressions cover
  these boundaries; all 24 affected source/compiler tests pass on Python 3.15.1.
- The full closure passed in both compiler cells of run `41435528210`. One
  unrelated uv manifest-download timeout received one targeted read-only job
  rerun; its first-failure log remains retained and the workflow passed.
- The driver now runs upstream copyright generation before the costly LLVM
  source stage. Accepted native attempts are stable `41436471133` and nightly
  `41436475314`, both at `24ae6cb6`. Their source/archive/compiler settings remain
  frozen; installation and installed probes passed in both profiles.
- A later local full-tox attempt stopped at the disk-space preflight. It is
  retained as a capacity stop, not a passing test result. Earlier full-tox
  evidence applies to unchanged runtime inputs; current hosted gates and the
  51 Linux qualification controls remain required for the final revision.

## Evidence and remaining scope

The [receipt manifest](evidence/native-rust-llvm-20261102/manifest.json) retains
failed native/bootstrap and observer evidence, final controls and local gates.
Compiler source archives and bootstrap settings stayed frozen while the vendor
closure, observation cleanup and early notice check were repaired. The ten
native input Git objects bind accepted revision `24ae6cb6`; later receipt and
documentation commits retain those exact objects. Both GitHub artifact archive
digests matched their downloaded bytes. Each retained file also has stored and
original SHA-256 values. The default LLVM target sets match the earlier cold
baseline; no backend or experimental target was removed.

Review covered the workflow, source/lock resolution, observation boundaries,
negative controls and terminal receipts against the plan and coding guidance.
No unresolved correctness or security finding remains in this bounded compiler
checkpoint. Final-head hosted gates and the protected merge remain required.

No source-built compiler binaries are retained by this checkpoint. Full Spack
and both EasyBuild installations, downstream use of these compilers, generation
isolation, cgroup/per-directory measurements and hotspot attribution stay open.
Bootstrap SDKs remain authenticated exceptions, never source-built outputs.
