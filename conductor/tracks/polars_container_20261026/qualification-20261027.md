# Python 3.15.1 and Polars acceptance: 27 October 2026

The accepted core/runtime revision is `6675f6b5200df3e21d488af48f15e8609e6c1ee8`.
The isolated replay follow-up is `23273b335e82b528b7e8aef772266757eabed0d5`.
`acceptance-python315.json` binds exact tested trees, commands, source/log digests,
packages, source retrieval and hosted job URLs. The final documentation commit
must independently pass the protected merge gate. Original diagnostics and the
prior superseded acceptance receipt remain preserved.

Late review reproduced eight mutable-NumPy aliasing failures. NumPy factories
now copy input buffers once; public frame constructors explicitly snapshot even
externally borrowed buffers. Subclasses retain their normal constructor and
post-init hooks. C/F arrays, perspective arrays, dictionaries, copies, empty and
single-row frames, chunked frames and exact NaN/signed-zero bits are covered.
All 80 focused tests pass. Private validated factories avoid redundant copies
without changing public signatures or bypassing validation.

## Supported runtime and capability boundary

The actual repository-local interpreter is Python 3.15.1. All active workflow
runtimes, package constraints, locks, tox/nox configuration, helper defaults and
native Python ABI now select the qualified series. Polars 2.1.0 is the working
floor. Older Polars 1.42.1 and 1.44.2 remain unqualified after their timezone
Series construction failed on this runtime. The frozen clean environment,
native extension build and strict dependency-frontier audit passed.

The maintainer explicitly approved marking the optional PyTorch engine
unavailable while stable CPython 3.15 wheels are absent. Its API and explicit
constructor diagnostics remain, and no engine is substituted silently. The
supported core and scientific dependencies are qualified. Historical v1/v2
contracts and published HPC release profiles remain immutable; verified
published-sdist metadata is separate from current development requirements.

## Acceptance evidence

- All 13 default tox environments passed using `tox run-parallel -p 2` under
  the resource guard. Full coverage run: 5,226 passed, 16 capability skips.
- Aggregate coverage is 95.15%; changed-line coverage is 99.06%; changed-branch
  coverage is 100%. All critical-module coverage thresholds passed.
- The exact scientific/Arrow/interchange, installed clean-package consumer,
  v3 contract, schema, numerical-parity and CLI checks passed. Core-API,
  numerical-reference, R and Julia fixture data is unchanged.
- Rust workspace: 244 tests passed; strict Clippy passed. Current hosted Rust,
  R and Julia binding jobs passed without fixture edits. The Rust 1.85 MSRV is
  retained; a compare/exchange loop preserves saturating telemetry increments
  on both old and current compilers.
- All 17 required hosted checks passed on the replay follow-up, and the
  unchanged native binding jobs passed. Supplementary paper assurance retains
  its separate job state in the receipt; it does not replace a required gate. Review threads were resolved. Branch
  protection readback retains strict checking, signed commits, linear history
  and every other rule while replacing retired Python contexts with 3.15.1.
- Restoring pytest caching changed mutation configuration identity. The hosted
  65-mutant universe, source/tool lock identity, 78.462% score and debt of 14
  are unchanged. The reviewed baseline and its external digest were updated;
  the original baseline is retained and no threshold was lowered.

## Performance and resource controls

`benchmark-owned-buffers-python315.json` compares hash-verified historical and current
container source on the same Python 3.15.1 environment and unchanged inputs.
The first run exposed query-plan overhead in null validation. Column null
metadata and nested-array checks retained validation and restored all original
bounds: 2D construction 0.100 ms versus 0.228 ms; 3D 0.339 ms versus 0.267 ms;
extraction 0.071 ms; wide-to-long 0.113 ms versus 5.876 ms.

The final gate stayed below the 4 GiB sampled memory ceiling, at about 1.7 GiB
peak. It completed in 459 seconds. Native threads and coverage workers are
bounded, temporary trees live outside the checkout on the SSD, successful
temporary files are removed, and detached Git monitors are disabled through
process-local configuration. The guard also enforces elapsed-time/free-space
limits and records termination reasons. Its control and cleanup tests passed.
No underlying memory leak is asserted fixed by these safeguards.

## Historical paper replay and handoff

The old immutable replay lock has no supported PyArrow 25 wheel for Python
3.15. It remains preserved as historical evidence. The separate v3 profile
`paper/reproduction-manifest-python315.json` binds an exact source commit and
qualified project/lock archives, retains the original scientific inputs and
output hashes, and passed both isolated native replay and hosted assurance on
actual Python 3.15.1. A fresh shallow main checkout initially lacked the pinned
PR source. The verifier fetched that exact source only inside its private clone;
the real replay passed with the caller checkout unchanged. All 49 focused replay
and JOSS tests passed. This preserves post-squash reproducibility without
rewriting history or the frozen environment. Manuscript content, human venue decisions and submission
status are unchanged. This proves a new replay, not historical execution.

The pandas-free I/O track retains its original T1.1 baseline and may resume at
T1.2 using the accepted container, Polars 2.1 floor and these performance and
recovery receipts. Pandas/xarray dependency removal remains its responsibility.
The translation reference remains planning only. No package publication,
venue submission or external operational acceptance is claimed.
