# Python 3.15.1 and Polars acceptance: 27 October 2026

The runtime candidate `03b278ba6ec32f8e0d5cd804ab10dcd1a113b2bd` passed every
local and hosted acceptance gate. `acceptance-python315.json` binds the tested
revision/tree, source and log digests, packages, commands, results and hosted
job URLs. The final documentation commit must independently pass the current
PR merge gate. Initial failed diagnostics and historical receipts are retained.

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
  the resource guard. Full coverage run: 5,211 passed, 16 capability skips.
- Aggregate coverage is 95.15%; changed-line coverage is 98.95%; changed-branch
  coverage is 100%. All critical-module coverage thresholds passed.
- The exact scientific/Arrow/interchange, installed clean-package consumer,
  v3 contract, schema, numerical-parity and CLI checks passed. Core-API,
  numerical-reference, R and Julia fixture data is unchanged.
- Rust workspace: 244 tests passed; strict Clippy passed. Current hosted Rust,
  R and Julia binding jobs passed without fixture edits. The Rust 1.85 MSRV is
  retained; a compare/exchange loop preserves saturating telemetry increments
  on both old and current compilers.
- All 17 required hosted checks and all 66 executed hosted jobs passed;
  four non-applicable jobs were skipped. Review threads were resolved. Branch
  protection readback retains strict checking, signed commits, linear history
  and every other rule while replacing retired Python contexts with 3.15.1.
- Restoring pytest caching changed mutation configuration identity. The hosted
  65-mutant universe, source/tool lock identity, 78.462% score and debt of 14
  are unchanged. The reviewed baseline and its external digest were updated;
  the original baseline is retained and no threshold was lowered.

## Performance and resource controls

`benchmark-python315.json` compares hash-verified historical and current
container source on the same Python 3.15.1 environment and unchanged inputs.
The first run exposed query-plan overhead in null validation. Column null
metadata and nested-array checks retained validation and restored all original
bounds: 2D construction 0.098 ms versus 0.208 ms; 3D 0.337 ms versus 0.261 ms;
extraction 0.071 ms; wide-to-long 0.100 ms versus 5.721 ms.

The final gate stayed below the 4 GiB sampled memory ceiling, at about 1.9 GiB
peak. It completed in 373 seconds. Native threads and coverage workers are
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
actual Python 3.15.1. Manuscript content, human venue decisions and submission
status are unchanged. This proves a new replay, not historical execution.

The pandas-free I/O track retains its original T1.1 baseline and may resume at
T1.2 using the accepted container, Polars 2.1 floor and these performance and
recovery receipts. Pandas/xarray dependency removal remains its responsibility.
The translation reference remains planning only. No package publication,
venue submission or external operational acceptance is claimed.
