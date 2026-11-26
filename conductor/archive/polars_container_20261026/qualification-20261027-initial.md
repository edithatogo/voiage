# Container qualification and I/O handoff: 27 October 2026

The container implementation is delivered locally. Final acceptance is open:
the required Python 3.15.1 full gate and hosted PR checks have not passed.
The earlier completed labels described implementation, not accepted delivery.
Historical evidence remains unchanged; this record supersedes its closeout claim.

## Current evidence

All Python commands in this follow-up used explicitly selected repo-local
CPython 3.15.1 interpreters. The existing Python 3.12 environments were retained.
`qualification-receipt.json` binds source/configuration hashes, dependency
versions, commands, results and retained local log digests.

- Initial focused qualification: 119 tests passed on the unchanged implementation.
- Review then reproduced six failures: unsupported array element types/N-D
  arrays, nulls inside perspective arrays, and caller mutation of supplied
  frames affecting both container types.
- The repairs enforce one-dimensional `Array(Float64, n)` columns, reject
  nested nulls and isolate the supplied frame and perspective-label list.
- Focused schema, parity, interchange and v3-contract verification after repairs:
  126 tests passed. A further 70 core numerical/ingestion tests and 32
  backend/JAX regression tests passed. These selections overlap and are not
  a full-suite or coverage result.
- Conductor normalization and v3 governance: 8 tests passed. GitHub
  cross-reference validation passed.
- The Polars 1.42.1 declared-floor run initially failed eight tests. Switching
  the nested-null check to the expression API repaired seven container-path
  failures. The final floor run has 125 passing tests and one failure inside
  Polars timezone-Series construction (`NoneType.dt`). Its first failure and
  final result are both retained; the floor is not qualified on Python 3.15.1.
- The typechecker exposed the lazy JAX object's use as a type. Static
  annotations now use `jax.Array` under `TYPE_CHECKING`, retaining lazy runtime
  imports. The typecheck rerun and a separate Python 3.15 static target passed
  without adding ignores. Ruff check/format passed.

## Full-gate result and blockers

The full tox environment set was attempted with concurrency two. Local CLI
overrides selected Python 3.15.1 for every environment, replaced the old
Python matrix with `py315`, used an isolated tox work directory and used uv
for installation. Scientific binary-wheel requirements made wheel availability
explicit. This was a qualification attempt, not an unchanged default-gate pass.

Lint, Vale, JOSS and frontier-contract passed. Harness, docs, version-sync,
ingestion-conformance, py315 and dependency-bound environments failed setup;
coverage was not established. Typecheck initially failed, then passed after
the annotation repair. The declaration failures are:

| Existing declaration | Observed Python 3.15.1 gap |
|---|---|
| `scipy>=1.16.3,<1.17` | No matching wheel in the declared range |
| `pandas>=1.3,<3` | No matching wheel in the declared range |
| `pyarrow>=25,<26` | No matching wheel in the declared range |
| `jax>=0.7.1,<0.8` | Matching `jaxlib` wheels absent |

The isolated latest-dependency environment used SciPy 1.18.1, pandas 3.1.0,
PyArrow 26.0.0, JAX/jaxlib 0.12.0.1 and Polars 2.1.0. It is a preview
qualification environment outside the declared ranges, not a supported clean
package installation or permission to promote these bounds automatically.

The preflight `uv lock --upgrade` and strict dependency-frontier audit completed.
The strict audit checks range/lock policy, not interpreter wheels. Its refreshed
lock and report were retained locally; the unqualified lock refresh was not
adopted. No generated Conductor context pack was present, so the canonical
index-linked context, track plans and specs supplied the context.

The active Python defaults, CI matrices, source-build ABI policy, helper
scripts, setup guidance and dependency bounds still need a coherent Python
3.15.1 migration. Do not declare that migration complete or run older Python
to turn this qualification green. The pandas-free track also needs to remove
its obsolete live-dependency declarations and qualify its replacement engines.

A fresh benchmark is retained as an observation only. It does not revalidate
the older cross-runtime performance bounds: interpreter/dependencies changed,
and the benchmark actually uses 5,000 samples for the 3D fixture versus 20,000
for 2D. A controlled matched-environment comparison remains required.

## Next work

1. Complete the coherent Python 3.15.1 runtime/dependency qualification and
   supported clean-package gate, preserving the named optional-acceleration
   boundary and the existing wire/binding contracts.
2. Rerun failed full-gate environments and affected checks on that candidate;
   require the full final gate, coverage, controlled performance evidence and
   current hosted checks before container acceptance/merge.
3. I/O T1.1 is frozen in `../pandas_free_io_20261026/phase-1-baseline.json`:
   36 pandas-use hits and 42 signatures. Continue with failing behavioral
   tests and T1.2; do not treat the inventory as a completed I/O migration.
4. Keep the translation reference pending the accepted container and its
   documented semantics. No release/publication claim is made here.
