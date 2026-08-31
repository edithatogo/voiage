# Bounded local test feedback

Issue #1028 adds an optional `test-acceleration` extra and the `test-feedback`
tox environment. Neither belongs to the required tox matrix. Full clean
coverage, mutation, supported Python, release, and hosted checks remain the
promotion authority. Selective runs cannot establish whole-suite correctness.

```sh
uv sync --extra ci --extra test-acceleration
uv run --no-sync python scripts/test_acceleration.py testmon tests
# Equivalent isolated environment:
tox -e test-feedback -- testmon tests
```

## Selection and invalidation

The runner uses serial pytest-testmon with the C coverage core. It disables
pytest-cov and gremlins, clears inherited pytest options and shard selection,
and accepts only existing paths below `tests/`. A process lock prevents two
local profiles from sharing a database concurrently. Inspect a stale lock
before removing it; a timeout is not evidence that a process stopped.

The local database lives below `.conductor/local/test-acceleration/`. Its
signature includes the checkout path, interpreter, installed distributions,
selected paths, tracked and unignored file inventory, and non-Python file
contents. Dependency, configuration, fixture, native-source and test-inventory
changes rebuild the baseline. Testmon tracks edits to existing Python code.
Failed or interrupted runs invalidate the completed-run signature. Exit code 5
is accepted only for an empty selection against an unchanged successful
baseline. A cold empty collection remains an error.

Run ordinary tests after changes involving runtime-generated code, subprocess
behavior, dynamic imports, environment-dependent behavior or external state.
Testmon dependency tracking cannot prove these cases safe. `pytest-picked` is
not installed: direct filename selection adds no dependency information beyond
this profile and can miss affected consumers.

## Bounded experiments

```sh
uv run --no-sync python scripts/test_acceleration.py ordinary tests/test_mutation_score.py
uv run --no-sync python scripts/test_acceleration.py gremlins
uv run --no-sync python scripts/test_acceleration.py ctrace tests/test_mutation_score.py
uv run --no-sync python scripts/test_acceleration.py sysmon tests/test_mutation_score.py
uv run --no-sync python scripts/evaluate_http_replay.py
```

Gremlins always targets only the mutation-score policy, its existing tests,
comparison and boolean operators, and one worker. It is an experiment alongside
the required Mutmut gate, not a replacement. Keep surviving mutants and tool
failures visible; neither is a successful mutation outcome.

The experiment was executed with Python 3.14.6; other interpreter combinations
are not claimed as measured. The coverage profiles use isolated
branch-coverage files and emit the actual
measurement-core diagnostic. Python 3.12 and 3.13 cannot use sysmon for branch
coverage. Dynamic contexts, tracing plugins and some concurrency modes are
incompatible; testmon therefore explicitly uses ctrace. A measured sysmon
result does not change CI's coverage core or thresholds.

The HTTP experiment accesses only public PyPI metadata for one fixed package
version, with no proxy or credential handler. It rejects other request URLs,
methods and bodies, removes transport headers, verifies the replay digest,
and deletes its temporary cassette. Replay runs with recording disabled.
The generated in-process mock remains the preferred deterministic unit-test
transport. Real response replay is reserved for explicitly scoped public
integration investigation; never record application traffic or private data.

Freezegun is deliberately excluded. Scientific-review expiry checks already
accept `at_time` or `now`, exercised by explicit dated tests. Global clock
freezing could alter asyncio, timeouts and concurrency and provides no benefit
for these existing injectable clocks.

Reports retain actual executed counts, pytest collection counts where emitted,
subprocess time and the total measured runner phase (including signatures). Absolute timing depends on host load;
no speed threshold or broad performance claim follows from one sample.

## Primary tool references

- [Testmon configuration](https://www.testmon.org/)
- [Gremlins configuration](https://pytest-gremlins.readthedocs.io/en/latest/guide/configuration/)
- [Coverage measurement cores](https://coverage.readthedocs.io/en/latest/config.html#run-core)
- [VCR filtering](https://vcrpy.readthedocs.io/en/latest/advanced.html)

## Observed trial

The dated receipt is
`specs/submission-readiness/test-acceleration-evaluation-20260831.json`.
The final policy slice includes the new zero-denominator regression:

| Local profile | Collected | Executed | Pytest process | Measured runner phase |
|---|---:|---:|---:|---:|
| Ordinary | 16 | 16 | 43.54 s | 48.80 s |
| Cold testmon | 16 | 16 | 36.81 s | 39.06 s |
| Unchanged warm testmon | 16 | 0 | 20.22 s | 23.37 s |

The warm run reused the successful signature without invalidation. These are
single observations under shared host load, not a portable speed comparison.
The final gremlins trial killed all 29 generated comparison/boolean mutants,
with no survivors, timeouts or errors. The full report is retained beside the
receipt; the existing Mutmut gate remains unchanged.

Both coverage cores reported 48/48 statements and 10/10 branches. Their raw
trace arcs differed, so only the reportable totals are equivalent. Sysmon took
52.07 s versus 48.19 s for ctrace on the same slice; there is no measured reason
to promote it. Gremlins initially exposed a dynamic-context incompatibility;
explicit ctrace removed that warning and exposed missing candidate/baseline
zero-denominator assertions, now covered by an additional regression.
