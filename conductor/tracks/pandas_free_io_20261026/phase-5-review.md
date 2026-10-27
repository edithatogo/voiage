# Review Report: Pandas-free I/O Phase 5

## Summary
The direct-import ban and no-extras installed-wheel probe are enforced without
changes to runtime APIs, numerical code, wire formats or bindings. Hosted
acceptance remains required on the delivered PR revision.

## Verification Checks
- [x] **Plan Compliance**: T5.1-T5.3 implemented and locally qualified. Baseline and deliberately failing automation contracts precede activation.
- [x] **Style Compliance**: Ruff, Bandit, typecheck, prose, harness, documentation and metadata normalization pass.
- [x] **New Tests**: 24 controls cover direct/from/submodule/lazy/unreachable/type-only imports, suppression attempts, stubs, empty scans, actual fictional imports, forbidden installed distributions, source shadowing and tox/CI wiring. Three missing-wiring contracts were red before activation; no skip or retry was introduced.
- [x] **Test Coverage**: Aggregate/critical-module policy passes at 94.73%. Runtime source is unchanged; changed-line percentages are vacuous and are not presented as configuration proof.
- [x] **Test Results**: All 14 default tox environments pass; full suite 5,337 passed, 14 existing capability skips. The actual installed no-extras lane reports Python 3.15.1 and empty forbidden distribution/module sets from inside its own environment.

The implementing agent reviewed source-lint/probe code, negative controls,
bootstrap, tox and workflow isolation. No Critical, High, Medium or Low
implementation findings remain. The AST lint has no exemptions and scans all
Python sources and stubs. Producer tests remain outside its runtime scope;
metadata/template strings are not import nodes and remain a Phase 7 boundary.
The probe runs in a fresh isolated subprocess and verifies the package prefix
and distribution version, so checkout imports cannot pass as wheel proof.

CI uses only the locked unpublished verification-tool group to bootstrap the
same no-extras lane on Linux, macOS and Windows. Its target has no extras,
dependency groups or test dependencies. A separate bootstrap qualification
confirms Python 3.15.1, tox/tool versions and absence of the project/producer
libraries from that tooling environment. Updated compatible dependencies were
qualified locally by the full gate; platform-specific packages still require
hosted results. The initial PyPI timeout and its single bounded read-only retry
are retained rather than counted as a passing first audit.

Every runtime, Rust, pinned profile, wire and binding tree remains byte-identical
to the merged prerequisite. Runtime dependency audit found no known
vulnerabilities. Verification retains the 4 GiB sampled process-group guard,
one-hour/free-space limits, native thread limits, two coverage workers and
owned temporary cleanup. Later producer/reference cleanup, native Conda release
proof and the existing frontend audit findings remain separate gates. Exact
revisions, source hashes and native reports are in `phase-5-acceptance-python315.json`.

## Protected-check follow-through

The final CI wiring also runs the source ban directly in the existing protected
`Lint & Type Check` and the real no-extras lane in protected
`Unit Tests (Python 3.15.1)`. This does not depend on adding new matrix job names
to branch protection. The new contract was red before the change, then all 25
enforcement controls and 11 harness cases passed. A fresh full 14-environment
gate passed with 5,338 tests and 14 existing capability skips, 94.73% aggregate
coverage and a 2,056,736 KiB guarded peak. The successful temporary tree was
removed. See `phase-5-protected-ci-python315.json`; the earlier receipt retains
its original candidate evidence. Hosted acceptance remains pending.

## Hosted dependency incompatibility

The first hosted candidate passed all three no-extras platform lanes. The
benchmark jobs exposed pytest-benchmark 5.3.0 accessing `_nodeid`, removed in
pytest 9.2. The same benchmark fails locally with 9.2 and passes with 9.1.1.
The active CI extra and tox dependency declarations therefore cap pytest below
9.2 pending an upstream compatible plugin. No runtime Python downgrade or
private-attribute monkeypatch is used. Historical reproduction environments
retain their existing snapshots. A new full gate qualifies this dependency fix.

The corrected candidate `89adc736` passed all 14 tox environments, 5,338 tests
and 14 existing capability skips, with 94.73% aggregate coverage. The guarded
peak was 1,828,336 KiB and its successful temporary tree was removed. The strict
frontier records pytest 9.2 as an explicit upper-bound blocker. See
`phase-5-benchmark-compatibility-python315.json`; final hosted checks remain
required on the delivered revision.
