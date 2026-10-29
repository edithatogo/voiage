# G04 shard aggregation preflight

Integrator lane: `codex/ci-shard-aggregation`, worktree
`/Users/doughnut/.codex/worktrees/ci-shard-aggregation/voiage`. Native scope:
none. Base: `ba17ae35c74e1285d6e66ddca63da0a057c4abee`. Next operation:
design a failing behavioral witness for an actual aggregate before editing a
workflow. This is not acceptance evidence.

The required operational assurance job collects and checks coverage in one
job. The preview's deterministic shard matrix is explicitly non-blocking; its
`continue-on-error` setting can hide a failed shard from downstream job-result
aggregation. Neither establishes the required failure or cancellation result
for a real shard aggregate. The previously rejected attempt to change the
preview's non-blocking contract remains recorded in the acceptance audit.

Frozen SHA-256 inputs:

- `.github/workflows/operational-assurance.yml`: `cf086b99ce9fc7e8dd639c6ca11a7f163076f5c617d8fbd792fba1ac0c29fce0`
- `.github/workflows/dependency-preview.yml`: `861bae6ff6c8d9dbbf1eda73cc2b4a4a723dd84c4e41c908b7ca24adcc534b21`
- `scripts/coverage_artifact_identity.py`: `386a6508079ef136b56f7cda1da22574c8570af984d1f96447d3f0e37bd3bee0`

Any proposed aggregate must consume actual shard outcomes and identity-bound
coverage artifacts, fail on missing, failed or cancelled inputs, and retain the
existing required job names and coverage thresholds. A local status simulator
alone cannot prove GitHub cancellation behavior. Before implementation, choose
the exact workflow and test subset, obtain a meaningful red assertion, and
plan hosted positive and negative evidence. No cold/warm saving is claimed.

The first implementation slice owns `scripts/validate_shard_inputs.py` and
`tests/test_ci_shard_inputs.py`. The red witness invokes the actual CLI
with a two-shard fixture and requires failure for a failed or cancelled shard,
a missing report, or an identity mismatch; the complete valid fixture must
pass. It must not accept an empty collection, duplicated shard IDs or a report
whose bytes do not match its receipt. Validate with Python 3.15.1 under the
resource guard using `pytest tests/test_ci_shard_inputs.py -q`. Integration
into an actual hosted workflow and real cancellation evidence remain required
after the local component is qualified; a component test is not track closure.

The guarded Python 3.15.1 behavioral run first failed to import the repository
conftest because the isolated test environment lacked NumPy; that was not a red
witness. With `--noconftest -o addopts=` for this standalone subprocess test,
the stub produced the intended seven assertion failures and one positive pass.
After implementation and a separate expected-identity input, nine focused tests
pass under the guard; Ruff formatting and lint pass. These tests exercise local
fixtures only. The expected identity must be generated independently from the
actual hosted execution, and the component must be integrated with real shard
outcomes and coverage aggregation before any G04 acceptance claim.

A second adversarial review found that the first component accepted an empty
coverage file and trusted the manifest's own shard count. A guarded red test
reproduced the empty-report false green; the revised component takes the count
independently and rejects empty coverage. Eleven focused tests and Ruff now pass.
The component still does not authenticate its inputs, merge coverage data, or
exercise a hosted job; those requirements remain open.

A further guarded red run showed that nonempty reports with zero executed lines
or malformed line values were also accepted. The component now checks those
shapes; 13 focused tests and Ruff pass. This remains input validation only:
syntactically plausible data is not authenticated coverage evidence.

The required coverage policy evaluates branch arcs as well as line totals.
JSON summaries alone must not be hand-merged as proof: qualification needs
identity-bound native coverage data, coverage.py's native combination, and the
existing policy gate on the combined report. The input validator is explicitly
named and documented as such so its success cannot be mistaken for that gate.

The next bounded slice adds `scripts/combine_shard_coverage.py` and
`tests/test_combine_shard_coverage.py`. The witness must create genuine
coverage.py data for two disjoint portions of one checked-in source file,
combine it with coverage.py, and invoke `check_coverage_policy.py` against the
actual temporary Git checkout. A missing or altered native artifact, failed
shard, or unmet policy must fail without a successful policy receipt. Use the
locked coverage.py version and Python 3.15.1 under the resource guard. This
local witness does not replace actual hosted shard and cancellation evidence.

The native-data witness uses coverage.py 7.17.0, the locked version, to combine
two actual coverage databases and invokes the existing policy on a temporary Git
checkout. A changed configuration was initially accepted; a guarded red run
exposed it, and the component now verifies checkout lock and configuration hashes
and the executing interpreter before combining. A second red run showed that
line-only data could falsely pass a branch-enabled configuration; the component
now requires native arcs for that configuration. Nineteen focused tests pass,
including real line and arc combination, tampered native data and a failing
policy. Ruff passes. Actual hosted identities, authentic shard outcomes,
cancellation handling and policy integration remain unqualified.

A guarded dirty-source witness then showed that matching HEAD alone was
insufficient. The combination command now rejects a dirty checkout and checks
the actual HEAD before reading coverage. Twenty focused tests and Ruff pass;
this remains local component evidence, not hosted acceptance.

Another guarded red witness changed a shard's JSON report and updated its
digest while leaving native data unchanged; the first combination incorrectly
passed. The command now regenerates each report from its native data and
compares file-level results before combining. Twenty-one focused tests and Ruff
pass. A matching digest and regenerated report still do not authenticate a
producer or establish actual hosted shard outcomes.

The next bounded slice owns `scripts/prepare_shard_receipts.py` and
`tests/test_prepare_shard_receipts.py`. A producer receipt will hash the actual
native and JSON files and record source, lock, interpreter, features and
configuration. The aggregate will independently derive its expected identity,
read fixed per-shard artifact locations, and refuse any matrix result other
than success. A local red/green witness must cover absent artifacts, failed and
cancelled matrix results, and identity mismatch. Hosted integration must use
real workflow `needs` results and prove actual cancellation separately; a
fixture is not a hosted outcome.

The receipt helper now derives identity from a clean checkout, hashes fixed
per-shard artifacts and rejects non-success matrix results before writing an
aggregate manifest. The guarded local suite passes 27 tests, including an actual
producer-receipt to native coverage combination and policy evaluation. This is
still fixture evidence: a workflow must supply authentic job outcomes and
independently produced artifacts, and hosted cancellation must be observed.

A separate pull-request qualification workflow now runs two deterministic
partitions and aggregates native branch coverage only after the actual matrix
result succeeds. Its per-partition zero threshold is provisional: the unchanged
repository policy is applied to the combined result, and the existing required
operational job remains authoritative. Actionlint, Ruff and the 27 guarded local
component tests pass. This workflow is uncommitted and has not run on GitHub;
its positive result and actual failed/cancelled shard behavior remain open, as
do the final full local gate and self-review. No replacement or performance
saving is claimed.

A guarded Python 3.15.1 runtime probe used the locked pytest-cov and xdist
versions, the workflow's native coverage filename and report options, and an
actual two-worker pytest run. The resulting branch database regenerated the
same file-level JSON report. The aggregate comparison base now follows the
existing operational job's provenance-floor rules. This probe does not prove
GitHub artifact transport, matrix failure propagation or cancellation.

An adversarial guarded test exposed that omitting `--matrix-result` from the
collector defaulted to success and could create a manifest without an outcome.
The collector now rejects an omitted result, and its workflow supplies the real
`needs` result explicitly. The targeted guarded suite passes 28 tests; Ruff and
actionlint pass. Hosted failure and cancellation evidence remain outstanding.
