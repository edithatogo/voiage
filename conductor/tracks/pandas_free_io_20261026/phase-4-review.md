# Review Report: Pandas-free I/O Phase 4

## Summary
The runtime packaging surfaces agree and the clean base wheel resolves and runs
without pandas or xarray. Test producers are explicit unpublished tooling.
Native Conda release proof and later enforcement/reference phases remain open.

## Verification Checks
- [x] **Plan Compliance**: T4.1-T4.3 implemented and locally qualified. Frozen baseline precedes edits; published runtime roots, Conda requirements, tox floor/latest and frontier/preview configurations agree.
- [x] **Style Compliance**: Ruff, typecheck, Bandit, prose and version checks pass. Harness and documentation failures were repaired and their complete environments rerun successfully.
- [x] **New Tests**: Eight policy tests include mutation controls for all four dependency surfaces and explicit producer opt-in. A negative VOP control rejects a missing external producer group. No skip or retry was introduced.
- [x] **Test Coverage**: Aggregate and critical-module coverage policy pass. No runtime source lines changed; changed-line/branch percentages are vacuous, not configuration proof.
- [x] **Test Results**: Full run: 5,310 passed, two failed and 14 capability skips. The stale packet hash and VOP core-dependency assumption were repaired; all four affected tests passed, including the new negative control. Three final VOP tests pass. Ten initial environments passed; complete harness/docs reruns and affected coverage repair qualify the other three. Initial failed results are preserved.

The agent reviewed the dependency, workflow, policy, validator and test diffs.
No unresolved implementation findings remain. Runtime NumPy/SciPy/scikit-learn
bounds and every byte of `voiage/`, Rust, wire, pinned VOP profiles and bindings
remain unchanged. Polars 2.1.0 is both the floor and observed latest stable;
actual imports in both qualified Python 3.15.1 lanes verify it and Arrow 26.

The unpublished producer group is excluded from uv defaults and wheel metadata;
CI/tox explicitly select it so actual producer/oracle tests are retained. A fresh
29-package no-extras wheel environment has neither legacy distribution nor
module, passes dependency checking, executes EVPI and preserves Polars/Arrow
round trips. Python runtime audit: 28 dependencies, zero known vulnerabilities.

Documentation production audit remains non-green: the unchanged frontend lock
has 29 advisories (10 high, 14 moderate, five low). No frontend dependency changes
were made or audit success claimed. The historical Conda source version/hash is
also retained; a matching published migrated sdist and native Conda proof are
separate release gates. Hosted checks must pass on the delivered PR revision.

## Hosted secret-scan follow-up

The hosted generic API-key rule matched the public core API Git tree ID in the
qualification receipt. Git verified that the value is the unchanged tree object.
The correction matches only the complete field/value line in that exact receipt
path. Native Gitleaks 8.30.1 history scanning passes. Controls still detect a
fictional key on the same path and the tree ID under a different key. The
configuration itself has no findings. No broad scan/rule exclusion or history
rewrite was introduced. See `phase-4-security-followup.json`; hosted rescanning
remains required for the updated PR revision.
