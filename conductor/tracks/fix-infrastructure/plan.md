# Track Implementation Plan: Fix Infrastructure and Configuration

## Phase 1: Fix pyproject.toml Configuration [checkpoint: ]

### 1.1 Fix ruff exclude configuration [x]
- [ ] Remove `"voiage/"` from `[tool.ruff].exclude` — this excludes the entire library from linting
- [ ] Remove `"*.py"` from `[tool.ruff].exclude` — this glob excludes ALL Python files
- [ ] Keep legitimate exclusions: `examples/`, specific broken test files that are genuinely unfixable
- [ ] Verify: `ruff check voiage/` now actually scans the library code

### 1.2 Fix pytest ignore list [x]
- [ ] Audit all `--ignore` patterns in `[tool.pytest.ini_options].addopts`
- [ ] Identify which ignored test files are:
  - Duplicates of active tests (remove the file entirely)
  - Tests for unimplemented features (keep ignore, add TODO comment)
  - Tests that should be enabled but are currently failing (fix or remove ignore)
- [ ] Remove redundant ignores for tests that now pass
- [ ] Target: Reduce ignored test files by at least 80%

### 1.3 Fix Python version consistency [x]
- [ ] `requires-python = ">=3.10"` — keep this (project uses modern features)
- [ ] Remove `Programming Language :: Python :: 3.8` and `3.9` from classifiers (inconsistent with requires-python)
- [ ] Change `[tool.ruff].target-version` from `"py38"` to `"py310"`
- [ ] Change `[tool.mypy].python_version` from `"3.9"` to `"3.10"`

### 1.4 Fix project metadata [x]
- [ ] Update `authors` email from `info@example.com` to actual project email
- [ ] Fix `[project.urls]` — make all URLs consistent (currently mixes `edithatogo` and `doughnut` repos)
- [ ] Verify `Homepage`, `Documentation`, `Repository`, `Changelog`, `Issues` all point to correct repo
- [ ] Update `Development Status` classifier if appropriate (currently `2 - Pre-Alpha`)

---

## Phase 2: Clean Up Stale Test Files [checkpoint: ]

### 2.1 Audit and prune test files [x]
- [ ] List all 93 test files in `tests/`
- [ ] Identify duplicates (e.g., `test_hta_comprehensive.py`, `test_hta_enhanced_95.py`, `test_hta_final_95_achievement.py` — all test the same thing)
- [ ] For each duplicate cluster, keep the best version and delete the rest
- [ ] Target: Reduce test file count to ~30-40 well-maintained files

### 2.2 Categorize remaining test files [~]
- [ ] Mark each test file with its purpose: unit, integration, e2e, benchmark
- [ ] Ensure each test uses appropriate pytest markers
- [ ] Remove `pytest.mark.skip` from tests that should now run

---

## Phase 3: Ruff — Progressive Strictness to Zero Errors [checkpoint: ]

### 3.1 Establish baseline and set strict ruff config [PENDING]
- [ ] Run `ruff check voiage/` with current ignore list — capture baseline error count and categories
- [ ] Review current `[tool.ruff.lint].ignore` list — every ignored rule needs a justification comment
- [ ] Remove unjustified ignores from the `ignore` list (e.g., `ANN001`, `ANN002`, `ANN003`, `ANN201`, `ANN202`, `ANN204` — these disable type annotation checks)
- [ ] Remove `D100`-`D107` ignores (docstring rules) — we want docstrings enforced
- [ ] Set `ruff check` as a blocking gate: zero errors required

### 3.2 Fix ruff errors by category [PENDING]
- [ ] **F401 (unused imports):** Remove all unused imports across voiage/
- [ ] **F841 (unused variables):** Remove or use all unused variables
- [ ] **E/W (style):** Run `ruff format` to auto-fix, then manually fix remaining
- [ ] **I (isort):** Run `ruff check --fix --select=I` to auto-sort imports
- [ ] **B (bugbear):** Fix all potential bugs (mutable defaults, zip without strict, etc.)
- [ ] **C4 (comprehensions):** Replace unnecessary list/set/dict calls with comprehensions
- [ ] **SIM (simplify):** Simplify boolean comparisons, duplicate conditions
- [ ] **PT (pytest):** Fix pytest style issues (yield fixtures, deprecated patterns)
- [ ] **RUF (ruff-specific):** Fix all RUF-prefixed issues
- [ ] **PL (pylint):** Fix all pylint-equivalent issues
- [ ] **TRY (tryceratops):** Fix exception handling anti-patterns (or selectively ignore with justification)
- [ ] **UP (pyupgrade):** Upgrade to modern Python 3.10+ syntax (match statements, walrus, etc.)
- [ ] **PERF (perflint):** Fix performance anti-patterns

### 3.3 Enforce type annotation rules [PENDING]
- [ ] Re-enable `ANN001` (missing type annotations on function args) — add types to all public functions
- [ ] Re-enable `ANN201` (missing return type annotation) — add return types to all public functions
- [ ] For internal helpers where annotations add noise, use `# noqa: ANN` with reason
- [ ] Verify: `ruff check voiage/ --select=ANN` returns zero errors (or documented, justified exceptions)

### 3.4 Enforce docstring rules [PENDING]
- [ ] Re-enable `D101`-`D107` docstring rules for public API
- [ ] Ensure every public class has a class docstring
- [ ] Ensure every public function has a docstring with Args, Returns, Raises sections
- [ ] Ensure every public module has a module-level docstring
- [ ] Verify: `ruff check voiage/ --select=D` returns zero errors for public API

### 3.5 Verify strict ruff pass [PENDING]
- [ ] Run `ruff check voiage/ tests/` — must return zero errors
- [ ] Run `ruff format --check voiage/ tests/` — must pass
- [ ] Add ruff check to pre-commit hook if not already present

---

## Phase 4: mypy — Progressive Strictness to Zero Errors [checkpoint: ]

### 4.1 Tighten mypy configuration [PENDING]
- [ ] Change `python_version` from `"3.9"` to `"3.10"` to match requires-python
- [ ] Remove `warn_return_any = false` — set to `true`
- [ ] Remove `[[tool.mypy.overrides]]` for `voiage.*` that sets `disallow_untyped_defs = false` — enforce typed defs
- [ ] Remove `[[tool.mypy.overrides]]` for `voiage.*` that sets `disallow_incomplete_defs = false` — enforce complete types
- [ ] Add `strict = true` to `[tool.mypy]` base config, then selectively disable only what's truly impossible
- [ ] Ensure `ignore_missing_imports` is set for all third-party libs without type stubs

### 4.2 Fix mypy errors progressively [PENDING]
- [ ] Run `mypy voiage/` with strict config — capture baseline error count
- [ ] **Pass 1: Fix `arg-type` and `return-value` errors** — these are runtime bugs
- [ ] **Pass 2: Fix `var-annotated` errors** — add missing type annotations
- [ ] **Pass 3: Fix `union-attr` errors** — narrow types before attribute access
- [ ] **Pass 4: Fix `assignment` errors** — fix type mismatches
- [ ] **Pass 5: Fix `call-arg` and `call-overload` errors** — fix function call signatures
- [ ] **Pass 6: Fix remaining errors** — operator, index, name-defined, etc.
- [ ] For third-party libraries without types, use `# type: ignore[import-untyped]` or create stub files

### 4.3 Add type stubs for internal modules [PENDING]
- [ ] Create `voiage/stubs/` for any internal modules that can't be fully typed inline
- [ ] Ensure all public API functions have complete type signatures
- [ ] Verify: `mypy voiage/` returns zero errors

### 4.4 Verify strict mypy pass [PENDING]
- [ ] Run `mypy voiage/ --strict` — must return zero errors
- [ ] Add mypy to CI pipeline gate
- [ ] Document any `# type: ignore` comments with justification

---

## Phase 5: Coverage — Progressive to >90% [checkpoint: ]

### 5.1 Set coverage targets and remove exclusions [PENDING]
- [ ] Change `[tool.coverage.run].fail_under` from `0` (implicit) to `90`
- [ ] Remove `voiage/plot/*` from `[tool.coverage.run].omit` — plotting must be covered
- [ ] Remove `*/__init__.py` from omit — at minimum test that imports work
- [ ] Remove `voiage/cli.py` from omit — CLI must be tested
- [ ] Remove `voiage/__main__.py` from omit — entry point must be tested
- [ ] Set `--cov-fail-under=90` in pytest addopts

### 5.2 Measure current coverage baseline [PENDING]
- [ ] Run `pytest tests/ --cov=voiage --cov-report=term-missing` — record baseline %
- [ ] Generate HTML coverage report: `--cov-report=html`
- [ ] Identify all modules below 90% coverage
- [ ] Identify all functions with 0% coverage (completely untested code)

### 5.3 Write missing tests for uncovered code [PENDING]
- [ ] For each module below 90%:
  - [ ] Identify untested functions/branches from coverage report
  - [ ] Write unit tests for each untested function
  - [ ] Tests must cover happy path, edge cases, and error paths
- [ ] Priority order (highest to lowest):
  1. `voiage/methods/basic.py` (EVPI, EVPPI) — core functions
  2. `voiage/methods/sample_information.py` (EVSI, ENBS)
  3. `voiage/analysis.py` (DecisionAnalysis class)
  4. `voiage/methods/structural.py`
  5. `voiage/methods/network_nma.py`
  6. `voiage/methods/adaptive.py`
  7. `voiage/methods/portfolio.py`
  8. `voiage/methods/sequential.py`
  9. `voiage/methods/observational.py`
  10. `voiage/methods/calibration.py`
  11. `voiage/plot/ceac.py`
  12. `voiage/plot/voi_curves.py`
  13. `voiage/cli.py`
  14. `voiage/backends/`
  15. `voiage/stats.py`
  16. `voiage/config.py`
  17. `voiage/exceptions.py`

### 5.4 Enforce branch coverage [PENDING]
- [ ] Enable `branch = true` in `[tool.coverage.run]`
- [ ] Verify branch coverage is also above 90%
- [ ] Add `# pragma: no cover` only for truly unreachable code (e.g., `if TYPE_CHECKING`)

### 5.5 Verify >90% coverage [PENDING]
- [ ] Run `pytest tests/ --cov=voiage --cov-fail-under=90` — must pass
- [ ] Verify branch coverage is also ≥90%
- [ ] Add coverage gate to CI pipeline

---

## Phase 6: Verify Full Tooling Pipeline [checkpoint: ]

### 6.1 Run complete verification suite [PENDING]
- [ ] `ruff check voiage/ tests/` — zero errors
- [ ] `ruff format --check voiage/ tests/` — passes
- [ ] `mypy voiage/ --strict` — zero errors
- [ ] `pytest tests/ --cov=voiage --cov-fail-under=90` — all tests pass, coverage ≥90%
- [ ] Document any remaining justified exceptions

### 6.2 Update taskipy commands [PENDING]
- [ ] Update `[tool.taskipy.tasks]` to reflect new strict settings
- [ ] Ensure `task check` runs ruff + mypy + tests
- [ ] Ensure `task lint` runs strict ruff check
- [ ] Ensure `task typecheck` runs strict mypy

### 6.3 Update todo.md and roadmap.md [PENDING]
- [ ] Rewrite todo.md with all discovered items organized by track
- [ ] Update roadmap.md Phase 4 status to reflect current understanding
- [ ] Add Phase 5 for SOTA methods (efficient EVSI, CEAF, extended dominance, Value of Heterogeneity)
- [ ] Mark which roadmap items are actually implemented vs. stubbed

---

## Phase 7: Autonomous Track Review, Archive and Progression [checkpoint: ]

### 7.0 Pre-Execution Decision Record [PENDING]
These decisions are pre-resolved so autonomous execution can proceed without blocking:

- **Test file pruning (Phase 2):** Keep the file with the most recent modification date and most comprehensive test count. Delete all others in the same cluster.
- **Author email:** Use `voiage@users.noreply.github.com` (GitHub noreply) until a real project email is provided.
- **Repository URLs:** Use `edithatogo/voiage` as canonical (the `doughnut/voiage` references in Changelog/Issues are stale — fix them).
- **Development Status:** Keep `2 - Pre-Alpha` until v0.3 is released.
- **Ruff D-rule scope:** Enforce docstrings for public API only (`D100`-`D107`). Private functions (leading underscore) are exempt.
- **Mypy strict baseline:** Start with `warn_return_any = true`, `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`. Do NOT add `strict = true` globally — it enables too many rules at once. Instead, enable strict rules individually and record which ones are active.

### 7.1 Phase Review Protocol — Execute after EVERY phase (1-6) [PENDING]
After completing each phase above, execute the following protocol **before** marking the phase `[x]` and proceeding:

1. **Record rollback checkpoint:** Record current commit hash in this plan next to the phase: `[rollback: <7-char-sha>]`.
2. **Single commit per phase:** Squash all phase changes into one commit:
   - `git add -A`
   - `git commit -m "conductor(track1): Complete phase <N> of fix-infrastructure"`
   - If multiple commits were made during the phase, squash: `git reset --soft <phase_start_sha> && git commit -m "..."`
3. **Invoke `/conductor:review`** targeting all changes since the previous checkpoint commit.
4. **Apply all Critical and High severity fixes** identified by the review automatically.
5. **Re-run verification:** `ruff check voiage/ tests/ && mypy voiage/ --strict && pytest tests/ --cov=voiage --cov-fail-under=90 -q`
6. **If verification fails:**
   - **Attempt 1:** Fix the failure, commit, re-run verification.
   - **Attempt 2:** Fix the failure again, commit, re-run verification.
   - **If still failing after 2 attempts:**
     - **Escape hatch:** `git revert HEAD~2..HEAD` to rollback to pre-phase state.
     - Mark the specific failing task as `[DEFERRED → v1.1]` with a note explaining the failure.
     - Report to user with details and await guidance **OR** if the task is non-blocking, skip it and complete remaining phase tasks.
7. **Commit review fixes** (if any): `git add -A && git commit -m "fix(conductor): Apply automated review fixes for phase <N>"`
8. **Mark phase complete** in this plan file (change `[PENDING]` → `[x]`).

### 7.2 Track Completion Protocol — Execute after final phase [PENDING]
After Phase 6 is complete and all phase reviews pass:

1. **Invoke `/conductor:review`** targeting the **entire track** (from track start commit to HEAD).
2. **Apply all Critical, High, and Medium severity fixes** automatically.
3. **Re-run full test suite:** `ruff check voiage/ tests/ && mypy voiage/ --strict && pytest tests/ --cov=voiage --cov-fail-under=90`
4. **Commit review fixes:** `git add -A && git commit -m "fix(conductor): Apply final track review fixes for fix-infrastructure"`
5. **Push to remote and verify CI:**
   - `git push origin main`
   - Wait for GitHub Actions: `gh run list --limit 3`
   - If any workflow fails, analyze with `gh run view <run-id> --log-failed`, fix, commit, push, re-check.
   - **Max 3 CI fix retries.** If still failing after 3, halt and report to user.
6. **Archive the track:**
   - `mkdir -p conductor/archive && mv conductor/tracks/fix-infrastructure conductor/archive/fix-infrastructure`
   - Update `conductor/tracks.md`: change `[ ]` → `[x]` for this track, add `[completed: <date>]` and archive link.
7. **Commit archive:** `git add -A conductor/ && git commit -m "chore(conductor): Archive completed track fix-infrastructure"`
8. **Push archive commit:** `git push origin main`
9. **Read next track:** Read `conductor/tracks/activate-public-api/plan.md` and begin execution from Phase 1, Task 1.1.
10. **Announce:** "Track 1 (fix-infrastructure) complete. Starting Track 2 (activate-public-api)."
