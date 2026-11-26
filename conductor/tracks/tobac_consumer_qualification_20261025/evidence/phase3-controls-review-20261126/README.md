# Phase 3 Engineering, Context, Versioning & Security Controls Review (`P3.3`), 26 November 2026

Task `P3.3` reviewed `voiage`'s existing engineering harness, Conductor execution-packet validation, cross-manifest versioning, security/policy lints, and mutation/test-acceleration tooling on branch `codex/tobac-r11-contracts-20261126` (`d0cd33b45d95a1fb383b37b3769885782d90fa08`) using the already-installed `.venv` (`Python 3.15.2`, `ruff 0.17.4`) with no new package installs and no runtime edits.

## 1. Executed Control Checks

| Control | Command | Exit | Result |
| --- | --- | ---: | --- |
| Engineering & workflow harness | `.venv/bin/python scripts/repo_harness.py` | `0` | `status: pass`, `workflow_count: 43`, `finding_count: 0` |
| Control & contract pytest suite | `.venv/bin/python -I scripts/run_installed_pytest.py -o addopts='' tests/test_repo_harness.py tests/test_execution_packets.py tests/test_version_sync.py tests/test_mutation_score.py` | `0` | `60 passed in 1.92s` (`test_repo_harness.py`: 12, `test_execution_packets.py`: 14, `test_version_sync.py`: 18, `test_mutation_score.py`: 16) |
| Pandas-free source policy | `.venv/bin/python scripts/check_pandas_free.py --source voiage` | `0` | `passed: true`, `violations: []` |
| Reference / provenance policy | `.venv/bin/python scripts/check_reference_policy.py` | `0` | `passed: true`, `scanned_files: 5012`, `unapproved: []`, `errors: []` |
| Cross-manifest version sync | `.venv/bin/python -I voiage/versioning.py --repo-root .` | `0` | `validated version synchronization against pyproject.toml @ 3.1.0` |
| Security static lint | `ruff check voiage --select=S --ignore=S101` | `0` | `All checks passed!` |

## 2. Assessment of `pytest-gremlins`, `mutmut`, and Context/Security Controls (`spec.md` Acceptance #6)

- **`pytest-gremlins` & `test-acceleration` extra**: [pyproject.toml](../../../../pyproject.toml#L51-L58) already defines `[project.optional-dependencies].test-acceleration` containing `pytest-gremlins>=1.9.0,<2` (alongside `pytest-testmon>=2.2.0,<3`, `coverage>=7.16.0,<8`, and `vcrpy>=8.3.0,<9`), explicitly documented as *"Local feedback experiments only; not a replacement for CI or coverage."* [scripts/test_acceleration.py](../../../../scripts/test_acceleration.py#L160-L177) already provides an isolated `gremlins` local profile strictly scoped to `voiage/mutation_policy.py` and `tests/test_mutation_score.py` (`--gremlin-operators=comparison,boolean`, `--gremlin-workers=1`). Neither `pytest-gremlins` nor `mutmut` is installed in the minimal qualification `.venv`, and installing them into this environment is unnecessary.
- **Authoritative mutation gate (`mutmut`)**: [pyproject.toml](../../../../pyproject.toml#L95-L97) pins `mutmut==3.8.1` in `dev`, and `[tool.mutmut]` ([pyproject.toml](../../../../pyproject.toml#L273-L286)) configures the cohort targets (`voiage/contracts/capabilities.py`, `voiage/contracts/digests.py`) and test selection. Authoritative enforcement runs through `nox -s contract_mutation` and `.github/workflows/ci.yml` (`mutmut export-cicd-stats`, `scripts/check_mutation_score.py`, `scripts/check_mutation_cohort.py`, and `scripts/run_critical_mutation_lane.py . --threshold 90`). [tests/test_mutation_score.py](../../../../tests/test_mutation_score.py) (16 tests passing) verifies fail-closed mutant accounting, interruption rejection, threshold bounds (`(0, 100]`), and broad baseline score/debt ratcheting.
- **Execution-packet & context boundaries**: [tests/test_execution_packets.py](../../../../tests/test_execution_packets.py) (14 tests passing) verifies `scripts/validate_execution_packets.py` fail-closed guards for required/unknown fields, duplicate/unknown task IDs, dependency cycles, escaping or out-of-scope changed paths (without mutating the input packet), stale `input_sha256` hashes, unsupported result states, pending prerequisite tracks, and zero subprocess execution during validation.
- **Repository harness & security surfaces**: [scripts/repo_harness.py](../../../../scripts/repo_harness.py) (`43` workflows, `0` findings) and [tests/test_repo_harness.py](../../../../tests/test_repo_harness.py) (12 tests passing) enforce SHA-pinned Actions, `setup-uv` `0.13.6`, required CI/security workflows (`ci.yml`, `codeql.yml`, `dependency-review.yml`, `gitleaks.yml`, `operational-assurance.yml`, `rust-fuzz.yml`, `rust-security.yml`, `sbom.yml`, `scorecard.yml`, `zizmor.yml`), `AGENTS.md` context sections, and pinned coverage/mutation operational-assurance thresholds.
- **Versioning & source policies**: [voiage/versioning.py](../../../../voiage/versioning.py) and [tests/test_version_sync.py](../../../../tests/test_version_sync.py) (18 tests passing) validate version `3.1.0` synchronization across Python, Rust, Julia, and R manifests; `check_pandas_free.py`, `check_reference_policy.py` (`5012` scanned files), and `ruff check voiage --select=S --ignore=S101` all pass cleanly.

## 3. Disposition

All reviewed engineering, context, versioning, mutation-policy, and security controls passed cleanly with zero findings (`60/60` control tests passing; `43` workflows verified). Because `pyproject.toml` already configures `pytest-gremlins` in the opt-in `test-acceleration` extra and `mutmut` in `dev`/CI without any demonstrated gap, no new tool stack, package installation, or runtime code change is justified under `spec.md` Acceptance #6.

Full machine-readable receipt: [receipt.json](./receipt.json).
