# R13a research-handoff review

The published voiage 2.2.0 consumer replayed the historical VOP scenario on
2026-08-30 through a hash-bound CSV exchange. VOP's clean pinned checkout and
frozen environment remained separate from the normal PyPI installation.
Both environments passed `uv pip check`. All 500 generated rows reproduce the
historical CSV digest; EVPI is 0.0 and agrees with the independent NumPy
definition. The July human-use record is preserved, not recertified.

Evidence: `paper/joss-developer-research-use.json` binds the script and both
retained receipts under `paper/research-use/v2.2.0/`. The CSV is retained in
the ignored local evidence directory and can be regenerated from pinned public
VOP inputs. No runtime dependencies, released artifacts or tags changed.

## Verification

- The initial regression test failed because the new handoff script did not
  exist; the completed implementation passes all 30 focused tests.
- Dedicated script coverage: 97 percent, including receipt/CSV alteration,
  invalid matrices, wrong source identity, installed-package boundaries,
  numerical-reference disagreement and no-overwrite behavior.
- `VOIAGE_TEST_UV_CACHE_DIR=/Users/doughnut/.cache/uv uv run tox -p 1 --parallel-live`:
  all 15 environments passed in 538.41 seconds; the full suite passed 4,560
  tests with 16 configured skips. Coverage remains above the 90 percent gate.
- Full log: `/tmp/voiage-vop-handoff-tox-20260830.log`, SHA-256
  `d7790072f79aafe9fb4eaa2da31463c002fc40478856e8ec6dd52b5e44ebda00`.
- Direct Ruff, script type checking, Vale, full Conductor, GitHub
  cross-reference and whitespace validation passed.
- `uv run python scripts/generate_paper_health_example.py --verify-tracked`
  confirmed that the retained worked-example outputs match clean regeneration.

## Review disposition

Checked source identities, data orientation, sampling order, finite-value
validation, exact-byte binding, output preservation and environment separation.
No unresolved correctness finding remains in this slice. Hosted checks are
still required before merge. The replay does not add a personal attestation,
independent adoption, in-process integration or a venue submission. R13b must
still repair current manuscript and disclosure projections.
