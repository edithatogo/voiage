# Phase 5 automated review

## Scope and outcome

The stable dependency refresh, isolated preview candidates, profiling evidence,
and CI/test optimizations were reviewed. No Critical, High, or remaining Medium
repository finding was identified after the observation-harness repairs.

Stable dependencies are at the latest versions allowed by reviewed bounds.
Hosted run 33261056383 exercised every major candidate and both deterministic
Python shards. No candidate was promoted: successful runtime probes remain
preview-only until their full promotion gates pass; Ruff 0.16 is rejected for
now because it enables 20 additional findings; Python 3.15 is blocked by the
absence of a `jaxlib` cp315 artifact.

The first Rust accelerator observation revealed that setting
`SCCACHE_GHA_ENABLED` alone does not initialize the GitHub cache backend. The
workflow now uses the pinned official sccache action. Independently, pinned
nextest ran all 221 Rust tests locally in 2.502 seconds after a 21.94-second
build, with all tests passing.

## Measured results

- Full Python suite: 3,369.19 seconds serial baseline to 248.49 seconds with
  eight workers locally; six workers remain the conservative stable CI choice.
- Repository harness: 162.49 to 2.59 seconds.
- Full 15-environment tox matrix: 1,097 to 643 seconds, a 41.39% observational
  reduction with all environments passing.
- Exact-head PR #1034: 66 successes, five intentional skips, zero failures,
  95.11% coverage.
- Documentation: duplicate polyglot build removed while retaining all 1,030
  generated pages and link validation.

The full release gate remains fresh, non-sharded, fail-closed, and unchanged.
