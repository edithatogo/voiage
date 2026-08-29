# Phase 3 preflight: submission and distribution assurance

Run on 2026-08-29 from the local staging branch before full project assurance.

## Commands

```sh
uv run python scripts/validate_pyopensci_submission_staging.py .
uv run python scripts/validate_submission_readiness.py
uv run python scripts/reproducible_build.py . \
  --output .assurance/pyopensci-staging-reproducible-build.json \
  --dist-dir .assurance/pyopensci-staging-dist
tox -p 3 -e lint,vale,joss,harness,docs,version-sync
```

The generated reproducible-build report and distributions were local
verification products. Their results are summarized here; no candidate was
uploaded, released, or substituted for the already published `v2.1.0`
artifacts.

## Results

- pyOpenSci staging: passed; all external actions remain unperformed.
- Submission readiness: passed across 22 targets, including ten pyOpenSci and
  ten rOpenSci criteria.
- Reproducible build: passed with a complete, byte-identical artifact set.
  - wheel SHA-256: `30acc475024954922e731c47cd1b8de372c2b855454abae0ea64708ad4b0c28f`
  - wheel inventory SHA-256:
    `8b73469c16a014a8808245fba0ffdddbb2195dff55d32f1199b1f53c1dfce577`
  - source distribution SHA-256:
    `277ee069fd23968786cf89bcebd11b4fc2e3adbd82568b709ce958db75a1747f`
  - source distribution inventory SHA-256:
    `6885fb14a55aaae5cd9d592d2f99695bda379be258f6659045d6d8f499c65d7f`
- Parallel tox slice: all six environments passed.
  - `lint`: Ruff formatting/lint and Bandit passed.
  - `vale`: repository prose checks passed.
  - `joss`: repository-owned manuscript/package checks passed.
  - `harness`: workflow security harness and its tests passed.
  - `docs`: Astro/Starlight validation, check, and build passed.
  - `version-sync`: package-manifest synchronization passed.

The package is ready for the full local tox matrix. Maintainer confirmation of
the `v2.1.0` pyOpenSci candidate remains pending, and no external submission,
survey, contact, review, acceptance, DOI, badge, JOSS referral, pull-request
merge, or release was performed.
