# R13g delivery verification — 13 November 2026

## Ownership and boundaries

Chat `01a1d44c-538d-7ac3-b556-7f1304675a7b` owns this bounded R13g
evidence reconciliation in
`/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-engineering-acceptance`,
branch `codex/venue-submissions-r13g-20261113`. Its write scope is this
verification record and the v2.2 track's R13g status/evidence metadata. It
does not own the private survey, a maintainer declaration, author-written
communication, an authenticated venue issue, or any editorial decision.
The evidence readback completed at `2026-11-13T11:05:04Z` on macOS arm64.

## Scope continuity

R13g merged as [PR #1265](https://github.com/edithatogo/voiage/pull/1265),
commit `91d11ef7c0a82233cff151df1e038b2a954b6b6d`. Both bounded comparisons
below produced no paths:

```text
git diff --name-only 91d11ef7c0a82233cff151df1e038b2a954b6b6d \
  d688abd2aa71a72d2dd1aa2ed94c2581d61da110 -- \
  docs/release/pyopensci-submission-draft.md \
  specs/submission-readiness/pyopensci-submission-template.json \
  specs/submission-readiness/pyopensci-submission-staging.json \
  scripts/validate_pyopensci_submission_staging.py \
  tests/test_pyopensci_submission_staging.py \
  conductor/tracks/v2_2_release_and_venue_submissions_20260830

git diff --name-only d688abd2aa71a72d2dd1aa2ed94c2581d61da110 \
  2e742ef8d8d4a9a6de3f86d66ac7e8977d31c961 -- [same paths]
```

The only differences from the full-gate source `d688abd2` to the reconciled
head `2e742ef8` are the changelog and Agent-safe engineering acceptance
artifacts. The active configuration inputs remain byte-identical:

| Input | SHA-256 |
| --- | --- |
| `pyproject.toml` | `285123be46b49099e778ae659619386a262d8bf798b3713eed2f4f72027dddc4` |
| `tox.ini` | `75d733fc99623b792474474f72e90631156cc17f02fbd2f6fdb8b9e3bd2a4716` |
| `uv.lock` | `54fa67673aebabe8a8acf8635ba8cd3748793f295c6582ed53aa8c5a8292c912` |

## Reused full local evidence

The clean source commit `d688abd2aa71a72d2dd1aa2ed94c2581d61da110`, tree
`835b527fdd48c50b7a96b20424029bda3782bef1`, ran the full repository gate
after R13g and before the evidence-only changes above:

```text
/Volumes/PortableSSD/GitHub/voiage/.venv-py315/bin/python -m tox run-parallel \
  --workdir /Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-engineering-acceptance-runs/tox-ssd \
  --result-json /Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-engineering-acceptance-runs/tox-final-result.json \
  -p 2
```

It completed successfully in 548.542 seconds on CPython 3.15.1: all 15
configured test environments passed, the coverage run reported 5,860 passed,
29 skipped and 94.83% coverage. The hash-bound resource report is
`e93f7c8bb8ff7a139eadb5b582fc6212566f4d6953cf7a31f0f9aca8cf02d0dc`, the
tox result is `2aaa6d14c9ae978d4d9aa2b2f9090cc51e82c8469626afb71833566028b5dd44`,
and the retained command log is
`3c96266bb58efa75566ae67148ea7d22795f59724e9b7f6ac2d142218fd35ce6`.
Unchanged scope and configuration make this the applicable full local delivery
evidence; it is not represented as a new run.

## Hosted and current checks

PR #1265's exact merge commit has 41 successful and three governed-skipped
checks, with no unsuccessful or non-terminal check. The upstream template was
also re-read through the GitHub API on 13 November: `main` still resolves the
template path to commit `bdc87671b60fdef417f0fea1619a338c715bb42b`, blob
`76f5c4d9bcfa3b57e006ec46b03f4977e6b4f318`, 9,072 bytes and SHA-256
`83b484e506ddf72e02a8dacacba350e032a72c47d112a4e54bedb6982556ce4d`.
That exactly matches the packet pin.

The current focused recheck ran on Python 3.15.1. All 54 tests in
`tests/test_pyopensci_submission_staging.py` passed, the staging validator
reported that the packet is valid and external actions remain unperformed, and
Ruff passed for the changed validator and test module. The focused test used
`-o addopts=''` only because the retained `py315` environment deliberately
lacks the coverage plugin named in the global pytest addopts; the unchanged
full coverage receipt above remains the coverage gate.

```text
PYTHONDONTWRITEBYTECODE=1 /Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-engineering-acceptance-runs/tox-ssd/py315/bin/python -m pytest \
  -o addopts='' tests/test_pyopensci_submission_staging.py
/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-engineering-acceptance-runs/tox-ssd/lint/bin/python scripts/validate_pyopensci_submission_staging.py
/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-engineering-acceptance-runs/tox-ssd/lint/bin/ruff check \
  scripts/validate_pyopensci_submission_staging.py \
  tests/test_pyopensci_submission_staging.py
```

The retained environments are under
`/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-engineering-acceptance-runs/tox-ssd`.
The current-template readback used GitHub's authenticated CLI API against the
public `pyOpenSci/software-submission` repository; it performed no mutation.

## Result and remaining gates

R13g's repository-owned template and delivery checks are complete. This
record does not affirm the new template disclosures, complete the private
survey, supply human-written submission text, create a pyOpenSci issue, or
claim an editorial result. R14 remains the next task: the maintainer must
complete the survey and provide their own submission text, then separately
assess the exact submission disclosures before an authenticated action can be
considered.
