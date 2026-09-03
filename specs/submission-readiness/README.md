# Submission readiness contracts

`targets.json` is the authoritative cross-venue inventory for package
registries, archives, identifiers, journals, community software review, and
sustainability affiliations.

The dated current baseline is
`current-requirements-source-manifest-20260829.json` together with
`current-requirements-baseline-20260829.md`. It pins the primary-source
revisions used by the canonical comprehensive-hardening track and must be
refreshed before candidate freeze or any external venue action.

Repository evidence state is refreshed separately in
`cross-venue-evidence-refresh-20260903.json`. It binds the terminal checks and
exact tree-equal merge for final-root PR #1087 while retaining native HPC
qualification and upstream action as pending. A status refresh never extends
the lifetime of the dated external criteria baseline before an external action.

The contract records repository evidence and outstanding gates without
authorizing external action. Preparation, submission, review, acceptance,
publication, and indexing are distinct states. Criteria URLs must be refreshed
before an author decides to submit because external policies can change.

## Execution plan

Every retained target belongs to at least one repository-controlled execution
lane in `targets.json`. The validator rejects a required target that has no
lane or a lane without a `voiage` GitHub issue. The lanes deliberately close
only repository work:

- #614 maintains criteria, evidence, authority, and status boundaries for all
  destinations;
- #296 owns the current paper, journal, and identifier boundaries while #312
  retains the later-arXiv action; historical #299 supplied the repository
  manuscript package;
- #1037 governs the current pyOpenSci-first execution while preserving the
  private survey, human-written communication, authenticated submission, and
  venue outcomes as separate gates; historical #616 supplied repository
  readiness evidence;
- #615 matures the R package for possible rOpenSci and R Journal routes;
- #1026 retains current sustainability and persistent-identifier work after
  historical #617 recorded the distinct-publication decision;
- #1025 maintains current Spack/EasyBuild recipe and build evidence before any
  HPC upstream or curation decision; historical #622 supplied the initial
  locally testable recipe contract.

No lane authorizes an inquiry, upload, submission, application, review, or
acceptance claim. Those actions remain human or external even after all
repository-controlled tasks are complete.

Validate the inventory from the repository root:

```sh
uv run python scripts/validate_submission_readiness.py
uv run pytest tests/test_submission_readiness_contract.py --no-cov
```

Specialised contracts remain authoritative within their domain:

- JOSS: `paper/joss-readiness-manifest.json` and `scripts/validate_joss.py`;
- language registries: `docs/release/binding-submission-checklist.md` and the
  registry audit snapshot;
- arXiv: `paper/readiness-manifest.json` and the non-submitting manuscript
  assurance workflow;
- Conductor: `research_software_registry_readiness_20260721`.
