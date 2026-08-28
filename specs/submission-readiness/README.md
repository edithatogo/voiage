# Submission readiness contracts

`targets.json` is the authoritative cross-venue inventory for package
registries, archives, identifiers, journals, community software review, and
sustainability affiliations.

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
- #299 maintains the preprint/JOSS/RRID package while preserving author and
  editorial decisions as external;
- #616 prepares the Python package for the selected pyOpenSci-first route while
  preserving inquiry and submission as separate maintainer actions;
- #615 matures the R package for possible rOpenSci and R Journal routes;
- #617 records whether a distinct JSS, NumFOCUS, or Zenodo outcome is justified;
- #622 prepares portable Spack/EasyBuild recipes before any HPC upstream or
  curation decision.

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
