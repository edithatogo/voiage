# Scientific-review evidence contract v1

This directory defines the machine-readable evidence boundary for #318 Phase 5
and issue #842. The schemas cover the immutable review packet and artifact
manifest, reviewer identity and conflict attestations, role reports, findings,
disagreements, dispositions, independent adjudication and scientific approval,
the separate maintainer promotion receipt, and candidate-delta classification.

Schema validity is necessary but not sufficient. The semantic validator also
requires one eligible independent report for every required reviewer role,
matching attestations, consistent commit/tree/packet and capability scope,
independently verified Critical/High dispositions, resolved scientific dissent,
and two independent signatures for any bounded metadata-only delta. All other
deltas invalidate the candidate by default. A Boolean such as `approved: true`
is not an approval contract and is rejected.

Validate a bundle with:

```bash
uv run --extra ci python scripts/validate_scientific_review_evidence.py \
  specs/frontier/governance/scientific-review/v1/fixtures/valid-review-bundle.json
```

The bundled fixture is synthetic contract-test data. It is not a scientific,
promotion, release, publication, registry, or issue-closure approval.
