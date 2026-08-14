# Scientific-review evidence contract v1.1

This directory defines the machine-readable evidence boundary for #318 Phase 5
and issue #842. The schemas cover the immutable review packet and artifact
manifest, reviewer identity and conflict attestations, role reports, findings,
disagreements, dispositions, independent adjudication and scientific approval,
the separate maintainer promotion receipt, and candidate-delta classification.

Schema validity is necessary but not sufficient. Version 1.1 distinguishes
typed Git object identifiers from SHA-256 content digests and recomputes every
declared canonical-JSON digest. For non-synthetic evidence, the validator also
resolves the declared commit and tree and hashes each manifest artifact directly
from the frozen Git tree.

The semantic validator requires complete finding and disagreement inventories,
one eligible independent report for every required role, exact reviewer/scope
attestation matches, identity-bound human receipts for the chair, approver and
maintainer, separation of
authors, remediators, orchestrator, reviewers, chair, and approver, and current,
unsuperseded decisions. Medium findings require affected-role re-review;
Critical/High findings require independently verified resolution. Reviewed
exclusions bind exact capability removals. A bounded delta is permitted only for
field-level administrative changes with two eligible, distinct governance and
scientific signers. Every other delta fully invalidates the candidate. A Boolean
such as `approved: true` is rejected.

Validate a bundle with:

```bash
uv run --extra ci python scripts/validate_scientific_review_evidence.py \
  path/to/review-bundle.json --repository-root .
```

The bundled fixture may be validated without `--repository-root` because it is
explicitly marked synthetic contract-test data. Its external-receipt strings are
non-authoritative examples. It is not a scientific, promotion, release,
publication, registry, or issue-closure approval.
