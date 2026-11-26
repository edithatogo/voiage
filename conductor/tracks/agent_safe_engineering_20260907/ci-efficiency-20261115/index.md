# Track: Reduce unnecessary GitHub Actions setup

Status: implementation merged in [PR #1332](https://github.com/edithatogo/voiage/pull/1332),
carrying the signed equivalent of [PR #1330](https://github.com/edithatogo/voiage/pull/1330).
The merged candidate passed the full local tox gate and required hosted checks.
The linked validation record preserves an earlier capacity-blocked attempt;
it is historical evidence, not the final merge result. Comparable hosted queue
and execution timings remain pending, so no end-to-end speedup is claimed.
Existing test, platform, coverage, security and source-identity contracts remain.

- [Specification](./spec.md)
- [Implementation plan](./plan.md)
- [Metadata](./metadata.json)
- [Measured baseline](./baseline.json)
- [Validation evidence](./validation.json)
- [Hosted timing comparison (2026-11-26)](./hosted-comparison-20261126.json)

This is a bounded follow-up to the completed G04 measurement work in
`agent_safe_engineering_20260907`; it does not change that historical evidence
or reopen the deferred HPC programme.
