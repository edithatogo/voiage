# Specification: Assurance Frontier

## Overview

C14 turns the C13 domain-contract foundation into continuously enforced,
cross-repository scientific and software assurance. VOIAGE consumes a pinned,
deterministic VOP–VOIAGE contract bundle without importing VOP source and
validates every available runtime against its contracts.

## MoSCoW requirements

### Must

- Verify one deterministic, versioned contract bundle containing schemas,
  Arrow identity, compatibility fixtures, migration policy and SHA-256 manifest.
- Verify forward/backward schema evolution and fail closed on incompatible
  dtype, unit, identity, version or provenance changes.
- Add property, metamorphic and differential tests across legacy/typed,
  NumPy/JAX and JSON/Arrow/Parquet boundaries where capabilities exist.
- Increase broad mutation assurance with score and unresolved-debt ratchets;
  retain at least 90% for critical-invariant lanes.
- Enforce deterministic CPU, memory, allocation and serialization performance
  budgets with durable Scalene/benchmark evidence.
- Emit SBOM, dependency/security, build-provenance and reproducibility evidence
  without requiring repository secrets on pull requests.
- Correlate structured logs with run, trace, span, backend, fallback and
  numerical-policy identifiers, with tested redaction.
- Run read-only GitHub governance drift detection and emit an approval-ready
  reconciliation artifact; never close or mutate records automatically.
- Validate calculations against independent analytical reference cases and
  record tolerances, units, assumptions and provenance.

### Should

- Validate every available language binding against the same fixture manifest.
- Maintain stable and frontier dependency lanes with explicit promotion rules.
- Produce release-ready Sigstore/SLSA attestation inputs when GitHub OIDC is
  available.

### Could

- Add hardware-specific performance baselines when suitable runners exist.
- Consume a standalone released contract asset after approval.

### Won't

- Merge pull requests, close governance issues, publish releases or use paid
  hardware without explicit human approval.
- Introduce a runtime dependency between VOP and VOIAGE.

## Acceptance criteria

- Bundle bytes and manifest hashes are reproducible in both repositories.
- Compatibility and migration tests cover the previous and current versions.
- Unsupported provenance values and incompatible schema changes fail closed.
- Mutation and performance policies fail CI on measured regression.
- Security, SBOM, provenance, observability and drift artifacts are retained.
- Independent review reports no remaining Critical, High or Medium findings.
- Exact implementation heads pass local focused gates and hosted CI.
