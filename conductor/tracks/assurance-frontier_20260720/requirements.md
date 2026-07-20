# C14 requirements: Assurance Frontier

## MoSCoW

### Must

- **C14-M01 — Deterministic contract:** VOP shall publish a versioned bundle
  whose manifest, schema inventory, Arrow identity and fixtures are reproducible;
  VOIAGE shall verify the pinned producer commit and aggregate SHA-256 without a
  VOP runtime dependency.
- **C14-M02 — Evolution safety:** incompatible dtype, unit, identity, version,
  nullability, order or provenance changes shall fail closed.
- **C14-M03 — Scientific assurance:** property, metamorphic, differential and
  independent analytical-reference tests shall record units, assumptions,
  provenance and tolerances.
- **C14-M04 — Performance and mutation:** CI shall ratchet mutation debt and
  deterministic CPU, memory, allocation and serialization budgets, retaining
  benchmark and Scalene evidence.
- **C14-M05 — Supply chain:** pull-request-safe CI shall produce reproducible
  distributions, CycloneDX SBOM and vulnerability evidence; release-only OIDC
  shall attest the exact publishable artifacts.
- **C14-M06 — Observability:** analysis boundaries shall correlate run, trace,
  span, backend, fallback and numerical-policy identifiers while recursively
  redacting secrets and protecting reserved fields.
- **C14-M07 — Governance safety:** drift auditing shall be read-only, distinguish
  unchecked from clean, emit durable evidence and preserve human approval gates.
- **C14-M08 — Verification:** all applicable local and hosted quality, security,
  build and contract gates shall pass at the exact implementation heads.

### Should

- **C14-S01 — Binding conformance:** each available language binding should
  validate the same independent fixture manifest; unavailable toolchains must be
  reported as explicit capability gates.
- **C14-S02 — Dependency frontier:** stable/frozen and isolated frontier lanes
  should emit promotion evidence without weakening the required stable lane.
- **C14-S03 — Provenance:** release artifacts should carry GitHub/Sigstore
  attestations when OIDC is available.

### Could

- **C14-C01 — Hardware baselines:** add runner-specific accelerator budgets when
  trustworthy hosted hardware is available.
- **C14-C02 — Bundle release:** publish the contract bundle as a standalone,
  signed release asset after human approval.

### Won't

- **C14-W01 — Autonomous governance mutation:** C14 will not merge, close issues,
  publish releases or modify GitHub Project truth without explicit approval.
- **C14-W02 — Runtime coupling:** VOIAGE will not import VOP production source or
  depend on a VOP runtime installation.

## Traceability

The implementation plan maps phases 1–6 to these IDs. Evidence belongs in git
notes, retained CI artifacts, issue #42 and GitHub Project #28; credentials and
private local artifacts are never committed.

