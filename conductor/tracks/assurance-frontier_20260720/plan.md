# Implementation plan: Assurance Frontier

## Phase 1 - Versioned shared contract bundle

- [x] Task: Add failing bundle reproducibility and consumer-verification tests.
- [x] Task: Generate schemas, Arrow identity, fixtures, migration policy and manifest in VOP.
- [x] Task: Pin and verify the bundle independently in VOIAGE.
- [x] Task: Phase verification and checkpoint.

## Phase 2 - Evolution, property and differential assurance

- [x] Task: Add failing schema-evolution and incompatible-change tests.
- [x] Task: Add Hypothesis/metamorphic typed-legacy and interchange tests.
- [x] Task: Add NumPy/JAX and available binding conformance checks.
- [x] Task: Phase verification and checkpoint.

## Phase 3 - Mutation and performance ratchets

- [x] Task: Raise broad mutation baselines with production-behaviour tests.
- [x] Task: Add deterministic CPU, memory and serialization budget contracts.
- [x] Task: Retain benchmark and Scalene evidence and enforce regressions in CI.
- [x] Task: Phase verification and checkpoint.

## Phase 4 - Supply chain and observability

- [x] Task: Add SBOM, dependency, reproducible-build and provenance evidence.
- [x] Task: Add trace/run/backend/fallback/policy logging correlation and redaction tests.
- [x] Task: Maintain stable/frontier dependency promotion evidence.
- [x] Task: Phase verification and checkpoint.

## Phase 5 - Governance drift and numerical references

- [x] Task: Consume read-only GitHub issue/Project drift artifacts.
- [x] Task: Add independent analytical numerical reference cases and tolerances.
- [x] Task: Verify private-data and human-approval safeguards.
- [x] Task: Phase verification and checkpoint.

## Phase 6 - Cross-repository closeout

- [x] Task: Update MoSCoW requirements, Mermaid design and developer documentation.
- [x] Task: Run full local and hosted quality/security/build suites.
- [x] Task: Complete independent principal review and remediate findings.
- [x] Task: Synchronize issue #42 and Project #28 while preserving human gates.
- [x] Task: Phase verification and checkpoint.

## Completion evidence

- VOIAGE implementation head `10f356f`: strict BasedPyright reported zero
  diagnostics, 93 focused bundle tests passed, and full branch coverage reached
  90.03% without exclusions or threshold changes.
- VOP implementation head `efcfec2`: CI, supply-chain, documentation, and
  expensive Quality Frontier passed; critical mutation reached 98.592%.
- Independent reviews found no remaining Critical, High, or Medium defect after
  remediation. Issue closure, merge, release, and publication remain human gates.
