# Implementation plan: Domain Abstraction Excellence

## Phase 1 - Canonical contracts

- [x] Task: Add failing tests for concern, analysis and result schemas. `08fc460`
- [x] Task: Implement strict immutable Pydantic v2 contract models. `08fc460`
- [x] Task: Export deterministic JSON Schema and fixtures. `835dccf`
- [x] Task: Phase verification and checkpoint. Strict contract/schema and focused harness evidence verified 2026-07-20.

## Phase 2 - Kernel and backend capabilities

- [x] Task: Add failing generic protocol and capability-selection tests. `08fc460`
- [x] Task: Implement calculation-kernel, backend capability and numerical policy contracts. `08fc460`
- [x] Task: Adapt existing NumPy/JAX/backend selection without breaking APIs. `835dccf`
- [x] Task: Phase verification and checkpoint. Capability selection, empty-device rejection, and numerical parity verified 2026-07-20.

## Phase 3 - Method and result adoption

- [x] Task: Adapt perspective methods and ValueArray/ParameterSet boundaries. `835dccf`
- [x] Task: Add typed result-envelope and legacy result adapters. `835dccf`
- [x] Task: Verify numerical and serialization parity. `835dccf`
- [x] Task: Phase verification and checkpoint. Typed/legacy adapters and canonical Arrow/provenance interchange verified 2026-07-20.

## Phase 4 - Governance and automation

- [x] Task: Implement privacy-safe concern-ledger and GitHub synchronization payloads. `f8ee5a5`, VOP `55b3e5b`
- [x] Task: Update MoSCoW requirements, Mermaid design and architecture documentation. `d8d0d1c`, `f51d0e1`
- [x] Task: Extend CI/type/mutation/profile/contract gates. `f51d0e1`
- [x] Task: Phase verification and checkpoint. Governance mirror, MoSCoW/Mermaid, and CI contracts verified 2026-07-20.

## Phase 5 - Cross-repository closeout

- [x] Task: Validate pinned VOP contracts and fresh-process fixtures. `f8ee5a5`
- [x] Task: Run focused and full suites, build and dependency/security gates. Pull-request run `29691924913` passed all 13 executed jobs, including mutation.
- [x] Task: Complete formal review and hosted CI evidence. Principal findings remediated at `9f67c2f`; exact-head expensive run `29691957373` passed all 14 jobs.
- [x] Task: Phase verification and checkpoint. Completed 2026-07-20.

## Phase 6 - Principal review remediation

- [x] Task: Resolve C13 contract integrity, governance, and interchange findings.

## 2026-07-20 closeout evidence

- Exact VOIAGE head `9f67c2f` passed all 14 jobs in expensive run `29691957373`: unit and integration tests, coverage, E2E, Ruff/type checks, repository harness, version synchronization, documentation, profiling, mutation, benchmarks, frontier contracts, prose lint, and Python 3.14t observation.
- Pull-request run `29691924913` passed the same bounded suite including the new mandatory mutation gate; companion checks passed CodeQL, dependency review, workflow audit, zizmor, manifest smoke, performance benchmarks, and Rust profiling baselines.
- Broad production mutation is ratcheted from the truthful 51/65 (78.462%) hosted baseline by both score and unresolved debt; the production critical-invariant lane remains 100% (33/33 killed) against a 90% threshold.
- Shared Arrow metadata now carries the VOP–VOIAGE schema identity, version, metadata-independent fingerprint, producer, method-contract version, and IPC/Parquet format marker. Unsupported or nondeterministic JSON provenance values fail closed.
- Paired VOP head `ea418c7` includes retained security/profile/experimental evidence, an actual JAX experimental-backend probe, and its established dual mutation ratchets.
- Final code audits found no remaining Critical, High, or Medium findings. Human approval remains separate under the shared C13 `close_requires_approval` policy.
