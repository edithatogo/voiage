# VOIAGE requirements

This repository implements the production consumer side of the VOP–VOIAGE
programme. The canonical cross-repository requirements are maintained in
`vop_poc_nz/conductor/requirements.md`.

## MoSCoW priorities

### Must have

- Directional current-information EVoP and perspective methods retain versioned,
  deterministic, public contracts.
- The pinned VOP compatibility contract, Arrow schema fingerprint, IPC/Parquet
  fixtures, and PyArrow/Polars round trips validate in hosted CI.
- Every archived Conductor track remains discoverable in `conductor/tracks.md`
  and is represented in the cross-repository GitHub historical ledger.
- Python 3.14, current compatible dependencies, security checks, coverage,
  repository harnesses, and benchmark regression gates remain green.
- External maturity, data, hardware, registry, and publication gates remain
  explicit even when repository implementation is complete.
- Git-tag-derived dynamic versions, Pydantic v2 logging settings, structured
  run context, and uv/Pixi parity are enforced as production contracts.
- Ruff, `ty`, BasedPyright, package builds, unit/property/integration/E2E tests,
  security checks, and benchmark regression remain visible fast gates.

### Should have

- New interchange profiles reuse the shared compatibility schema and canonical
  logical-field fingerprint algorithm.
- Free-threaded Python remains a bounded observational lane until the required
  wheels are published.
- Pull requests and historical development eras remain represented in the
  VOP–VOIAGE GitHub Project.
- Scalene, mutation, dependency-audit, and experimental lanes emit bounded
  scheduled/manual evidence rather than slowing every pull request.

### Could have

- Cross-language consumers and accelerators validated by the same fixtures.
- Automated synchronization of archived tracks and project fields.
- Signed release attestations for promoted interchange bundles.

### Won't have now

- Automatic external publication or maturity promotion.
- Direct imports from the VOP source tree or repository consolidation.
- Production accelerator claims without parity and hardware evidence.
- Publication of credentials, private evidence, or local-only agent state.
