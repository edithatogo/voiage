# Working Notes: HPC Registry Readiness

## Scope

This phase is documentation-driven and external-boundary-aware:

- In-repo release automation is complete for language artifacts.
- HPC registry pathways are defined but not repository-owned, so this track
  documents what is ready now and what depends on registry maintainers.

## Immediate target matrix

- **Spack**: recipe expectations, version/source, checksum, and dependency pinning
  requirements for maintainers.
- **EasyBuild**: easyconfig expectations, module generation model, and reproducible
  build flags.
- **HPSF**: metadata and visibility expectations for integration with the
  software library ecosystem.
- **E4S**: alignment expectations for curation and HPC-package visibility.

## Readiness Matrix

| Target | Required artifact | External action | Evidence |
| --- | --- | --- | --- |
| Spack | Source tarball, checksum, package metadata | Maintainer PR/review/merge | `docs/release/binding-submission-checklist.md` |
| EasyBuild | Easyconfig, build flags, module expectations | Maintainer review/merge | `docs/developer_guide/hpc_distribution_contract.rst` |
| HPSF | Package identity, provenance, maturity metadata | Curation review | `docs/release/binding-submission-checklist.md` |
| E4S | Curation packet and dependency disclosure | Inclusion review | `docs/developer_guide/hpc_native_roadmap.rst` |

## Evidence to capture

- `docs/developer_guide/hpc_distribution_contract.rst`
- `docs/developer_guide/hpc_native_roadmap.rst`
- `docs/release/binding-submission-checklist.md`
- this track’s handoff notes and status matrix

## External dependency boundary (current)

- Spack/EasyBuild recipes are external ecosystem processes and are not created or
  merged by this repository in this track.
- HPSF/E4S entries are visibility/curation channels and require external
  maintainer acceptance.

## Current decision

- Keep this track as a readiness/contract track until the project owners can route
  real recipe and curation submissions into maintainers with minimal ambiguity.
- The documentation and handoff expectations are now explicit; the remaining
  external acceptance steps are outside repository control.
