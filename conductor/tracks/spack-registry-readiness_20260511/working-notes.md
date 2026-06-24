# Working Notes: Spack Registry Readiness

## Status

- Scope completed to documentation and handoff; no Spack recipe file is written in
  this repository in this track.
- External maintainer workflow remains the gate.

## Handoff Notes

- Recipe-ready artifacts are the released source tarball, checksum, and dependency
  bounds already captured in the repository release docs.
- The repository does not own Spack PR submission or merge approval.

## Expected Recipe Inputs

- Release tag and source tarball URL
- Deterministic checksum and build args
- Python dependency bounds matching the repository canonical constraints
- Runtime behavior expectations (CPU-first reference contract)

## External maintainer gate

Spack submission PR and approval still happen in the Spack ecosystem, outside
this completed readiness track.
