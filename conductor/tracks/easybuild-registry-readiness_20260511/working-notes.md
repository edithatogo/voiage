# Working Notes: EasyBuild Registry Readiness

## Status

- EasyBuild handoff is scoped to deterministic inputs, build flags, and maintainer
  acceptance criteria.
- This repository does not currently create or merge EasyBuild easyconfig PRs.

## Handoff Notes

- EasyBuild inputs are the repository source archives, deterministic build flags,
  and dependency constraints already described in the release docs.
- Final acceptance remains external to this repository.

## Required Inputs

- GitHub Release source artifacts and checksums
- Build toolchain constraints
- Python binding behavior matrix (CPU-first contract)
- Dependency constraints from `pyproject.toml` and lock-sensitive releases

## External blocker

External EasyBuild repository approval is required to close this track.
