# Track Specification: Python Release Submission

## Overview

This track closes the Python release submission path. It keeps the Python
package aligned with the documented tag-driven publish flow and the binding
submission checklist so the repo can distinguish automated publishing from the
external conda-forge feedstock step.

## Functional Requirements

1. Keep the Python package version, tag prefix, and release workflow aligned.
2. Preserve the PyPI/TestPyPI publish automation already defined in the repo.
3. Keep the conda-forge recipe update flow explicit about the external feedstock
   merge.
4. Keep the release checklist and release docs honest about what is automated
   here versus what still depends on external registry-side action.

## Non-Functional Requirements

1. Preserve the stable Python façade over the Rust core.
2. Avoid runtime behavior changes in this track.
3. Keep the release path reproducible and explicit.

## Acceptance Criteria

1. The release workflow, versioning policy, and checklist agree on the Python
   submission path.
2. The docs make clear that conda-forge still needs external feedstock action.
3. The repo can describe the Python release as automated here, with the
   registry-side boundary still explicit.

## Out of Scope

1. Changing Python runtime semantics.
2. Replacing the existing publishing automation.
3. Claiming live registry state without external verification.
