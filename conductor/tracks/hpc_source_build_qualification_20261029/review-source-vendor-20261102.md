# Review Report: Linux source/vendor qualification

## Summary
The bounded source/vendor checkpoint is implemented with explicit provenance,
negative controls and offline replay. Native build acceptance stays open.

## Verification Checks
- [x] **Plan Compliance**: Partial - implements the Cargo/compiler-source subset
  of P2, while P3-P6 and full package-manager install acceptance stay open.
- [x] **Style Compliance**: Pass - focused Ruff checks and formatting passed.
- [x] **New Tests**: Yes - corruption, shadowing, absent offline input, traversal,
  vendor tampering/inventory, lock repair and checksum-order controls.
- [x] **Test Coverage**: Partial - focused helper controls plus hosted real-input
  cells. This does not claim complete branch coverage of the qualification tools.
- [x] **Test Results**: Passed - 12 focused controls on actual Python 3.15.1.
  The local full tox attempt stopped before execution at the disk-space guard.
  The receipt records the tested head, dirty input hash and observed free space.
  Hosted checks are retained separately and must pass before merge.

## Findings and applied fixes

- The Polars sdist lock included entries absent from its shipped workspace.
  Froze and authenticated a pruning patch without introducing package identities,
  versions or registry checksums. All-feature replay remains locked. Both current
  package managers consume the same patch, with upstream MIT terms preserved.
- EasyBuild's crate checksum list must match source fetch order. Inserted the
  three actually measured Git archive hashes at their exact source positions and
  added a regression covering all six current Cargo-backed provider recipes.
- The fetch-only EasyBuild wrapper needed the explicitly selected Python on
  PATH and Lmod. Qualified the real supported CLI without a compiler build.
- Vendor source-identity collisions now fail closed and receipts retain the
  complete registry/Git source identity beside each vendor checksum.
- Added the resource guard to workflow trigger paths and cancel superseded runs.
  Actions retain first failures and terminal resource receipts on unsuccessful
  qualification as well as successes. No security, numerical or coverage gate
  was weakened.

## Review disposition

Reviewed the workflow, both helpers, negative controls, frozen lock diff and
provider changes against the specification and current project rules. No
unresolved correctness or security finding remains in this bounded change.
The source/vendor and graph receipt manifests are the terminal evidence.
No independent reviewer approval is required by the solo-maintainer policy.
