# Engineering closeout checkpoint — 22 November 2026

## B8 repository verification

The signed head `2c1e30a92a67bd9f9b891a5f210f3f439109a3c2` passed all
15 local tox environments on Python 3.15.2, with 95.00% aggregate coverage
and all 17 required hosted checks passing. Its tree
`fa6454e2b4feb6d466bfcb4c1c6bb58b8850d138` matches the squash merge
`7d44e3d9b5ff86506d5a2edb4ba99a6a5b2836af`. The merged revision's CI and
operational assurance runs also succeeded. [PR #1386](https://github.com/edithatogo/voiage/pull/1386)
· [Main CI](https://github.com/edithatogo/voiage/actions/runs/44912976276)
· [Operational Assurance](https://github.com/edithatogo/voiage/actions/runs/44912976305)

The retained binding run passed Rust and MSRV checks, installed R smokes,
the Julia source matrix, and cross-language differential conformance.
The workflow's watched inputs are byte-identical between the tested
`41f825a14c9ffd27610f5567bf13b44b1b738d07` and `7d44e3d9`.
[Binding matrix](https://github.com/edithatogo/voiage/actions/runs/44907189007)

B8's installed Spack and EasyBuild acceptance remains open under the HPC
deferral. The earlier Mac tox result at `3fdac2faa` is historical evidence
for that revision.

## B12 recovery and B13 inventory

An independent mirror restore matched all 239 captured refs and all 11 ordered
stash entries, including the 11 notes refs in the ref inventory. The restored
repository passed `git fsck --full --strict`. The bundle, restore and full
local inventory are retained privately on the external SSD.

Four local branches were retired with exact commit leases. Each tip was
recoverable from the bundle and had the same tree as its merged PR. None was
checked out or referenced by an open PR. Their remote heads were already
absent. The 18 registered worktrees were retained for their current owners.

The [B13 inventory](b13-inventory-20261122.json) records the exact PR heads,
open issue numbers, branch counts and recovery digest after retirement. GitHub
and Git transport returned identical remote branch names and tips. The final
inventory is due after the remaining authorized delivery and cleanup.

## B14 Conductor validation

The current registry normalization and GitHub cross-reference checks passed.
The September pre-closeout validator rejected its historical snapshot against
the current local repository. The live custom ref inventory has 22 additional
names and one changed tip. The dated checkpoint and its required native Spack
receipt remain in place. Final archival awaits the outstanding acceptance and
the post-closeout inventory.

## G03.4 Julia registry gate

Yggdrasil PR #14292 remains open with a green aggregate status at
`e479eeea400e65f0e2b78f4fb4a3fc9ed99200fa`. The generated wrapper repository
lookup returned HTTP 404, and the General registry had no Voiage entry at
`43b2e6fdfbfaaba589f32a0c200879137c05538f`. The passing source-binding checks
remain valid. An accepted JLL, clean-depot tests using it, and General
registration remain necessary for G03.4. [Yggdrasil PR #14292](https://github.com/JuliaPackaging/Yggdrasil/pull/14292)
· [General registry snapshot](https://github.com/JuliaRegistries/General/blob/43b2e6fdfbfaaba589f32a0c200879137c05538f/Registry.toml)
· [Voiage #1023](https://github.com/edithatogo/voiage/issues/1023)

## 23 November repository verification and recovery

The signed tree `3f2e7df7b3d37c5bf3b20c19dec5c993d69de73c` passed all 15 tox
environments and 17 required hosted checks. Its squash merge
`69a60cbf678d62f0bfb308059fc8408d35017331` has the same tree. GitHub Pages
deployed from the completed build artifact on the first attempt.
[PR #1398](https://github.com/edithatogo/voiage/pull/1398)
· [Full tox](https://github.com/edithatogo/voiage/actions/runs/45079435732)
· [Pages deployment](https://github.com/edithatogo/voiage/actions/runs/45080827061)

The three local branch tips for PRs #1396, #1397 and #1398 were recovered
in an independent Git restore before retirement. Each tree matched its
signed merge. The SHA-256 of the private bundle is
`6df60d59e405b1bede17583f688589932c3f67df27869c2121f929973e9bdfa6`.
All three remote heads were already absent. The current worktrees were
retained.
