# Backlog verification and inventory, 14 November 2026

## B8 repository verification

The local tox receipt for the installed-adapter PR covers all 15 environments
on Python 3.15.1, with 5,860 passes, 29 skips and 94.83% coverage. Its tested
tree `4442bfb8894fde8ee867307cb8f049b23c35e7f6` matches main after PR #1301.
All 17 required hosted checks passed on the signed PR head. The refreshed
repository, recovery and Conductor subset passed 106 tests. The last
[full binding matrix](https://github.com/edithatogo/voiage/actions/runs/42215182738)
passed its Rust and MSRV lanes, R development and installed tests, Julia source
matrix, and differential conformance. Its Rust, binding, fixture and workflow
inputs match the observed main revision.

Renovate tried to build copied paper manifests that depend on the original
source checkout. Its PEP 621 file filter now screens out the two frozen
directories before extracting dependencies. The installed Renovate matcher
excluded three archived manifests and retained the live root project. Strict
validation and the targeted regression passed. Affected security PRs still
need a hosted Renovate refresh and their own checks.
[Renovate file filter](https://docs.renovatebot.com/configuration-options/#ignorepaths).

The two live readability environments now pin NLTK 3.10.4 and provide the
exact packages required by `uv pip sync`. The installed JOSS and arXiv
audits gave byte-identical reports on the same 557-word, 40-sentence fixture.
The [repair receipt](nltk-security-update-20261114.json) records the advisory
and package evidence. Earlier evidence retains its original inputs.

PR #1303 merged as `faa356283` with all 17 required checks passing.
The full local gate on its exact tree passed all 15 tox environments on
Python 3.15.1. GitHub recorded all eight NLTK alerts as fixed. The hosted
Renovate app rejected the new dashboard option syntax after the merge.
The compatibility correction restores the dashboard option supported by
the hosted version and retains the paper manifest filter. See
[Renovate's configuration report](https://github.com/edithatogo/voiage/issues/1305).

## B13 live inventory and B12 recovery

The 00:51 UTC snapshot is bound to
[`9f53fcb33bc4e40c5a84f3b3fc31f3287f1f1a3b`](https://github.com/edithatogo/voiage/commit/9f53fcb33bc4e40c5a84f3b3fc31f3287f1f1a3b),
after the installed-adapter merge. The [inventory](b13-inventory-20261114.json)
records 12 open PRs, 30 issues, 36 local branches, 22 remote branches and five
worktrees. GitHub and Git transport agreed on the remote branch tips. The
private capture retains exact refs, worktree status and the ordered stash
reflog.

An independent, complete-history bundle restore matched 237 source refs,
all 11 stash entries and 11 notes refs, and passed strict full `git fsck`.
Bundle SHA-256
`47c6f93423f8b972916769fc5bb7073b4a60b81de84ca24044ff62bb37ec3368`.
The recovery copy remains on the external SSD. Two unchecked-out local tips
from PRs #1299 and #1300 had exact tree matches to their squash merges. Their
remote heads were already absent. The local tips were retired in one transaction
that checked both expected commit IDs and verified their absence.

After the snapshot, seven issue records with satisfied acceptance criteria were
closed with links to their merged work. These cover execution packets, version
identities, Rust logging, logging privacy, the scoped VOP pilot, the Polars
containers and the completed tabular migration. The installed-adapter worktree
from #1301 and its local branch were also retired. The restored checkout matched
70,931 filesystem entries and all three restored submodule Git stores passed
strict verification. The recovery archive is on the external SSD with SHA-256
`51be2279d334e672722a31608518438e70df0a759bf1d9c5f7223a2e9afda189`.

## Remaining acceptance

- B8's installed Spack and both EasyBuild generations depend on the deferred
  HPC work.
- B12 and B13 still require disposition of remaining unique or owner-held
  branches and an inventory after final delivery and cleanup. B14 stays open.
- The dependency queue includes updates above the Rust 1.85 MSRV, workflow
  proposals with interpreter and release-control failures, and five security
  PRs with bot artifact failures. Each proposal has its own required checks.
- Julia registry acceptance, maintainer venue text and disclosures,
  scientific review and identifier outcomes remain in their owning issues.
