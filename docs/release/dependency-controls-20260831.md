# Dependency and coverage controls, 31 August 2026

The repository inherits `github>edithatogo/renovate-config`. The owner preset was
inspected through GitHub's contents API. Repository overrides retain the
14-day release delay, supported Python and Cargo dependency ranges, optional
preview extras, explicit Node engine review, and pull-request review for all
executable dependency changes. Exact library environments remain in lockfiles.
Lint grouping matches only `ruff` and `ty`; runtime and stub packages cannot
enter through a substring match.

The required pre-change `uv lock --upgrade` and strict dependency-frontier check
passed. The frontier observed newer compatible hypothesis, joblib, linkify-it-py
and websocket-client releases. Those observations were not promoted in this
control repair: the root lock retains its reviewed bytes. Existing dependency
PRs require fresh validation and their own accepted changes.

Astro and Starlight tests require exact pins within the supported major/minor
families and lockfile agreement. The R test checks the optional bridge and its
minimum floor without freezing a particular compatible dependency release.
Historical reproduction bytes are preserved separately from the current lock;
see `paper/reproduction-environment/README.md` for the source-label limitation.

The authoritative full-coverage job grants only contents-read and OIDC token
permissions. Its pinned Codecov action uploads the generated `coverage.xml`,
disables unrelated report discovery, and fails the job on upload errors. No
stored Codecov token is used. Hosted upload success, repository activation, and
commit status must still be observed after delivery; local workflow validation
cannot establish those external outcomes.

Sources: [owner preset](https://github.com/edithatogo/renovate-config/blob/main/default.json),
[Renovate configuration](https://docs.renovatebot.com/configuration-options/),
[Codecov OIDC](https://github.com/codecov/codecov-action#using-oidc).

The review follow-up preserves the original paper manifest as an unchanged
archival receipt. The current manifest resolves both archived project and lock
paths and selects an exact Git source for an isolated frozen replay. The
verifier checks those declared paths against the selected source and binds a
successful result to the manifest and output hashes. This new replay does not
establish the original receipt's unverified `v2.0.0` source label.

Renovate excludes only the recorded paper and frontier environment directories
from dependency updates. Current project manifests and locks remain eligible
for their existing reviewed update policies.
