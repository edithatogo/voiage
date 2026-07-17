# Binding Submission Checklist

This checklist records what the repository can submit automatically and what
still depends on an external registry or feedstock process.

It is intentionally conservative:

- "Automated here" means a tag or release push can carry the package to the
  registry from this repository, subject to credentials being present.
- "Tag-driven indexing" means a tag or release push creates the release
  artifacts or module state that the registry ingests, but the repo does not
  perform a direct registry upload itself.
- "External/manual" means the repository can prepare artifacts, but the final
  registry-side submission or approval still happens outside this repo.
- "Submitted" in this file means "ready to submit through the documented path";
  it does not assert that a live registry already contains the package.

## Status Summary

## Conductor publication-program evidence

The active Conductor publication-program handoff records the current status of
all 13 registry channels in
`conductor/archive/external-registry-publication-program_20260625/handoff/registry-manifest.json`.
It distinguishes repository readiness from `submitted`, `published`, `indexed`,
and `approved` states; unresolved channels retain explicit owner, next action,
evidence URL, and external-gate fields. No external registry submission or
maintainer approval is inferred from an in-repo workflow or release tag.

| Language | Submission path from this repo | External dependency | Live registry state verifiable here? |
| --- | --- | --- | --- |
| Python | Automated PyPI/TestPyPI publish, tag-driven release, conda-forge update PR | conda-forge feedstock merge | No |
| R | GitHub Release source archives | CRAN and r-universe | No |
| Julia | TagBot sync plus GitHub Release artifacts | Julia General registry approval | No |
| TypeScript | Automated npm publish with provenance | None beyond credentials | No |
| Go | Tag-driven module publication plus GitHub Release artifacts | Go module proxy indexing | No |
| Rust | Automated `cargo publish` | None beyond credentials | No |
| .NET | Automated NuGet publish | None beyond credentials | No |
| Spack | Manual recipe preparation for Spack repository | Spack maintainer review and PR merge | No |
| EasyBuild | Manual easyconfig preparation for EasyBuild repository | EasyBuild maintainer review and PR merge | No |
| HPSF | Manual curation submission | External curation review / listing policy | No |
| E4S | Manual curation submission | External curation review / inclusion policy | No |

## Python

- [x] Package build and test gates exist in CI.
- [x] PyPI publication is automated on `v*` tags through trusted publishing.
- [x] TestPyPI publication is automated on `v*` tags.
- [x] The in-repo conda-forge update workflow creates a feedstock PR.
- [x] The Python façade remains the stable release surface.
- [ ] External conda-forge feedstock merge remains manual/outside this repo.

## R

- [x] Package build and check gates exist in CI.
- [x] GitHub Release source archives are produced from `r-v*` tags.
- [x] The package docs/manual flow is versioned in-repo.
- [x] The R package remains the thin reticulate bridge over the shared contract.
- [ ] CRAN submission remains external/manual.
- [ ] r-universe indexing remains external/manual.

## Julia

- [x] Package test gates exist in CI.
- [x] GitHub Release source archives are produced from `julia-v*` tags.
- [x] TagBot synchronization is configured.
- [x] The Julia binding remains the thin adapter over the shared contract.
- [ ] Julia General registry submission/approval remains external/manual.

## TypeScript

- [x] Package lint/test and pack dry-run gates exist in CI.
- [x] npm publication is automated on `typescript-v*` tags with provenance.
- [x] Release assets are attached to GitHub Releases.
- [x] The TypeScript binding remains the thin adapter over the shared contract.

## Go

- [x] Package test and vet gates exist in CI.
- [x] GitHub Release source archives are produced from `bindings/go/v*` tags.
- [x] Module publication is driven by semver tag pushes and downstream module proxy indexing.
- [x] The Go binding remains the thin adapter over the shared contract.

## Rust

- [x] `cargo fmt`, `cargo clippy`, `cargo test`, `cargo doc`, and `cargo package` gates exist in CI.
- [x] crates.io publication is automated on `rust-v*` tags when credentials are present.
- [x] GitHub Release source archives are attached to the release.
- [x] The Rust crate remains the canonical execution core and contract owner.

## .NET

- [x] Build, test, and pack gates exist in CI.
- [x] NuGet publication is automated on `dotnet-v*` tags when credentials are present.
- [x] GitHub Release source archives and nupkg artifacts are attached to the release.
- [x] The .NET binding remains the thin adapter over the shared contract.

## Practical Answer

If the question is "are all language versions submitted to their corresponding
registries today?", the repo can only answer this partially:

- The in-repo publishing workflows are in place for Python, TypeScript, Rust,
  and .NET.
- Go is tag-driven and release-artifact driven, with submission realized by
  the registry's module-proxy indexing flow.
- Julia, conda-forge, CRAN, and r-universe still require external registry-side
  action or approval.
- Spack, EasyBuild, HPSF, and E4S are all explicit external/manual paths with
  no live confirmation from this repository.

The repository is now explicit about that distinction, but live registry state
still has to be checked in each target ecosystem.

To keep this process synchronized with the Conductor roadmap, the current
snapshot is also stored in:

- `docs/release/registry_audit_snapshot.json`

## Follow-Through Publication Tracks

The completed readiness tracks are not reopened for live publication evidence.
The active follow-through tracks are:

- `external-registry-publication-program_20260625`
- `archive/conda-forge-feedstock-publication_20260625`
- `r-cran-runiverse-publication_20260625` (handoff: `conductor/archive/r-cran-runiverse-publication_20260625/handoff/r-registry-evidence.json`)
- `julia-general-registry-publication_20260625` (handoff: `conductor/archive/julia-general-registry-publication_20260625/handoff/julia-registry-evidence.json`)
- `spack-package-merge-followthrough_20260625`
- `easybuild-easyconfig-merge-followthrough_20260625`
- `hpsf-curation-submission-followthrough_20260625`
- `e4s-inclusion-followthrough_20260625`

These tracks must distinguish readiness, submitted, published, indexed,
approved, blocked, and not-found states. GitHub Actions and `gh` are the
preferred repeatable evidence path; browser automation is reserved for external
portals and must pause before irreversible submissions or account actions.

## How To Refresh Live Registry Evidence

The live registry evidence packet is a static JSON artifact that can be refreshed
on demand with the refresh utility:

```bash
python scripts/refresh_binding_registry_audit.py
```

The utility writes the latest channel checks into
`docs/release/registry_audit_snapshot.json` and preserves a history of each
channel-level `checked_at` timestamp and confidence level in the snapshot.

Use offline mode when network checks are not available:

```bash
python scripts/refresh_binding_registry_audit.py --offline
```

Offline mode intentionally does not infer publication state; it rewrites entries
with `not_checked` so the gap stays explicit.

## Live Registry Audit

The public registry checks performed on 2026-05-10 did not find a published
package for the binding names used in this repository:

- Python `voiage` on PyPI returned `404`
  - https://pypi.org/project/voiage/
- TypeScript `@voiage/core` on npm returned `404`
  - https://www.npmjs.com/package/%40voiage%2Fcore
- Rust `voiage-core` on crates.io returned `404`
  - https://crates.io/crates/voiage-core
- .NET `Voiage.Core` on NuGet returned `404`
  - https://www.nuget.org/packages/Voiage.Core
- R `voiageR` on CRAN returned `404`
  - https://cran.r-project.org/web/packages/voiageR/index.html
- Go `github.com/edithatogo/voiage/bindings/go` was reachable on the module
  proxy but reported no released versions
  - https://proxy.golang.org/github.com/edithatogo/voiage/bindings/go/@v/list
- Julia `Voiage` was not present in the General registry contents API
  - https://github.com/JuliaRegistries/General
- conda-forge `voiage` was not present in the anaconda.org package API
  - https://anaconda.org/conda-forge/voiage
- r-universe `voiageR` was not present in the package API
  - https://edithatogo.r-universe.dev/voiageR
- Spack `py-voiage` was not present in the upstream package tree
  - https://packages.spack.io/package.html?name=py-voiage
- EasyBuild `voiage` was not present in the upstream easyconfig tree
  - https://github.com/easybuilders/easybuild-easyconfigs
- HPSF and E4S remain external/manual curation targets because this repository
  cannot confirm inclusion from a package API alone.

So the current live state is: the release automation exists, but the registry
submissions have not been confirmed as published for these package names.
