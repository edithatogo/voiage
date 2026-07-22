# v1.0 Release Checklist for voiage

This checklist is the operator-facing companion to
`.github/workflows/release.yml`. The workflow and executable tests remain
authoritative if this document drifts.

## 1. Freeze and reconcile

- [ ] Merge or explicitly defer every v1.0 issue and pull request.
- [ ] Reconcile remote branches, the Conductor registry, roadmap, `todo.md`,
  and `changelog.md`.
- [ ] Confirm Rust is the sole stable numerical execution authority.
- [ ] Confirm Python/PyO3, R, and Julia are thin retained bindings and Mojo is
  recorded as an external upstream boundary.
- [ ] Record external publication gates without presenting readiness as
  acceptance or indexing.

## 2. Version and source identity

- [ ] Set the authoritative Rust workspace version to `1.0.0`.
- [ ] Propagate that version to the exact internal Rust dependencies, R and
  Julia manifests, conda recipe, and generated lockfiles.
- [ ] Run the version synchronization validator and verify the dynamic Python
  metadata resolves to `1.0.0`.
- [ ] Confirm the source tree is clean and every release commit is
  GitHub-verifiable.

## 3. Complete release-candidate matrix

- [ ] Run `CI=true uv run tox -q`.
- [ ] In `docs/astro-site`, run
  `pnpm install --frozen-lockfile`, `pnpm run check`, and
  `pnpm run build` with the repository's explicit zero-delay release policy.
- [ ] Run the Rust workspace, MSRV, Clippy, formatting, tests, coverage,
  benchmarks, Miri, sanitizer, and dependency-policy gates.
- [ ] Run Python/PyO3, R, and Julia conformance, lifecycle, error, packaging,
  and clean-install tests against the Rust core.
- [ ] Record Mojo as externally gated unless an approved toolchain and binding
  contract are available.
- [ ] Build native Python artifacts with
  `maturin build --locked --release` and the canonical sdist workflow.
- [ ] Install each wheel and the sdist-derived wheel in clean environments and
  run the black-box VOI smoke analysis.
- [ ] Generate and inspect checksums, the CycloneDX SBOM, build provenance,
  artifact attestations, and signature evidence.
- [ ] Confirm there are no unresolved critical or high security findings.

## 4. Signed tag and private staging

- [ ] Create a signed annotated `v1.0.0` tag only after every required
  release-candidate gate is green.
- [ ] Verify GitHub reports the tag and its target commit as signed and valid.
- [ ] Push the exact tag and allow the release workflow to build and stage a
  private draft release.
- [ ] Inspect the staged wheels, sdist, manifests, checksums, SBOM, provenance,
  attestations, and platform coverage.
- [ ] Record the exact `expected_wheel_sha256` and
  `expected_sdist_sha256` values from the reviewed draft.

## 5. TestPyPI and PyPI publication

- [ ] Manually dispatch the release workflow with `publish=true`, the exact
  signed tag, and the reviewed artifact digests.
- [ ] Verify TestPyPI trusted publishing succeeds.
- [ ] Run the workflow's bounded TestPyPI registry-only installation and Rust
  execution smoke test.
- [ ] Permit PyPI trusted publishing only after the TestPyPI smoke gate passes.
- [ ] Verify the PyPI files, hashes, provenance, version metadata, and a clean
  registry-only installation.
- [ ] Publish the GitHub Release only after PyPI verification succeeds.

## 6. Retained binding and package registries

- [ ] Submit or update the conda-forge feedstock from the immutable v1.0 source
  artifact and verify its checksum.
- [ ] Submit the Julia package to Julia General using the immutable Julia tag.
- [ ] Submit the R source package to CRAN or the approved R registry and verify
  r-universe indexing where configured.
- [ ] Verify every accepted registry in clean registry-only environments.
- [ ] Keep external review, merge, acceptance, and indexing gates open with
  precise evidence until the external registry completes them.

## 7. Final verification and closeout

- [ ] Verify the Astro deployment and all public release links.
- [ ] Run representative Rust-backed EVPI, EVPPI, and EVSI analyses from the
  published Python package and retained bindings.
- [ ] Record immutable artifact URLs, checksums, SBOM, provenance,
  attestations, signatures, registry identifiers, and smoke results.
- [ ] Reconcile and archive completed Conductor phases.
- [ ] Keep genuinely external publication tracks active or machine-readably
  gated; do not archive them as accepted before authoritative evidence exists.
- [ ] Close the v1.0 programme only when every acceptance criterion is proven.

## Rollback

If a published artifact is unsafe or materially incorrect:

1. Stop further registry publication.
2. Preserve the signed release and evidence; do not rewrite the tag.
3. Yank the affected Python version where appropriate and document the reason.
4. Open a security or release-blocker issue.
5. Prepare a new signed patch release through this complete workflow.
