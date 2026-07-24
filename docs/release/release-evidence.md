# Release evidence

Release evidence is bound to an exact version, Git tag, source commit, published
asset set, provenance statement, CycloneDX software bill of materials (SBOM),
and Software Heritage snapshot.

For `v1.0.0`, the checked-in evidence is:

- [`v1.0.0-release-evidence.json`](v1.0.0-release-evidence.json), the canonical
  machine-readable evidence record;
- [`v1.0.0-sbom.cdx.json`](v1.0.0-sbom.cdx.json), the reproducible CycloneDX
  Python runtime dependency inventory generated from the tagged lockfile;
- GitHub release assets and their `SHA256SUMS` file;
- SLSA provenance verified against the published package artifact digests; and
- Software Heritage snapshot
  `swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32`.

Validate the checked-in record and local tag with:

```console
uv run python scripts/release_evidence_manifest.py validate \
  docs/release/v1.0.0-release-evidence.json \
  --artifact-root . \
  --repository-root . \
  --allow-missing-release-assets
```

The `--allow-missing-release-assets` option applies only to the checked-in
record because the binary release assets remain on GitHub. The supply-chain
workflow downloads those files and validates them byte-for-byte without that
option.

## Mixed-language SBOM scope

The supply-chain workflow now composes one CycloneDX 1.6 release SBOM across
the distributed Python package and the Rust, R and Julia sources:

- **Python** is a resolved inventory of the installed release wheel and its
  non-development runtime environment, generated from the frozen `uv` export.
- **Rust** is the complete dependency graph in `rust/Cargo.lock`, including
  every workspace crate and the packages selected for normal, build,
  development and test use.
- **R** records the `voiageR` binding and every package declared by
  `Depends`, `Imports`, `LinkingTo`, `Suggests` or `Enhances` in
  `DESCRIPTION`. Because this repository has no R lockfile, dependency version
  requirements are retained as constraints and are explicitly labelled
  `declared-unresolved`.
- **Julia** records the `Voiage` binding and every package or standard-library
  entry declared in `[deps]` and `[extras]` in `Project.toml`. With no committed
  `Manifest.toml`, UUIDs and compatibility requirements are retained and these
  dependencies are likewise labelled `declared-unresolved`.

The R and Julia binding components point to the locked `voiage-ffi` Cargo
component that supplies their native interface. Top-level SBOM properties name
all four ecosystems and distinguish resolved inventories from declaration-only
ones. This makes the coverage boundary machine-readable instead of implying
that uncommitted R or Julia resolutions were reproduced.

`scripts/compose_polyglot_sbom.py` canonicalises component and dependency
ordering, rejects duplicate or dangling references, checks all four ecosystem
inventories, binds the source commit/tag/version and rejects release-time
binding version drift. CI then validates the same output against the official
CycloneDX 1.6 schema using a checksum-pinned CycloneDX CLI binary. The workflow
retrieves the release asset metadata from the official
`CycloneDX/cyclonedx-cli` repository, verifies the expected release tag, asset
name, download URL and GitHub-published digest, verifies the downloaded bytes
again locally, and retains that source-verification record.

The checked-in `v1.0.0-sbom.cdx.json` predates this composition path and remains
an honestly scoped Python runtime inventory. It is not retroactively described
as mixed-language evidence. The workflow artifact for subsequent revisions and
the SBOM retained with the next release use the mixed-language contract above.

## Hosted workflow

The supply-chain workflow runs for pull requests, release tags, published
releases, its weekly schedule, and manual dispatch. A manual release audit
accepts an existing tag and the Software Heritage snapshot containing it. The
workflow:

1. resolves the tag to its immutable commit;
2. builds reproducible distributions from that commit;
3. generates a resolved Python runtime inventory;
4. composes the Cargo lock graph and declared R/Julia binding dependencies into
   a revision-labelled mixed-language CycloneDX SBOM;
5. validates its scope, dependency graph and official CycloneDX 1.6 schema;
6. downloads the published release and verifies `SHA256SUMS`;
7. selects SLSA provenance whose subjects and source commit match the
   published package files;
8. creates and revalidates the release-evidence manifest; and
9. retains the source Python inventory, composed SBOM, evidence files and a
   checksum inventory as a GitHub Actions
   artifact for 90 days.

SBOM generation is independent of Software Heritage timing. A tag push always
builds and validates the mixed-language SBOM without requiring a snapshot
SWHID or an already-published GitHub Release. A published-release event may
audit the release assets, but records the evidence state as
`partial_missing_software_heritage_snapshot` when archival has not completed.
Manual dispatch with the release tag and snapshot SWHID performs the complete
release-evidence creation and validation. The retained
`release-evidence-status.json` distinguishes these states explicitly.

GitHub's `targetCommitish` release field is retained as descriptive metadata;
it can legally contain a branch name such as `main`. Source identity instead
comes from dereferencing the release tag to a commit and requiring that commit
to equal the checked-out revision. The workflow does not compare the
`targetCommitish` JSON string directly with a commit hash.

The workflow does not upload or attach files to a GitHub release. In
particular, the `v1.0.0` SBOM is recorded and reproducible but is not currently
attached to the public `v1.0.0` release. Attaching that file, or publishing a
later patch release whose release payload includes it, remains a separate
maintainer action.
