# Round 2 reproducibility and packaging review

Recommendation: **major revision before submission**

Score: **891/1000**

This is an internal simulated JOSS review, not an editorial decision. I reviewed
the working-tree snapshot available on 24 July 2026 against the panel rubric and
the current JOSS review criteria. The inspected Git `HEAD` was
`1dfcd3d42af591ca15fcc03ad958123cc153dbbf`, equal to `origin/main`, but the
working tree contained material tracked modifications and untracked release,
test, Rust, R, and review files. The result below therefore applies to that
inspected file state, not to an immutable commit.

## Decision

Round 1's R runtime and release-evidence defects have been substantially
improved. Both published Python distribution forms install cleanly, the repaired
R binding and the Julia binding execute the shared native EVPI calculation, the
release manifest validates, provenance verifies, and the Software Heritage
snapshot contains the signed `v1.0.0` tag.

The submission is not yet reproducibly frozen. The manuscript describes
scientific and binding corrections that are newer than `v1.0.0`, while its
availability section identifies only that release and says that the exact
reviewed revision will be frozen later. The current JOSS manuscript and the new
release-evidence files are not represented by one immutable revision, and the
successful hosted Open Journals build applies to clean `main`, not to the
inspected modified manuscript. This is a material revision/release ambiguity
under the fail-closed rubric and caps the score below 950.

## Reproduction record

All installation probes were run outside the repository in temporary
directories.

| Probe | Result |
| --- | --- |
| `python -m pip install voiage==1.0.0` in a clean Python 3.13 environment | Pass. The macOS ABI3 wheel installed, `voiage.__version__` reported `1.0.0`, `pip check` passed, and the reviewer-protocol calculation returned `EVPI: 0.667`. |
| Install the published `voiage-1.0.0.tar.gz` in a second clean Python 3.13 environment | Pass. The sdist built a wheel, installed as version `1.0.0`, imported outside the checkout, and passed `pip check`. The sdist contains the generated `rust/crates/voiage-python/source-provenance.txt` required by the release build. |
| Build the inspected current source after copying it without `.git` | Fail. `maturin` stopped with `release builds require a valid Git identity or complete VOIAGE source identity variables`. The release workflow prepares the sdist correctly, but an ordinary GitHub/Software Heritage source export does not carry the generated provenance file and the user-facing source-build path does not explain the three required variables. |
| Build `voiage-ffi`, build and install the current R source package, then call installed `voiageR::evpi()` | Pass with the separately built library supplied through `VOIAGE_FFI_LIBRARY`; the non-zero reference returned `0.667`. The installed package now resolves the symbol from the exact loaded library handle. |
| Instantiate and test the current Julia source package, then call `Voiage.evpi()` | Pass with the same separately built library supplied through `VOIAGE_FFI_LIBRARY`; 12 binding/reference assertions passed and the non-zero reference returned `0.667`. |
| `python scripts/validate_joss.py` | Pass; the body contains 1,001 words. |
| Release-evidence manifest validation | Pass with `--allow-missing-release-assets`; the local SBOM digest matches the manifest. |
| GitHub attestation verification for the published sdist | Pass. An attestation at source commit `05cc373d78ae74143194e889ff1317de4dfea52e` contains the exact public sdist and wheel subjects recorded by the manifest. |
| Software Heritage resolution | Pass. Snapshot `swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32` resolves; `refs/tags/v1.0.0` points to the archived signed release object and then to commit `05cc373d78ae74143194e889ff1317de4dfea52e`. Its archived `main` is an older revision and it does not preserve the inspected JOSS working tree. |
| Official Open Journals build | Pass for hosted run `30062546647` at clean commit `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`. Not verified for the inspected modified `paper.md`; no exact-snapshot PDF can therefore be credited in this round. |

## Release, provenance, SBOM, and archive assessment

### Release and revision identity

The signed annotated `v1.0.0` tag resolves to
`05cc373d78ae74143194e889ff1317de4dfea52e`. The public GitHub release contains
one sdist, three ABI3 platform wheels, and `SHA256SUMS`, matching the asset set
stated in `paper.md`.

The manuscript is nevertheless not a paper about that release alone:

- the inspected normal--normal EVSI implementation, R native-loading repair,
  method-maturity changes, and release-evidence machinery are post-release
  working-tree changes;
- `HEAD` was nine commits after `v1.0.0`, before considering the uncommitted
  changes;
- `paper.md` lines 157–162 cite `1.0.0` but defer identification of the exact
  reviewed revision;
- neither a patch release nor a commit-bound submission manifest identifies the
  complete software state actually described by the manuscript.

The correction is to merge the reviewed changes and either cut a patch release
that contains them or explicitly identify one immutable commit as the review
version and ensure every release-dependent sentence is true for `v1.0.0`.

### Release-evidence manifest

`docs/release/v1.0.0-release-evidence.json` is a meaningful improvement. It
binds version, signed tag, source commit, public asset names and SHA-256
digests, an exact-subject SLSA provenance record, a local SBOM digest and scope
fields, and the Software Heritage snapshot. The validator checks the local tag,
artifact hashes when present, exact attestation subjects, SBOM metadata, and
basic SWHID syntax.

Limits remain:

1. The manifest and validator were untracked in the inspected snapshot, so they
   are not durable repository evidence yet.
2. `--allow-missing-release-assets` necessarily omits local byte verification of
   the public binaries. The hosted workflow is designed to download and verify
   them, but no immutable hosted run for these exact working-tree files was
   available.
3. The Software Heritage check validates the SWHID shape, not that the snapshot
   resolves or contains the recorded tag and commit. That relationship was
   verified manually in this review.
4. The selected attestation is digest-correct and commit-correct, but its source
   URI is `refs/heads/main`, not the immutable tag. The exact source digest
   prevents ambiguity, but a tag-ref attestation or an explicit explanation
   would be clearer.

### SBOM

`docs/release/v1.0.0-sbom.cdx.json` is valid CycloneDX JSON, has a reproducible
digest (`76f8eed64b24eace04a0f78c31ad30121a0aff4f27106e1594663ca2fb78423a`),
and records the release tag and commit. The workflow can regenerate and retain
it with the manifest and checksums for 90 days.

Its scope must be stated more narrowly. The document contains the Python
environment dependency graph (30 components plus the root), but not the Rust
Cargo dependency graph, the native binary composition, or the R and Julia
package graphs. Calling it the mixed-language release's complete dependency
inventory would be misleading. Either produce a composed CycloneDX SBOM for
the Python and Rust build inputs, or call this file the Python lock-environment
SBOM.

The SBOM is not attached to the public `v1.0.0` release. The repository
documentation now says so accurately. Once committed, repository retention is
durable; the 90-day Actions artefact alone is not.

### Software Heritage

The SWHID is genuine and the snapshot contains `v1.0.0`. The manuscript's
phrase “the repository is preserved” is broader than the evidence needed by a
reviewer: this snapshot preserves an origin capture that includes the release
tag, not the current paper or post-release fixes. Prefer:

> The signed `v1.0.0` tag is included in Software Heritage snapshot
> `swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32`.

The acceptance-stage DOI archive remains external and is not a reason to delay
an otherwise eligible initial JOSS submission.

## R and Julia source-binding assessment

The manuscript now describes these as narrower source packages, which matches
the implementation.

The R repair is effective: an installed package can call the C ABI without
Python when the separately built native library is supplied. The R
`DESCRIPTION` records that library as a system requirement and correctly keeps
advanced EVPPI/EVSI paths behind optional Python and `reticulate`. However,
`devtools::install_github(...)` by itself does not produce a working EVPI
installation. The R README tells users to set `VOIAGE_FFI_LIBRARY` but does not
give a complete, version-bound native build and library-selection command.

The Julia package also passes when the reviewer first builds `voiage-ffi` and
sets `VOIAGE_FFI_LIBRARY`. It is not a standalone Julia package: it has no
artifact/JLL resolution and is not installable from General. The readiness
document states this boundary accurately.

These are acceptable secondary source bindings for a JOSS review centred on the
Python distribution, provided the manuscript continues to avoid registry or
standalone-installation claims. A reviewer protocol for the polyglot claim
should include exact Rust build commands, expected library paths on Linux,
macOS, and Windows, and one shared non-zero fixture.

## Reviewer-protocol assessment

`docs/release/joss-independent-validation.md` correctly:

- selects a clean Python 3.12–3.14 environment;
- pins `voiage==1.0.0`;
- gives a deterministic calculation and expected result;
- asks for operating-system, architecture, version, intervention, and failure
  evidence;
- refuses to substitute agents or CI for independent use.

The second exercise at lines 65–68 remains underspecified. “Follow one worked
example relevant to their interests” has no pinned page, command, expected
runtime, expected output, or troubleshooting route. This weakens
reproducibility and makes two validation reports difficult to compare. Add one
named, versioned second exercise while optionally inviting an additional
participant-selected example.

No non-author report was present. The protocol is ready; the external
community-engagement gate is not complete.

## Manuscript reproducibility-claim audit

| Manuscript lines | Claim | Finding | Disposition |
| --- | --- | --- | --- |
| 39–45 | The package calculates VOI measures; Rust provides shared data rules, EVPI, and declared normal--normal EVSI; R and Julia expose narrower EVPI interfaces. | True for the inspected working tree. The normal--normal Rust/Python path and repaired R path are not in the cited `v1.0.0` release. | **Revision blocker:** bind the paper to a release or immutable commit containing these paths. |
| 57–63 | Decision/result objects preserve named context and malformed inputs are rejected; the structure applies across several domains. | Repository schemas and tests support preservation and validation. Cross-domain applicability is primarily developer-created evidence, not independent installation evidence. | Accept with the manuscript's restrained “can represent” wording. |
| 78–84 | Selected calculations and context cross language boundaries; R and Julia are narrower than Python. | The EVPI fixture executes through the shared C ABI in R and Julia when the external library is supplied. No broader cross-language parity was observed. | Verified as currently worded. |
| 88–95 | Rust owns selected shared calculations and R/Julia call shared EVPI, with packaging and parity costs. | Verified. The manual native-library prerequisite demonstrates the stated packaging cost. | Pass. |
| 97–104 | Promotion metadata separates maturity and approximation; normal--normal EVSI declares its model; compatibility estimators are non-stable. | Supported by the inspected post-release code, specifications, and tests. Not supported by the public `v1.0.0` artefacts named later in the paper. | **Revision blocker:** include in a reviewed patch release or distinguish unreleased revision from `1.0.0`. |
| 106–110 | Assurance covers references, invalid inputs, repeatability, parity, installation, operating systems, unit/integration/property/differential/mutation/fuzz/memory-safety, clean-install and cross-platform CI. | The repository contains substantive workflows and tests for these categories. The sentence describes the repository programme, not one result from one release run; “implementation parity” should remain limited to tested kernels. | Pass with a request to link reviewer-facing evidence from the availability section or docs. |
| 110–113 | `v1.0.0` contains an sdist, three wheels and checksums; GitHub records provenance; a separate workflow generates a CycloneDX SBOM. | Asset and provenance claims verified. The workflow and local SBOM exist in the inspected tree, but the SBOM is Python-scoped, not attached to the release, and the revised workflow is not yet frozen in an immutable revision. | Pass only with the existing “separate workflow” wording and explicit SBOM scope elsewhere. |
| 117–128 | Fixed-seed health-example results are reproducible and show study-size/implementation effects. | Machine-readable outputs and the inspected public normal--normal implementation support the values, but these corrected materials are post-`v1.0.0`. | Scientifically credible; release binding unresolved. |
| 130–135 | A versioned same-author cross-project contract exists; neither demonstration is independent adoption; development has been public since July 2025. | The qualification is accurate and appropriately conservative. | Pass; independent use remains an external pre-review gate. |
| 139–148 | AI-assisted output was checked against code, tests, sources, examples, and generated artefacts. | The repository provides extensive automated evidence, but this review cannot independently verify every historical human-review act. The disclosure explicitly assigns responsibility to the author. | Accept as an author declaration, not as independently reproduced history. |
| 157 | Python package and release `1.0.0` are public. | Verified on PyPI and GitHub. | Pass. |
| 157–159 | The fixed-seed script and machine-readable outputs use synthetic data. | Verified from the inspected script/data contract. These exact revised outputs are not identified by `v1.0.0`. | Pass on substance; freeze their revision. |
| 159–161 | The repository is preserved by the cited Software Heritage snapshot. | The SWHID resolves and contains `v1.0.0`, but not the inspected post-release manuscript/software state. | Narrow the sentence to the signed release tag. |
| 161–162 | The exact reviewed revision and release-evidence manifest will be frozen before submission. | Accurate statement of incomplete work, not present-tense reproducibility evidence. | **Submission blocker until performed.** |

## JOSS build and metadata

The repository validator passes and the paper is within the 750–1,750-word
limit. The pinned Open Journals action is appropriate, uses least privilege,
and has a successful hosted build at clean `main`.

The inspected manuscript differs from that hosted revision. A local validator
pass cannot replace the official Inara rendering, and Docker was not available
for an exact local reproduction. The final commit must receive a successful
official build, and the resulting PDF should be visually inspected and linked
from the final evidence record.

## Required changes before round 3

1. Create one immutable review target containing the scientific EVSI fixes, R
   repair, manuscript, validator, SBOM, and release-evidence files. Prefer a
   patch release because the manuscript currently presents these features
   alongside `v1.0.0`.
2. Replace the prospective sentence at `paper.md` lines 161–162 with the exact
   reviewed version, commit, script, seed, output paths, release-evidence
   manifest, and archive relationship.
3. Run the pinned Open Journals build for that exact revision and visually
   inspect the retained PDF.
4. Commit and host the manifest, validator, and SBOM workflow; run the
   release-audit workflow against the immutable target and retain its URL.
5. State the SBOM's Python-only scope or compose it with the Cargo dependency
   graph. Do not imply that the current 30-component graph is the complete
   mixed-language release inventory.
6. Document a source-export build path. Either include source provenance in an
   archival source bundle or give exact, derivable
   `VOIAGE_SOURCE_REVISION`, `VOIAGE_SOURCE_TREE_GIT_OID`, and
   `VOIAGE_SOURCE_CLEAN` instructions.
7. Add complete R and Julia native-library build commands and one common
   expected-output fixture to reviewer-facing documentation.
8. Pin the independent protocol's second exercise with an expected result,
   runtime range, and troubleshooting route.

Independent use under issue #471, the permanent arXiv identifier, JOSS
submission and editorial outcomes, and the acceptance-stage DOI are external
gates. They should remain separate from repository reproducibility.

## Rubric

| Dimension | Score | Deduction |
| --- | ---: | --- |
| Scope, significance, and research use | 156/180 | The research purpose is credible, but no attributable independent use or community engagement is yet documented. |
| Statement of need and audience | 116/120 | Clear problem and audience; only a small deduction for limited reviewer-facing connection from the need to reproducible installation evidence. |
| State of the field and build-versus-contribute case | 123/130 | The comparison and rationale are fair; reproducible comparative evidence remains descriptive rather than exercised in the reviewer protocol. |
| Scientific and numerical accuracy | 143/150 | The revised numerical evidence is strong, but the release cited by the paper does not contain all inspected scientific corrections. |
| Software design and research relevance | 93/100 | Trade-offs are explicit; the unbundled R/Julia native-library path remains a significant reproducibility burden. |
| Reproducibility, packaging, documentation, and tests | 69/100 | Published wheel and sdist, R and Julia probes, provenance, SWH and validators pass. Deductions are for the unfrozen review revision, no exact-snapshot JOSS build, incomplete mixed-language SBOM scope, raw source-export build failure, and underspecified secondary reviewer exercise. |
| Research-impact statement | 58/80 | Developer demonstrations are reproducible and correctly labelled, but independent use is absent. |
| Structure, metadata, and JOSS format | 58/60 | Required sections and word count pass; exact reviewed-version metadata remains prospective. |
| Clarity, accessibility, and sentence quality | 52/55 | Reproducibility boundaries are mostly plain and restrained; archive and SBOM scope need more exact wording. |
| Citations, provenance, declarations, and AI disclosure | 23/25 | Provenance and SWH resolve and declarations are complete; the archive citation does not identify the current reviewed state. |
| **Total** | **891/1000** | **Major revision; fail-closed revision/release blocker applies.** |
