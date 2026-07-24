# Round 2 independent research-software and polyglot review

## Review scope and disposition

This review covers the working-tree snapshot inspected on 24 July 2026:
`HEAD=1dfcd3d42af591ca15fcc03ad958123cc153dbbf` plus the uncommitted round-2
changes present during the review. It assesses the current Rust, Python, R, and
Julia architecture; real installed execution; maturity contracts; release
evidence; CI scope; and the corresponding claims in `paper.md`. It does not
award credit for planned publication, unrun hosted checks, or registry outcomes.

**Score: 866/1000. Disposition: major revision; not ready for JOSS submission.**

The current revision is substantially stronger than round 1. The analytical
normal--normal EVSI path has a declared study model and a Rust-owned reference
implementation. The repaired R package executes the installed native EVPI path,
and the Julia package passes its tests against the same built Rust library. The
paper now describes Python as the primary interface and R/Julia as narrower
source bindings, which matches the inspected architecture.

The snapshot nevertheless triggers the rubric's fail-closed rule. The current
Rust workspace does not pass its required Clippy gate; a clean wheel build from
the generated source distribution fails its provenance check; and the signed
`v1.0.0` release predates both the new scientific EVSI contract and the R native
loading fix. The paper and readiness material do not yet bind their claims to a
reviewable, released revision containing those changes.

## Rubric score

| Dimension | Score | Deduction |
| --- | ---: | --- |
| Scope, significance, and research use | 150/180 | The software is in JOSS scope and the same-author health and interoperability demonstrations are concrete. No attributable non-author research use is documented, and the paper correctly says so. The remaining deduction is for limited demonstrated use rather than manuscript concealment. |
| Statement of need and audience | 114/120 | The decision problem, intended analysts, and consequences of losing labels and assumptions are clear. The description of retained metadata is broader than the behaviour of the simple scalar-returning public functions and should be phrased as a capability of the contract/result workflows. |
| State of the field and build-versus-contribute case | 118/130 | The comparison now includes `voi`, `BCEA`, `dampack`, SAVI, and two Python alternatives, and it states the narrower method depth honestly. The separate-package rationale is plausible, but the language-neutral benefit remains demonstrated only for one shared EVPI kernel in R and Julia. |
| Scientific and numerical accuracy | 142/150 | The analytical EVSI benchmark is correct and independently exercised; the two-loop path targets the declared known-variance model; compatibility estimators emit non-stable warnings. The deduction reflects the still-public compatibility estimators whose information-fraction approximations are not study-likelihood models and the absence of a released revision containing the corrected contract. |
| Software design and research relevance | 94/100 | The Rust-kernel/Python-orchestration boundary and its packaging trade-off are explained in research terms. The current R and Julia surfaces remain consumers of an externally supplied dynamic library rather than independently distributable language packages. |
| Reproducibility, packaging, documentation, and tests | 55/100 | Focused Python, Rust EVSI, installed R, Julia, JOSS, Vale, release-manifest, workflow-lint, and repository-harness checks passed. However, required Rust Clippy fails; the source-distribution-to-wheel path fails without source-identity environment variables; the public `v1.0.0` R path contains the pre-fix symbol lookup; no hosted run exists for this dirty snapshot; and a clean PyPI import did not complete within 112 seconds in the reviewed macOS/Python 3.14 environment. |
| Research-impact statement | 60/80 | The numerical health example and same-author cross-project use are concrete and carefully bounded. They do not yet establish independent adoption or non-author research use. |
| Structure, metadata, and JOSS format | 58/60 | Required sections are present, the body is about 1,133 words, the repository validator passes, and the paper renders through the repository JOSS gate. The availability section still uses future-tense freezing rather than identifying the exact reviewed revision. |
| Clarity, accessibility, and sentence quality | 52/55 | The paper is concise, restrained, and accessible relative to round 1. A few architecture and assurance sentences still compress capability, validation, and release status in ways that obscure which revision or interface is meant. |
| Citations, provenance, declarations, and AI disclosure | 23/25 | Funding, conflicts, authorship responsibility, AI use, release provenance, and the Software Heritage snapshot are stated. The release-evidence record is valid for `v1.0.0`, but it is not evidence for the newer dirty revision under review. |
| **Total** | **866/1000** | **Fail-closed manuscript/release blockers remain.** |

## Executed checks

| Check | Result | Interpretation |
| --- | --- | --- |
| `uv run --extra ci python scripts/validate_joss.py` | Pass | Repository-owned manuscript checks pass. |
| Focused Python scientific, runtime, maturity, stable-API, and release-evidence tests | Pass, 41 tests | The new scientific and metadata contracts are internally consistent. |
| Rust normal--normal EVSI reference tests | Pass, 3 tests | The Rust kernel returns 124.1793655 for the declared 200-participant example and rejects invalid designs. |
| `cargo fmt`, workspace Clippy, then workspace tests | **Fail at Clippy** | `missing_errors_doc` and `cast_precision_loss` in `evsi_normal_normal.rs` prevent the required `-D warnings` gate; workspace tests were consequently not reached by this chained command. |
| Installed R package plus native EVPI call | Pass | After `R CMD INSTALL`, `evpi()` returned 0.5 using `libvoiage_ffi.dylib`; this verifies the current symbol-address repair. |
| R source-local test suite | **Fail, 8 documentation assertions** | `testthat::test_local()` cannot resolve the expected Rd/help surface from its execution context. The native test passes, but the source-local harness is not portable as invoked. |
| Julia `Pkg.test()` with built FFI library | Pass, 12 tests | The current Julia source package and shared EVPI fixture work with an explicitly supplied native-library path. |
| Release-evidence manifest validation | Pass | The checked-in manifest binds the signed `v1.0.0` tag, release assets, attestations, SBOM digest, and Software Heritage SWHID. |
| `tox -e joss` | Pass | Validator and seven JOSS-readiness tests pass. |
| `tox -e vale` | Pass | Zero prose findings in the repository corpus. |
| Repository workflow/security harness | Pass | Zero findings across 27 workflows. |
| `actionlint` on changed binding and SBOM workflows | Pass | The inspected workflow syntax is valid. |
| Current `uv build` through sdist | **Fail** | Maturin cannot rebuild the wheel from the generated sdist because release mode has neither a Git identity nor complete `VOIAGE_SOURCE_*` variables. This is a reviewer-installability defect. |
| Clean PyPI `voiage==1.0.0` wheel installation | Install pass; import unresolved | Dependencies and wheel install, but cold `import voiage` made no progress to user code within 112 seconds before interruption, inside eager SciPy/scikit-learn import. Treat as environment-specific but reproducible reviewer friction. |
| Signed tag verification | Pass | `v1.0.0` has a valid maintainer signature and resolves to `05cc373d78ae74143194e889ff1317de4dfea52e`. |

## Blocking findings

### B1. The reviewed Rust revision fails its required CI gate

`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`
fails on the newly added EVSI kernel because its fallible public function lacks
an `# Errors` section and because `usize` is cast directly to `f64`. This
contradicts readiness claims until corrected and rerun through the complete
hosted Rust job.

### B2. The source distribution is not independently buildable

`uv build` successfully creates `voiage-1.0.0.tar.gz`, then fails while building
a wheel from that archive. The archive build has no Git directory, and the
embedded source identity is not accepted, so `voiage-python/build.rs` aborts a
release build. A JOSS reviewer on an uncovered platform cannot rely on the
sdist. The provenance control should retain its fail-closed intent while
accepting a verified identity embedded in the sdist, and a clean archive-build
test should be required in CI.

### B3. The paper, current revision, and signed release are not the same software

The release-evidence manifest correctly describes `v1.0.0` at commit
`05cc373d78ae74143194e889ff1317de4dfea52e`. The inspected scientific EVSI
kernel, maturity contract, R symbol-loading repair, R operating-system smoke
matrix, and release-evidence workflow changes are later uncommitted work. The
tagged R code still calls `.C(..., PACKAGE = "voiageR")`, which was the failed
installed path repaired in the current tree. The tag also lacks the public
Rust-owned analytical normal--normal EVSI function described by the revised
paper.

Before submission, either publish a signed patch release containing the
reviewed fixes and bind the paper to it, or identify an exact immutable reviewed
commit and remove any implication that `v1.0.0` contains the new capabilities.
For normal JOSS review and archival practice, the patch release is the clearer
choice.

### B4. The current snapshot has no hosted polyglot evidence

The new R Linux/macOS/Windows installed-native matrix is well targeted, but it
exists only in the dirty local tree. Julia is tested only on Ubuntu and requires
`VOIAGE_FFI_LIBRARY`; it has no artifact/JLL-based installation. The current
Clippy failure also predicts a hosted Rust failure. Push the reviewed revision
through the complete required workflows and cite the exact successful run or
commit in the readiness evidence.

### B5. Primary Python import and optional-dependency boundaries need attention

The clean PyPI wheel installed on Python 3.14, but a cold top-level import
remained inside eager scikit-learn/SciPy imports for more than 112 seconds.
Although this may be specific to the reviewed machine and cold binary loading,
it reproduces prior reviewer concern and makes a basic installation smoke test
unreliable. The top-level package should avoid importing heavy optional
statistical stacks until the corresponding feature is called, and CI should
measure a bounded clean `import voiage` plus one EVPI calculation.

## Architecture and maturity assessment

The present architecture is **Python-primary with a Rust execution core for
selected contracts and kernels**, not yet a uniformly Rust-first polyglot
library. Rust owns domain/diagnostic/serialization crates and selected numerical
kernels. Python owns the broad public workflow, labelled objects, model
callbacks, plotting, and the normal-study two-loop orchestration. R and Julia
currently expose only EVPI through the Rust C ABI and require the user or CI to
supply the dynamic library path. The revised paper states this boundary
accurately.

The four-level maturity ladder (`planned`, `experimental`, `fixture-backed`,
`stable`) is now coherent, and approximation/backend dependence are correctly
separated from maturity. The stable EVSI entry is defensible only with its new
scope: the declared default normal-arm two-loop contract and analytical
normal--normal function. Regression, efficient, and moment-based compatibility
paths remain non-stable despite being callable. That distinction should be
visible in generated API documentation and in any CLI result metadata, not only
as a `FutureWarning`.

## Sentence-level audit of relevant paper claims

| Lines | Finding | Required action |
| --- | --- | --- |
| 30–37 | **Pass.** The VOI questions and EVPI/EVPPI/EVSI/ENBS mapping are accurate and accessible. | None. |
| 39–45 | **Revise.** The language split matches the current tree, but “keeping ... with the analysis” is not universal: simple public functions return scalars and do not carry every listed field. The analytical EVSI path is also newer than `v1.0.0`. | Use “contract and result workflows can retain...” and bind the claim to the reviewed release. |
| 49–55 | **Pass.** This is a clear and research-relevant statement of the interoperability problem. | None. |
| 57–63 | **Revise slightly.** Inspectable decision/result objects exist, but rejection and metadata retention vary by entry point. | Replace the universal “records” and “rejects” with a capability-bounded formulation or cite the exact stable workflow. |
| 67–76 | **Pass.** The established R and web alternatives are represented fairly, without dismissing them. | None. |
| 78–81 | **Pass with citation maintenance.** The narrower Python comparison improves the build-versus-contribute case. | Retain only if the cited package scopes remain verified at submission. |
| 83–89 | **Pass.** The separate-package rationale and loss of method depth are proportionate. | None. |
| 93–100 | **Pass for the current tree.** This accurately describes the selected-kernel architecture and the packaging/parity trade-off. | Ensure the immutable reviewed release contains the repaired R path. |
| 102–110 | **Pass scientifically; release qualification required.** The new Rust analytical benchmark and Python two-loop test support the declared normal model. Non-likelihood estimators warn as non-stable. | Do not attribute this contract to `v1.0.0`; expose maturity in result metadata and release a corrected version. |
| 112–119 | **Partly supported.** The repository contains the named test classes and workflows, and release assets/attestations/SBOM evidence exist. The current required Rust gate fails and the sdist cannot rebuild, so this paragraph must not imply that the reviewed snapshot is green. | Correct the gates, prove a clean archive install, run hosted CI, and then retain the paragraph. |
| 123–128 | **Pass.** The synthetic study assumptions are explicit and correspond to the analytical function's inputs. | None. |
| 130–138 | **Pass.** The fixed-seed results and EVSI benchmark agree with the checked numerical evidence; the interpretation is restrained. | Preserve the exact script, seed, and machine-readable outputs in the reviewed release. |
| 140–143 | **Pass.** The figure caption identifies the outputs and synthetic-data boundary. | Confirm the official JOSS renderer includes the figure in the final artifact. |
| 145–151 | **Pass.** Same-author research-workflow use is concrete and is not misrepresented as independent adoption. | Keep issue #471 external and unresolved until attributable evidence exists. |
| 155–164 | **Pass within this review's scope.** The AI disclosure describes tools, uses, retained-version limits, human validation, and responsibility. | No software-review change. |
| 168–169 | **Pass.** Funding and competing-interest declarations are explicit. | No software-review change. |
| 173–178 | **Revise before submission.** The public release and Software Heritage snapshot are real, but they bind `v1.0.0`, not the newer reviewed code. “Will be frozen” describes future work rather than availability. | State the exact released version, Git commit, archive identifier, and evidence manifest actually submitted for review. |

## Registry and external-gate separation

| Surface or gate | Repository-owned readiness | External state |
| --- | --- | --- |
| PyPI/TestPyPI | `v1.0.0` wheels and sdist are published, but the sdist rebuild defect and old scientific contract require repository remediation and preferably a patch release. | Registry publication itself is complete for `v1.0.0`. |
| crates.io | Rust crates are intentionally `publish = false`; JOSS does not require crates.io publication. | No external gate unless the project elects to publish a facade crate. |
| R / CRAN / r-universe | Current installed native EVPI works locally, but standalone native-library distribution is unresolved and the corrected package is unreleased. These are repository-owned blockers, not merely registry delay. | CRAN review and r-universe indexing remain external after packaging is complete. |
| Julia General | Source tests pass with an explicitly supplied library. Artifact/JLL packaging and a reviewer-installable registry path remain repository-owned work. | General registration and registry review remain external after that work. |
| conda-forge | Recipe/feedstock evidence is outside this focused execution review. | Feedstock review and merge are external; do not infer completion here. |
| Software Heritage | Snapshot SWHID is recorded and resolves in release evidence. | Complete for the `v1.0.0` snapshot; a later reviewed release needs its own archival binding. |
| arXiv | Repository records submission number `7861466`, not a permanent identifier. | Announcement and permanent arXiv identifier are external. |
| Independent use, issue #471 | Protocol and issue exist; the repository cannot manufacture non-author evidence. | Attributable installation/research-use evidence is external/human. |
| JOSS submission and review | Manuscript tooling is ready, but blockers B1–B5 prevent submission readiness. | Authenticated submission, editorial screening, review, acceptance, and JOSS DOI are external. |
| Archival DOI at acceptance | No DOI-bearing acceptance archive is claimed. | Create the requested Zenodo/Figshare archive at the JOSS acceptance stage. |

## Required path to a passing round

1. Fix the Rust Clippy errors and run the complete locked workspace checks.
2. Make wheel building from the sdist succeed in a Git-free clean room while
   retaining verifiable embedded provenance.
3. Add a bounded clean Python import-and-EVPI smoke test and remove unnecessary
   eager heavy imports from the top-level path.
4. Run the official R package check and installed native smoke on Linux, macOS,
   and Windows; make the source-local documentation tests location-independent.
5. Keep Julia's source-binding claim, or add artifact/JLL packaging before
   claiming normal Julia installation.
6. Commit the scientific, R, maturity, workflow, documentation, and evidence
   changes; pass all hosted checks at that exact commit.
7. Cut and sign a patch release containing the reviewed code, generate matching
   checksums/provenance/SBOM evidence, request a corresponding Software Heritage
   snapshot, and update the paper's availability sentence to exact identifiers.
8. Repeat the panel review against that immutable released revision. Independent
   use, arXiv announcement, registry acceptance, JOSS review, and DOI assignment
   must remain separately labelled external gates.
