# Independent JOSS verification report

Overall result: **major revisions required — 797/1000**.

The manuscript is scientifically coherent and unusually well supported by tests and release evidence, but it currently contains two material contradictions:

1. The R binding does not successfully execute the installed native Rust EVPI path.
2. The v1.0.0 GitHub release does not include the claimed SBOM.

A third factual error concerns maturity labels: the manuscript says “stable, developing, and experimental,” but `developing` is not part of the repository’s governed taxonomy.

Because these are material factual claims, the rubric’s fail-closed rule would cap the manuscript below 950 even if the remainder scored higher. I made no file changes.

## Sentence-level claim inventory

Status meanings:

- **Verified**: supported by source, executable evidence, or authoritative external records.
- **Contradicted**: available evidence conflicts with the manuscript.
- **Underspecified**: plausible but broader or less precise than the available evidence.
- **Unverifiable**: author declaration, judgement, or prospective intention that cannot be independently established.

| Paper lines | Claim | Evidence or command | Status | Proposed correction |
|---|---|---|---|---|
| [2](/Volumes/PortableSSD/GitHub/voiage/paper.md:2) | Title describes the software’s purpose. | Package APIs, examples and documentation. | Verified | No factual correction required. |
| [8–9](/Volumes/PortableSSD/GitHub/voiage/paper.md:8) | Keywords identify Rust, Python, R, Julia, health economics and decision science. | `rust/`, `voiage/`, `r-package/`, `bindings/julia/`; package functionality. | Verified | Consider adding “value of information” if keyword limits permit. |
| [11–22](/Volumes/PortableSSD/GitHub/voiage/paper.md:11) | Author, ORCID and affiliations. | [CITATION.cff](/Volumes/PortableSSD/GitHub/voiage/CITATION.cff), CodeMeta and paper metadata agree. Affiliations remain author-declared. | Unverifiable | Metadata is internally consistent. No correction unless the author changes it. |
| [23](/Volumes/PortableSSD/GitHub/voiage/paper.md:23) | Date is 22 July 2026. | This is the v1.0.0 release date, but not the eventual JOSS submission date. | Underspecified | Use the actual submission date when submitting, or omit it if the JOSS build supplies the date. |
| [25](/Volumes/PortableSSD/GitHub/voiage/paper.md:25) | Repository URL. | Live GitHub repository. | Verified | None. |
| [30–31](/Volumes/PortableSSD/GitHub/voiage/paper.md:30) | VOI measures how much a decision could improve with further information. | Standard VOI literature in [paper.bib](/Volumes/PortableSSD/GitHub/voiage/paper.bib). | Verified | More precise plain-language wording: “VOI estimates the expected gain from making the best decision with additional information, relative to deciding with current information.” |
| [31–34](/Volumes/PortableSSD/GitHub/voiage/paper.md:31) | VOI is useful across health, public policy, environment, marketing and business. | Health example is concrete; repository APIs are generic. No concrete research-use evidence or citations cover every named field. | Underspecified | Say the methods “can be applied” across these fields and add representative cross-domain VOI citations, or narrow the list to demonstrated settings. |
| [34–35](/Volumes/PortableSSD/GitHub/voiage/paper.md:34) | Voiage computes, checks and communicates VOI analyses. | Public Python API, schemas, diagnostics, CLI and examples. | Verified | None. |
| [35–39](/Volumes/PortableSSD/GitHub/voiage/paper.md:35) | Supports EVPI, EVPPI, EVSI, ENBS, CEAF, dominance and diagnostics. | v1.0.0 source; `voiage/analysis.py`, Rust numerics and tests. | Verified | Consider shortening the catalogue and connecting the measures to research decisions. |
| [41–43](/Volumes/PortableSSD/GitHub/voiage/paper.md:41) | v1.0.0 has Rust contracts, diagnostics, serialization and selected kernels. | [rust/Cargo.toml](/Volumes/PortableSSD/GitHub/voiage/rust/Cargo.toml); Rust domain, diagnostics, serialization and numerics crates at the release tag. | Verified | “Selected numerical kernels” is appropriately bounded. |
| [43–44](/Volumes/PortableSSD/GitHub/voiage/paper.md:43) | Python retains the wider analytical surface. | [pyproject.toml](/Volumes/PortableSSD/GitHub/voiage/pyproject.toml), Python modules and exports. | Verified | None. |
| [44–46](/Volumes/PortableSSD/GitHub/voiage/paper.md:44) | R and Julia source packages call the same versioned Rust EVPI interface. | Julia `Pkg.test()` passed against the built FFI. An installed R package call failed: `"voiage_v1_evpi_i32_r" not available for .C() for package "voiageR"`. | **Contradicted** | Fix the installed R native call and add a clean-install smoke test before retaining this statement. Until then, distinguish the working Julia path from the non-operational R adapter. |
| [45–46](/Volumes/PortableSSD/GitHub/voiage/paper.md:45) | The secondary bindings do not expose the full Python surface or standalone registry installation. | Julia exports EVPI only; R’s wider functions use Python; neither package was found in Julia General or CRAN. | Verified | None. |
| [46–48](/Volumes/PortableSSD/GitHub/voiage/paper.md:46) | The boundary is documented explicitly. | Binding READMEs and architecture documentation describe the intended boundary. | Underspecified | Documentation describes the intended design, but it should not imply that the R native route currently works. |
| [52](/Volumes/PortableSSD/GitHub/voiage/paper.md:52) | VOI addresses whether further research is worthwhile. | Claxton and Ades references; package ENBS/EVSI functionality. | Verified | None. |
| [53–58](/Volumes/PortableSSD/GitHub/voiage/paper.md:53) | Implementations are fragmented and can create inconsistent inputs, units, conventions and outputs. | Plausible design rationale, but no comparative study or direct citations establish the breadth or consequences claimed. | Underspecified | Add evidence or state this as the project’s motivating observation rather than an established empirical finding. |
| [60–62](/Volumes/PortableSSD/GitHub/voiage/paper.md:60) | Voiage associates decision descriptions, checks, examples, warnings and results. | Schemas, contracts, diagnostics, examples and serialization code. | Verified | None. |
| [62–65](/Volumes/PortableSSD/GitHub/voiage/paper.md:62) | Intended audience includes applied researchers and analysts. | Documentation and API design. | Verified as design intent | Prefer “is intended for” rather than implying demonstrated adoption across all listed audiences. |
| [65](/Volumes/PortableSSD/GitHub/voiage/paper.md:65) | Python is installable. | PyPI v1.0.0, GitHub release assets and successful hosted wheel-install tests. | Verified | A local clean-room import was inconclusive because dependency import stalled under heavy concurrent load; hosted evidence remains positive. |
| [66–68](/Volumes/PortableSSD/GitHub/voiage/paper.md:66) | R and Julia use the same validated inputs and rules through Rust. | Julia direct EVPI path passed; installed R native execution failed. | **Contradicted** | Restrict the statement to Julia until the R path is repaired and exercised after installation. |
| [72–73](/Volumes/PortableSSD/GitHub/voiage/paper.md:72) | VOI methods are well established. | Claxton and Ades references resolve and support the statement. | Verified | None. |
| [74–75](/Volumes/PortableSSD/GitHub/voiage/paper.md:74) | R package `voi` provides EVPI, EVPPI, EVSI and ENBS methods. | Current [CRAN `voi` manual](https://cran.r-project.org/web/packages/voi/voi.pdf). | Verified | None. |
| [75–76](/Volumes/PortableSSD/GitHub/voiage/paper.md:75) | BCEA provides Bayesian cost-effectiveness analysis facilities. | Cited BCEA JOSS paper and package record. | Verified | None. |
| [76–78](/Volumes/PortableSSD/GitHub/voiage/paper.md:76) | SAVI offers web-based VOI workflows. | [SAVI documentation](https://savi.shef.ac.uk/SAVI/). | Verified | None. |
| [78–80](/Volumes/PortableSSD/GitHub/voiage/paper.md:78) | Existing tools are appropriate within their settings. | Editorial assessment rather than a falsifiable fact. | Unverifiable | Acceptable as restrained interpretation. |
| [82–85](/Volumes/PortableSSD/GitHub/voiage/paper.md:82) | Voiage is not intended to replace specialist tools and has a different scope. | Repository design and documented feature boundaries. | Verified as design intent | Replace “cross-domain” with “domain-neutral” unless additional domain evidence is added. |
| [86–88](/Volumes/PortableSSD/GitHub/voiage/paper.md:86) | Adding to a specialist R package would change that package’s purpose. | Counterfactual design judgement. | Unverifiable | Frame it directly: “We therefore chose a separate package so that…” |
| [88–90](/Volumes/PortableSSD/GitHub/voiage/paper.md:88) | Voiage has less method-specific depth and narrow secondary bindings. | API comparison and binding exports. | Verified | This is an appropriately restrained limitation. |
| [94](/Volumes/PortableSSD/GitHub/voiage/paper.md:94) | Design separates domain models, numerical work and language interfaces. | Rust workspace and binding layout. | Verified | None. |
| [95–97](/Volumes/PortableSSD/GitHub/voiage/paper.md:95) | Rust owns validated types, reductions, interfaces and shared dependencies. | Rust domain/numerics/FFI crates and manifests. | Verified | “Owns” may be replaced with “implements” for non-developer readers. |
| [97–99](/Volumes/PortableSSD/GitHub/voiage/paper.md:97) | Python retains broader methods and optional integrations. | Python modules and dependency extras. | Verified | None. |
| [99–100](/Volumes/PortableSSD/GitHub/voiage/paper.md:99) | Released R and Julia packages exercise the shared Rust EVPI path. | Julia works. R fails after installation. No standalone R/Julia release or registry record was found. | **Contradicted** | Use “source bindings” rather than “released packages”; fix and test R before claiming both execute the path. |
| [102–103](/Volumes/PortableSSD/GitHub/voiage/paper.md:102) | Architecture provides consistent behaviour across languages. | Shared fixtures exist, but the R production path fails. | **Contradicted** | “The architecture is intended to reduce behavioural drift; parity is demonstrated for paths covered by the shared conformance tests.” |
| [103–105](/Volumes/PortableSSD/GitHub/voiage/paper.md:103) | Components can be tested without unrelated dependencies. | Rust crate boundaries, Python extras and focused tests. | Verified | None. |
| [105–106](/Volumes/PortableSSD/GitHub/voiage/paper.md:105) | Optional dependencies are isolated. | [pyproject.toml](/Volumes/PortableSSD/GitHub/voiage/pyproject.toml) extras and import boundaries. | Verified | None. |
| [106–108](/Volumes/PortableSSD/GitHub/voiage/paper.md:106) | Methods are labelled stable, developing and experimental. | Governed labels include `stable`, `fixture-backed`, `approximate`, `experimental` and `backend-dependent`; `developing` was not found. | **Contradicted** | Use the exact governed taxonomy, or reconcile the repository taxonomy first. |
| [110–113](/Volumes/PortableSSD/GitHub/voiage/paper.md:110) | Assurance includes analytical fixtures, property tests, cross-language conformance, fuzzing, Miri, sanitizers and cross-platform builds. | Relevant test files and successful GitHub Actions runs. | Verified | Say “targeted Rust fuzzing and Miri checks” to avoid implying whole-codebase coverage. |
| [113–114](/Volumes/PortableSSD/GitHub/voiage/paper.md:113) | v1.0.0 includes source and platform wheels and checksums. | [v1.0.0 release](https://github.com/edithatogo/voiage/releases/tag/v1.0.0): one source distribution, three platform wheels and valid SHA-256 checksums. | Verified with terminology error | Replace “source wheel” with “source distribution.” |
| [113–114](/Volumes/PortableSSD/GitHub/voiage/paper.md:113) | v1.0.0 includes an SBOM. | No SBOM is attached to the release. The workflow produces SBOMs as Actions artifacts, but no exact-release SBOM was found. | **Contradicted** | Attach and verify an exact-release SBOM, or say that CI generates SBOMs for reviewed revisions without claiming it is a v1.0.0 release asset. |
| [114](/Volumes/PortableSSD/GitHub/voiage/paper.md:114) | Release includes provenance. | `gh attestation verify` succeeded for a v1.0.0 wheel; SLSA attestations match release runs. | Verified | Prefer “verifiable build-provenance attestations.” |
| [118](/Volumes/PortableSSD/GitHub/voiage/paper.md:118) | Two developer-led research uses are documented. | Synthetic health example and `vop_poc_nz` integration contract. | Verified with boundary | “Two developer-led applications” is slightly more precise than “research uses.” |
| [119–122](/Volumes/PortableSSD/GitHub/voiage/paper.md:119) | Synthetic health example compares parameter uncertainty and study sizes under uptake, delay, population and cost assumptions. | [generate_paper_health_example.py](/Volumes/PortableSSD/GitHub/voiage/scripts/generate_paper_health_example.py), generated CSVs and [test_paper_health_example.py](/Volumes/PortableSSD/GitHub/voiage/tests/test_paper_health_example.py). | Verified | Consider reporting one restrained numerical result so the paper shows an analytical finding, not only the workflow. |
| [123–126](/Volumes/PortableSSD/GitHub/voiage/paper.md:123) | `vop_poc_nz` integration exchanges versioned decision records and validates provenance and analytical fixtures. | [VOP integration specification](/Volumes/PortableSSD/GitHub/voiage/specs/integration/vop-voiage), `validate_vop_compatibility.py`, public pinned upstream commits. | Verified | Expand `vop_poc_nz` in plain language on first mention. |
| [125–126](/Volumes/PortableSSD/GitHub/voiage/paper.md:125) | Uses are public and reproducible but not independent adoption. | Public repositories and validation commands; both are author-controlled. | Verified | The non-independence qualification is important and should remain. |
| [128](/Volumes/PortableSSD/GitHub/voiage/paper.md:128) | Repository has been public since July 2025. | GitHub creation and initial commit date: 3 July 2025. | Verified | None. |
| [128–131](/Volumes/PortableSSD/GitHub/voiage/paper.md:128) | Documentation and examples support future applied use. | Examples and documentation exist; actual future use is unknowable. | Underspecified | “Documentation and examples are provided for…” or “are intended to support…” |
| [130–131](/Volumes/PortableSSD/GitHub/voiage/paper.md:130) | No claim of external uptake is made without attributable evidence. | Issue #471 and readiness records explicitly retain that gate. | Verified | None. |
| [135–140](/Volumes/PortableSSD/GitHub/voiage/paper.md:135) | AI tools and model families were used as described. | Jules bot commits are observable; complete model/session provenance is not independently recoverable. | Unverifiable | Treat as an author disclosure. Avoid greater tool-level specificity than the retained records support. |
| [140–145](/Volumes/PortableSSD/GitHub/voiage/paper.md:140) | Human author reviewed and validated all outputs and retains responsibility. | Author attestation; tests and merges support process but cannot prove universal review. | Unverifiable | “The author reviewed the submitted manuscript and accepts responsibility for its contents” is more directly attestable. |
| [145–146](/Volumes/PortableSSD/GitHub/voiage/paper.md:145) | AI tools will not participate in editor/reviewer exchanges. | Prospective policy. | Unverifiable | “The author intends to respond personally to editor and reviewer correspondence.” |
| [150–153](/Volumes/PortableSSD/GitHub/voiage/paper.md:150) | Infrastructure acknowledgement, no external funding and no conflicts. | Author declarations; infrastructure names are plausible. | Unverifiable | Acceptable if reconfirmed immediately before submission. |
| [157](/Volumes/PortableSSD/GitHub/voiage/paper.md:157) | v1.0.0 is publicly available. | Live GitHub release and PyPI record. | Verified | None. |
| [158–159](/Volumes/PortableSSD/GitHub/voiage/paper.md:158) | Repository snapshot is preserved by Software Heritage. | Snapshot resolves as `swh:1:snp:767ef…`. | Verified | If exact-release preservation matters, cite an object proven to contain or identify the v1.0.0 tag. |
| [159–160](/Volumes/PortableSSD/GitHub/voiage/paper.md:159) | Citation metadata is supplied. | CITATION.cff, CodeMeta and software DOI record. | Verified | None. |

## Executable evidence reviewed

The following checks succeeded:

- `uv run python scripts/validate_joss.py`
- `uv run tox -e joss`
- Focused health-example, contract-bundle and VOP validation tests
- `uv run python scripts/validate_vop_compatibility.py`
- `uv run python scripts/check_consumer_matrix.py`
- Rust FFI release build
- Julia clean project instantiation and `Pkg.test()`
- GitHub release checksum verification
- `gh attestation verify` against a v1.0.0 wheel
- JOSS Actions build for commit `1dfcd3d4`
- Render inspection of the generated three-page JOSS PDF

The decisive failing check was the installed R-package native call:

```text
"voiage_v1_evpi_i32_r" not available for .C() for package "voiageR"
```

The R workflow currently builds and checks the package but does not execute this installed-package native path. Its green status therefore does not validate the manuscript’s R architecture claim.

The generated JOSS PDF is structurally healthy: three pages, readable, complete references, and no visible rendering loss. The body contains approximately 1,005 words, within JOSS’s 750–1,750-word guidance. See the current [JOSS paper guidance](https://joss.readthedocs.io/en/latest/paper.html) and [review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html).

## Scientific coherence

The scientific logic is sound:

- VOI is correctly connected to research prioritisation and decision uncertainty.
- EVPI, partial uncertainty, study value and implementation effects form a coherent analytical sequence.
- The synthetic health example has explicit priors, a fixed seed, a normal–normal study model, study costs, population, delay and uptake assumptions.
- The manuscript does not present synthetic results as empirical clinical evidence.
- The `vop_poc_nz` integration is correctly distinguished from independent adoption.

The principal scientific weaknesses are presentational rather than mathematical:

1. The manuscript names a worked analytical example but reports no result from it. One clearly labelled synthetic result would demonstrate that the analysis yields an interpretable decision rather than merely proving that the pipeline runs.

2. The domain argument is broader than the evidence. Health economics is concretely demonstrated; policy, environment, marketing and business are only asserted as possible applications.

3. “Fragmented implementations” and the risks of inconsistent conventions are plausible but not directly supported by citations or comparative evidence.

4. The state-of-the-field discussion is selective. It should consider at least `dampack`, and possibly `heemod` or `hesim`, or explain why the comparator set is intentionally limited.

5. The paper remains somewhat architecture-heavy for applied readers. The Rust separation should be linked directly to scientific value: preserving units, labels, decision alternatives and numerical meaning across language interfaces.

6. The manuscript’s VOI definition could be marginally more precise without becoming technical.

I found no numerical contradiction in the health example’s generated evidence. The R failure is an interface/packaging defect, not an error in the underlying Rust EVPI calculation.

## Rubric score

| Dimension | Maximum | Score | Deduction |
|---|---:|---:|---|
| Scope, significance and research use | 180 | **150** | Concrete developer-led examples, but significance is mostly prospective and no independent use is yet attributable. |
| Statement of need and audience | 120 | **101** | Audience is clear; the gap and its consequences are broad and lightly evidenced. |
| State of field and build-versus-contribute rationale | 130 | **108** | Existing descriptions are accurate, but the comparator set is incomplete and part of the rationale is counterfactual. |
| Scientific and numerical correctness | 150 | **136** | Core methods and synthetic model are coherent; broad domain claims are under-cited and no numerical result is reported. |
| Software design and research relevance | 100 | **66** | Architecture is substantive, but the R path fails, consistency is overstated, and the maturity taxonomy is wrong. |
| Reproducibility, packaging, documentation and tests | 100 | **58** | Strong automation and release provenance; major deductions for the untested failing R path and false release-SBOM claim. |
| Research impact and evidence of use | 80 | **58** | Two reproducible developer-led uses, but no attributable independent installation or research use. |
| Structure, metadata and JOSS compliance | 60 | **57** | Required structure, length and rendered PDF are good; date handling and lack of a reported example result cost minor points. |
| Clarity, accessibility and sentence quality | 55 | **43** | Restrained overall, but several dense engineering sentences and unexplained project terminology remain. |
| Citations, provenance, declarations and AI disclosure | 25 | **20** | Bibliography is internally complete; several broad claims lack direct sources and declarations remain author-attested. |
| **Total** | **1000** | **797** | **Major revisions required** |

## Manuscript blockers

These prevent a factually defensible JOSS submission:

1. Fix or remove the R native-path claims at lines 44–48, 66–68, 99–103.
2. Remove or correct the claim that the v1.0.0 release includes an SBOM.
3. Replace `developing` with the actual governed maturity labels.
4. Replace “source wheel” with “source distribution.”
5. Bound the cross-domain and implementation-fragmentation claims or add supporting citations.
6. Clarify that the two applications are developer-led evidence, not independent uptake.
7. Reconcile “released R and Julia packages” with their actual source-only, non-registry status.

Recommended but not strictly blocking:

- Add one result from the synthetic health example.
- Expand `vop_poc_nz` in plain language.
- Add or discuss `dampack` in the related-work section.
- Make the architecture paragraph explain research integrity rather than internal implementation.
- Set the manuscript date at submission.

## Repository blockers

These require code, packaging or release work rather than manuscript wording alone:

1. Repair R native symbol resolution after package installation.
2. Add an R clean-install test that calls the real Rust EVPI function without mocks.
3. Decide whether R should package/link the native library or resolve an explicitly loaded external symbol without the incorrect `PACKAGE` restriction.
4. Generate an exact-v1.0.0 CycloneDX SBOM and attach it to the release if the paper will claim it as a release asset.
5. Reconcile the maturity-label taxonomy across governance, contracts, specifications and manuscript.
6. Consider publishing independently installable R and Julia packages before calling those interfaces released packages.

## External gates

These are not manuscript defects:

- Attributable, non-author installation or research-use evidence under issue #471.
- Permanent arXiv identifier and announcement, if the author retains arXiv-first sequencing.
- JOSS submission, editorial screening, review and acceptance.
- Acceptance-stage archival release DOI.
- CRAN and Julia General registration, if pursued before or during review.

The lack of independent use may increase JOSS screening risk, especially for a single-author project, but it should remain an external evidence gate rather than being disguised through stronger manuscript wording.

<oai-mem-citation>
<citation_entries>
MEMORY.md:1390-1390|note=[used prior manuscript framing and no-overclaim preference]
</citation_entries>
<rollout_ids>
019f5676-c3ee-7fe1-a49b-0476d3dba926
</rollout_ids>
</oai-mem-citation>
