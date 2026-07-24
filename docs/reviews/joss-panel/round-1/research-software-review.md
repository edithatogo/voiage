# Research-software reviewer report

Role: Rust-centred polyglot research-software reviewer

Recommendation: major revision before submission

Score: **815/1000**

## Overall assessment

The software is plausibly within JOSS scope and the manuscript has the required
structure. Several statements do not match installed or released behaviour.

## Blocking findings

### 1. The installed R EVPI path fails

Lines 41–48 and 94–100 say the R and Julia source packages call the shared Rust
interface for EVPI.

The reviewer built `voiage-ffi`, installed `voiageR` into a clean R library,
set `VOIAGE_FFI_LIBRARY`, and called `evpi()`. R returned:

```text
"voiage_v1_evpi_i32_r" not available for .C() for package "voiageR"
```

`r-package/voiageR/R/voiageR.R` loads the external library and then restricts
symbol lookup to `PACKAGE = "voiageR"`, although the Rust library is not
registered as that package's DLL. The existing numerical-reference test
substitutes an R implementation instead of exercising the native path.

The Julia binding called the supplied Rust library successfully and passed all
12 tests.

Required remediation:

- repair the installed R native symbol lookup;
- add an installed-package test that invokes the real Rust library;
- exercise it on supported operating systems;
- correct `DESCRIPTION` compilation and system-requirement metadata;
- correct the R README's licence badge and statement;
- do not restore the released-path claim until a corrected release exists.

### 2. The public release does not include an SBOM

Lines 113–114 state that the v1.0.0 release includes an SBOM. The release
contains three wheels, one source distribution, and `SHA256SUMS`. GitHub
records provenance attestations. The CycloneDX SBOM is retained by a separate
workflow rather than attached to the release.

Accurate current wording:

> The v1.0.0 release provides source and platform wheels, checksums, and GitHub
> provenance attestations. A separate supply-chain workflow generates a
> CycloneDX software bill of materials.

### 3. The field comparison omits `dampack`

`dampack` provides decision-model output analysis, visualisation, sensitivity
analysis, and VOI in R. It is directly relevant to the build-versus-contribute
case and requires a consistent comparison and citation.

### 4. The paper uses a nonexistent maturity label

Lines 106–108 say “Stable, developing, and experimental methods”. The
repository governance taxonomy is `planned`, `experimental`, `fixture-backed`,
and `stable`. A separate method-metadata contract uses `stable`, `approximate`,
`experimental`, and `backend-dependent`. “Developing” appears in neither.
The repository taxonomies should be reconciled before the paper uses exact
labels.

### 5. “Research uses” overstates developer artefacts

The synthetic example and same-author integration contract show reproducibility
and integration readiness, but not substantive or independent research use.
They should be described as developer-produced research artefacts.

## Architecture and sentence findings

- Lines 30–34: define VOI as the expected benefit of reducing uncertainty
  before making a decision.
- Lines 34–39: prefix the capability list with “Through its Python interface”
  so readers do not infer cross-language parity.
- Lines 41–48: distinguish distribution, installability, and method parity
  after fixing R.
- Lines 60–68: replace “one clear description” with versioned decision and
  result schemas; explicitly limit shared binding functionality.
- Lines 82–90: replace categorical statements about other packages with the
  project's actual requirement and parity-maintenance cost.
- Lines 94–100: remove PyO3 and C-ABI detail; explain shared calculations and
  language-specific presentation.
- Lines 102–103: do not claim cross-language consistency until real
  conformance paths pass.
- Lines 104–106: say exactly which feature groups are optional; the base
  scientific Python installation is not lightweight.
- Lines 106–108: use the reconciled source taxonomy.
- Lines 110–114: qualify the scope of test families and accurately locate SBOM
  evidence.
- Lines 118–126: use “developer-produced research artefacts”.
- Lines 128–131: retain the restrained public-history and external-adoption
  boundary.
- Lines 135–146: preserve the AI scope and human-responsibility statement
  without inventing unavailable historical model identifiers.
- Lines 157–160: retain the archive citation; remove peripheral citation
  guidance if space is needed.

Suggested design paragraph:

> `voiage` separates shared calculations from language-specific data handling
> and presentation. Rust supplies versioned contracts and selected stable
> calculations. Python provides the primary modelling, validation, plotting,
> reporting, and command-line interface. R and Julia are narrower bindings
> intended to reuse the same calculations rather than maintain independent
> implementations. This design reduces the risk of language-specific
> numerical drift, but it requires additional packaging and conformance work
> for every binding.

## Bibliography findings

- The `voi` record lists Gianluca Baio as an author; current CRAN and Crossref
  metadata identify Christopher Jackson and Anna Heath as authors and Baio as a
  contributor.
- Add a `dampack` record.
- Normalise DOI casing and explicit CSL types in the retained citation audit.

## Rubric

| Dimension | Score |
| --- | ---: |
| Scope, significance, and research use | 142/180 |
| Statement of need and audience | 112/120 |
| State of the field and build-versus-contribute case | 96/130 |
| Scientific and numerical accuracy | 144/150 |
| Software design and research relevance | 73/100 |
| Reproducibility, packaging, documentation, and tests | 72/100 |
| Research-impact statement | 54/80 |
| Structure, metadata, and JOSS format | 60/60 |
| Clarity, accessibility, and sentence quality | 45/55 |
| Citations, provenance, declarations, and AI disclosure | 17/25 |
| **Total** | **815/1000** |

## External gates

Issue #471 has no non-author report. The arXiv identifier is pending but is not
a JOSS requirement. JOSS submission, screening, review, acceptance, and the
acceptance-stage DOI archive remain external.
