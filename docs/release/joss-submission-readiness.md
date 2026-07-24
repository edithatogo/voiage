# JOSS submission readiness

## Current boundary

The arXiv draft submission `7861466` was rechecked on 24 July 2026 in the
authenticated arXiv account and remains `submitted`. It is not yet listed under
the account's announced articles. A permanent arXiv identifier and announcement
are not available and must not be inferred from the submission number.

The JOSS package is:

- `paper.md`: the JOSS-format manuscript;
- `paper.bib`: the manuscript bibliography;
- `CITATION.cff` and `codemeta.json`: software citation and discovery metadata;
- `scripts/validate_joss.py`: fail-closed repository preflight;
- `.github/workflows/joss-paper.yml`: pinned Open Journals/Inara PDF build.

The canonical arXiv preprint remains `paper/main.tex`; the JOSS adaptation does
not replace it.

## Current JOSS screening matrix

| Requirement | Evidence | State |
| --- | --- | --- |
| OSI-approved licence | Root `LICENSE`, Apache-2.0 | Ready |
| Public repository and issue tracker | `edithatogo/voiage` on GitHub | Ready |
| More than six months of public development | Public history from July 2025 | Ready |
| Iterative open development | Distributed commits, issues, pull requests, changelog and tagged releases | Ready |
| Research application | VOI analysis for research prioritisation and probabilistic decision models | Ready |
| Developer-created demonstrations | Fixed-seed health example and versioned `vop_poc_nz` integration contract | Reproducible; not represented as independent research use |
| Research-use evidence | The same-author `vop_poc_nz` bundle documents an interoperability contract, not execution of `voiage` in research. The [independent validation protocol](joss-independent-validation.md) and issue #471 seek attributable installation, use, or feedback | Externally pending; no independent use is claimed |
| JOSS manuscript structure | All current required sections and 750–1,750-word contract | Ready |
| Design-thinking account | Kernel/orchestration boundary and cross-language trade-offs described | Ready |
| Author metadata | Dylan Mordaunt, ORCID and three affiliations confirmed by the author on 24 July 2026 | Ready |
| AI usage disclosure | Tools, retained version limits, and scope are stated. The comprehensive human decision, review, modification, and validation affirmation has been requested from the author | Awaiting explicit author attestation |
| Funding and competing interests | No external funding and no competing interests confirmed by the author on 24 July 2026 | Ready |
| Permanent software archive | Software Heritage snapshot SWHID recorded | Ready; DOI-bearing archive still required at acceptance |
| Release evidence | Exact v1.0.0 tag, commit, asset digests, verified provenance, Python runtime CycloneDX SBOM and SWHID are bound in `docs/release/v1.0.0-release-evidence.json`; version 2.0.0 is the reviewed release candidate | Historical evidence ready; publish and bind the v2.0.0 mixed-language SBOM, provenance, digests and archive identifier before submission |
| Reproducible JOSS PDF | Official Open Journals action pinned by commit; the available six-page PDF predates the current source | Rebuild and inspect from the exact reviewed revision |
| arXiv reference | Submission verified; permanent identifier pending | External gate |
| JOSS submission and review | No submission claimed | Author and external gate |

Run the repository-owned preflight with:

```console
uv run python scripts/validate_joss.py
uv run tox -e joss
```

The hosted workflow produces `paper.pdf` with the Open Journals toolchain and
retains it as an Actions artifact. It never submits the paper.

## Reviewer-facing packaging

JOSS reviews the software that the paper describes. The primary review surface
for version 2.0.0 is the Python package:

| Surface | Reviewer path | Current evidence | Boundary |
| --- | --- | --- | --- |
| Python | `python -m pip install voiage==2.0.0` | Release-candidate wheel/source build and Python 3.12–3.14 CI; public PyPI installation must be verified after publication | Primary JOSS installation |
| Rust | `cargo test --manifest-path rust/Cargo.toml --workspace --exclude voiage-python` plus the PyO3 wheel build | Native crates, property tests, fuzzing, Miri, sanitizers and coverage | Internal workspace; crates.io publication is not required for JOSS |
| R | `cargo build --manifest-path rust/Cargo.toml --release --locked --package voiage-ffi`, then `R CMD build r-package/voiageR`, `R CMD check --as-cran --no-manual voiageR_*.tar.gz`, and an installed smoke test with `VOIAGE_FFI_LIBRARY` set to the platform library | Installed package calls the separately built Rust EVPI library; tests pass, while the current package check reports two vignette warnings; Linux/macOS/Windows native smoke matrix is configured | Secondary binding; CRAN/r-universe review is independent of JOSS |
| Julia | `cargo build --manifest-path rust/Cargo.toml --release --locked --package voiage-ffi`, then `VOIAGE_FFI_LIBRARY=<platform-library> julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` | The fixture is packaged with the Julia source, and CI tests an archived standalone package copy against the separately built native library | Secondary binding; a Julia artifact/JLL is required before standalone General installation |

The paper therefore describes R and Julia as source bindings and does not claim
that they are independently installable from their language registries. A JOSS
reviewer can assess the primary Python package without installing those
secondary surfaces.

## Selected submission route

The selected route is **direct JOSS submission** after the remaining evidence
gates are satisfied. This preserves the package's Rust-centred, polyglot scope
rather than presenting the Python binding as the entire submission.

pyOpenSci remains a useful later review of the Python package's packaging,
usability, and maintenance practices. Its partner route is not selected for
this submission because it reviews the Python surface rather than the complete
Rust-centred polyglot package. The project will not enter simultaneous active
reviews at pyOpenSci and JOSS. rOpenSci would be relevant only if `voiageR`
became the primary independently installable research package.

The recommended sequence is:

1. publish and archive the exact software revision described by the final
   paper, and retain its hosted test, SBOM, provenance, and digest evidence;
2. continue seeking attributable independent installation or use evidence
   through issue #471 without representing it as a universal JOSS prerequisite;
   automated accounts and AI-agent runs do not establish independent adoption;
3. wait for and record the permanent arXiv identifier, following the author's
   preferred arXiv-first sequence;
4. submit the repository and `paper.md` directly to JOSS;
5. respond to review comments personally and without generative AI drafting;
6. after acceptance-ready review, tag the reviewed version and create a
   DOI-bearing Zenodo or Figshare archive requested by JOSS.

## Remaining gates

- Continue the independent installation and research-use request in issue #471
  as additional engagement evidence; do not claim independent adoption until
  attributable evidence exists.
- Record the permanent arXiv identifier when arXiv assigns it.
- Perform the selected direct authenticated JOSS submission.
- Treat editorial screening, review, acceptance and DOI assignment as external
  outcomes.
