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
| Demonstrated developer research use | Fixed-seed health example and versioned `vop_poc_nz` integration contract | Ready, but independent adoption would strengthen screening |
| Single-author community engagement | Public history contains only the author and automated accounts; [independent validation protocol](joss-independent-validation.md) and issue #471 are ready | External evidence required before submission |
| JOSS manuscript structure | All current required sections and 750–1,750-word contract | Ready |
| Design-thinking account | Kernel/orchestration boundary and cross-language trade-offs described | Ready |
| AI usage disclosure | Tools, retained version limits, scope, human decisions and validation stated | Ready for author verification |
| Funding acknowledgement | Draft says no external funding is declared | Author confirmation required |
| Permanent software archive | Software Heritage snapshot SWHID recorded | Ready; DOI-bearing archive still required at acceptance |
| Reproducible JOSS PDF | Official Open Journals action pinned by commit; three-page artifact visually reviewed | Ready |
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
for version 1.0.0 is the Python package:

| Surface | Reviewer path | Current evidence | Boundary |
| --- | --- | --- | --- |
| Python | `python -m pip install voiage==1.0.0` | PyPI wheel/source release, clean-room exercise, Python 3.12–3.14 CI | Primary JOSS installation |
| Rust | `cargo test --manifest-path rust/Cargo.toml --workspace --exclude voiage-python` plus the PyO3 wheel build | Native crates, property tests, fuzzing, Miri, sanitizers and coverage | Internal workspace; crates.io publication is not required for JOSS |
| R | Source package under `r-package/voiageR` | `R CMD check`, testthat, vignette and manual tooling | Secondary binding; CRAN/r-universe review is independent of JOSS |
| Julia | Source package under `bindings/julia` | `Pkg.test` after building `voiage-ffi` and setting `VOIAGE_FFI_LIBRARY` | Secondary binding; a Julia artifact/JLL is required before standalone General installation |

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

1. obtain one attributable or editor-verifiable independent installation and
   research-use report through issue #471;
2. resolve any problems identified by that exercise;
3. confirm the paper's funding statement, authorship, affiliations, ORCID,
   conflicts of interest, and AI disclosure;
4. wait for and record the permanent arXiv identifier, following the author's
   preferred arXiv-first sequence;
5. submit the repository and `paper.md` directly to JOSS;
6. respond to review comments personally and without generative AI drafting;
7. after acceptance-ready review, tag the reviewed version and create a
   DOI-bearing Zenodo or Figshare archive requested by JOSS.

## Remaining gates

- Confirm that “No external funding is declared for this submission” is
  accurate.
- Confirm the author list, affiliations and ORCID metadata.
- Obtain independent installation and research-use evidence through issue #471;
  automated accounts and same-author integrations do not satisfy this gate.
- Record the permanent arXiv identifier when arXiv assigns it.
- Perform the selected direct authenticated JOSS submission.
- Treat editorial screening, review, acceptance and DOI assignment as external
  outcomes.
