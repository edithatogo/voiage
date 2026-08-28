# JOSS submission readiness

## Current boundary

The authenticated arXiv account was rechecked on 26 July 2026. Submission
`7861466` is no longer present in the active-submission table. Replacement
submission `7870358` exists only as an incomplete start-stage draft, expires on
9 August 2026, and has no files, metadata, category, or licence recorded. It is
not evidence of a completed resubmission. The author needs to complete the
author-controlled fields and submission before the selected arXiv-first JOSS
sequence can continue.

The JOSS package is:

- `paper.md`: the JOSS-format manuscript;
- `paper.bib`: the manuscript bibliography;
- `paper/health-example-methods.md`: the complete analytical model and
  independently checked benchmarks behind the worked example;
- `paper/reproduction-manifest.json`: structured inputs, seeds, lock digest,
  outputs, hashes, and clean-regeneration command;
- `paper/joss-article-contract.json`: current-criteria section, evidence, and
  exact 1,600 ±2% word-count contract;
- `paper/joss-claim-evidence.json`: material claim-to-evidence ledger;
- `paper/joss-references.verification.json`: SourceRight reference records
  queued for final human review;
- `paper/joss-editorial-assurance.json`: source-hash-bound SourceRight,
  Authentext, Textstat, and human-gate state;
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
| Credible near-term significance | Fixed-seed health example, independent analytical benchmarks, sensitivity scenarios, machine-readable results, structured reproduction manifest, and clean regeneration in CI | Ready as specific reproducible material; this does not establish human engagement |
| Research-workflow integration | The same-author `vop_poc_nz` workflow installed `voiage` 2.0.0 from PyPI outside the checkout and executed a bounded HPV-vaccination model run recorded in `paper/joss-developer-research-use.json` | Demonstrated developer research use; no independent adoption claimed |
| Demonstrated research use | The released package was executed in the documented developer research workflow with a fixed revision, input digest, environment boundary, and reproduction command | Ready for the developer-use criterion; non-author engagement remains separate |
| Collaborative-effort screening | The detailed review criteria treat a single-author project without community engagement, external use, or collaborative input as not acceptable. The submission and editorial guides separately describe non-author engagement as a strong positive pre-review signal rather than a hard gate. The [independent validation protocol](joss-independent-validation.md) and issue #471 seek attributable evidence | Author-selected prerequisite and material review risk; agents, bots, and same-author repositories do not qualify |
| JOSS manuscript structure | All contracted sections are substantive and ordered; hosted release-bound run `30202496481` confirms the article contract, including 1,583 body words inside the repository's 1,568–1,632 target and JOSS's 750–1,750 range | Ready for the release-bound source |
| Citation and prose assurance | SourceRight reconciles all 18 occurrences, all 15 references have queued sidecars, and six non-DOI software/web warnings remain; selected Authentext blocking patterns report no finding | Machine checks ready; final human source check pending |
| Design-thinking account | The Rust reference-calculation boundary, protection against cross-language drift, native-build cost, and deliberately narrower R/Julia interfaces are described | Ready |
| Author metadata | Dylan Mordaunt, ORCID and three affiliations confirmed by the author on 24 July 2026; each affiliation is linked to its verified ROR record | Ready |
| AI usage disclosure | Tool families, retained identifier limits, scope, and verification approach are stated. `paper/joss-editorial-assurance.json` records the author's 27 July 2026 confirmation that every retained AI-assisted output was reviewed, modified as needed, and validated; `paper.md` records the best-available tool/model inventory without inventing unavailable identifiers | AI affirmation recorded; historical identifiers not exposed or retained remain explicitly unavailable |
| Funding and competing interests | No external funding and no competing interests confirmed by the author on 24 July 2026 | Ready |
| Permanent software archive | Software Heritage snapshot SWHID recorded | Ready; DOI-bearing archive still required at acceptance |
| Release evidence | Exact v2.0.0 tag, commit, asset digests, verified provenance, mixed-language CycloneDX SBOM and SWHID are bound in `docs/release/v2.0.0-release-evidence.json` | Reviewed release evidence ready |
| Reproducible JOSS PDF | Open Journals run `30202496481` built commit `5c472402ed0fa0e70af3bd98c6d1dde59b5d7811`; artifact `8632098142` has digest `sha256:85b58a21cba15577d874c690317c571192b9dde943007a0308a3a438bf127a8b`, and its six-page PDF has SHA-256 `132af479c9d76091478459652ff12091d04bd3dd426ef5e90265ec1e4bab3e71`. Every page was visually inspected; the hosted Textstat report remains review-only evidence | Ready for the release-bound source |
| arXiv reference | Submission `7861466` is absent. Replacement `7870358` is incomplete at the start stage, expires 9 August 2026, and has no permanent identifier | Author and external completion gate |
| JOSS submission and review | No submission claimed | Author and external gate |

Run the repository-owned preflight with:

```console
uv run python scripts/validate_joss.py
uv run python scripts/audit_joss_sources.py
uv run python scripts/audit_joss_authentext.py
uv run tox -e joss
```

The hosted workflow produces `paper.pdf` with the Open Journals toolchain and
retains it as an Actions artifact. It never submits the paper.

## Reviewer-facing packaging

JOSS reviews the software that the paper describes. The primary review surface
for version 2.0.0 is the Python package:

| Surface | Reviewer path | Current evidence | Boundary |
| --- | --- | --- | --- |
| Python | `python -m pip install voiage==2.0.0` | Public PyPI wheel clean-installed outside the checkout; runtime reports the Rust core at version 2.0.0 and source revision `e849e89152c306e79c96d0a8a9815ee5faca0529` | Primary JOSS installation |
| Rust | `cargo test --manifest-path rust/Cargo.toml --workspace --exclude voiage-python` plus the PyO3 wheel build | Native crates, property tests, fuzzing, Miri, sanitizers and coverage | Internal workspace; crates.io publication is not required for JOSS |
| R | `cargo build --manifest-path rust/Cargo.toml --release --locked --package voiage-ffi`, then `R CMD build r-package/voiageR`, `R CMD check --as-cran --no-manual voiageR_*.tar.gz`, and an installed smoke test with `VOIAGE_FFI_LIBRARY` set to the platform library | Installed package calls the separately built Rust EVPI library; tests pass, while the current package check reports two vignette warnings; Linux/macOS/Windows native smoke matrix is configured | Secondary binding; CRAN/r-universe review is independent of JOSS |
| Julia | `cargo build --manifest-path rust/Cargo.toml --release --locked --package voiage-ffi`, then `VOIAGE_FFI_LIBRARY=<platform-library> julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` | The fixture is packaged with the Julia source, and CI tests an archived standalone package copy against the separately built native library | Secondary binding; a Julia artifact/JLL is required before standalone General installation |

The paper therefore describes R and Julia as source bindings that share scalar
EVPI, not as semantic equivalents of the Python decision record or as
independently installable registry packages. A JOSS reviewer can assess the
primary Python package without installing those secondary surfaces.

## Selected submission route

The selected route is **pyOpenSci review first, followed by a JOSS partner
fast-track request** if pyOpenSci accepts the package and JOSS confirms that it
is in scope. pyOpenSci reviews the maintained Python package, packaging,
documentation, and usability; the JOSS paper continues to describe the
Rust-centred polyglot project without claiming that every secondary binding is
an independently distributed package.

The project will not enter simultaneous active reviews at pyOpenSci and JOSS.
pyOpenSci acceptance does not guarantee JOSS scope, editorial review, or
acceptance. rOpenSci remains separate and would be relevant only if `voiageR`
became the primary independently installable research package.

The recommended sequence is:

1. publish and archive the exact software revision described by the final
   paper, and retain its hosted test, SBOM, provenance, and digest evidence;
2. document completed research-workflow use, at minimum by the developer, to
   satisfy the demonstrated-use pre-review gate;
3. obtain attributable human community engagement, external use, or
   collaborative input through issue #471 or another documented route; this is
   a detailed-review criterion, a strong positive pre-review signal, and the
   author's selected prerequisite, while automated accounts and AI-agent runs
   do not qualify;
4. wait for and record the permanent arXiv identifier, following the author's
   preferred arXiv-first sequence;
5. after a separate maintainer instruction, open the pyOpenSci inquiry or
   review and complete its external review process;
6. record pyOpenSci acceptance, the accepted revision, and review issue;
7. after another maintainer instruction, request JOSS partner fast-track
   handling and identify the accepted pyOpenSci review;
8. respond to editor and paper-review comments personally and without
   generative AI drafting;
9. after acceptance-ready review, tag the reviewed version and create a
   DOI-bearing Zenodo or Figshare archive requested by JOSS.

## Remaining gates

- Obtain and document qualifying human community engagement, external use, or
  collaborative input in issue #471; do not claim it until attributable
  evidence exists.
- Keep the reviewed release-bound PDF evidence synchronized if `paper.md` or
  its release evidence changes.
- Record the permanent arXiv identifier when arXiv assigns it.
- Perform the selected authenticated pyOpenSci inquiry or submission only
  after a separate maintainer instruction.
- After pyOpenSci acceptance, perform the JOSS partner referral only after a
  separate maintainer instruction.
- Treat editorial screening, review, acceptance and DOI assignment as external
  outcomes.
