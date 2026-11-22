# FAIR research software assessment

Status: **provisional, not accepted**. This record documents review obligations; it is not a FAIR certification or a claim that checks passed.

## Automated evidence

The read-only [howfairis](https://github.com/fair-software/howfairis) action checks the public repository, recognized license, software registry, citation metadata and software quality checklist. Its output is a **five-point indicator**, not verification of scientific results, accessibility or FAIR4RS compliance.

- Run: GitHub Actions > `FAIR research software assessment` (after the workflow is admitted).
- Record the run URL, evaluated default-branch commit and each observed result here when available.
- A failed or unavailable assessment stays **unknown**, never inferred as a pass.
- External action pinned to upstream v0.2.1 commit `4c11146488125aa6e1531184eed51d781bcd5871`; review supply-chain and hosted execution before admission.

## Evidence-based review

| Question | Evidence required | Current disposition |
| --- | --- | --- |
| Public, version-controlled source | Public GitHub URL and relevant release/commit | Not yet reviewed |
| License and third-party rights | Actual license, provenance and dependency/data rights | Not yet reviewed |
| Software registry / persistent identity | Verified registry record and versioned DOI if available | Not yet reviewed |
| Citation and authorship | Verified `CITATION.cff`, authors, software version and publication metadata | Not yet reviewed |
| Software quality checklist | Existing tests/CI, contribution, security and quality documents | Not yet reviewed |
| ARDC FAIR checklist | [Interactive ARDC assessment](https://fairsoftwarechecklist.net/) with answer-level evidence | Not yet reviewed |
| Reproducibility | Exact commit, fixtures, environment, commands, expected/observed outputs | Not yet reviewed |

## Admission rules

1. Prefer an existing repository document, CI control or dependency to adding another tool (the [Ponytail](https://github.com/DietrichGebert/ponytail) reuse-first principle). Do not install Ponytail or duplicate Conductor to obtain a FAIR score.
2. Fix real gaps individually; keep scientific-method, data-rights, security and reproducibility gates independent of FAIR badges.
3. Never manufacture a license grant, DOI, authorship, registry entry, publication or checklist answer. Consult an owner where rights or attribution are unclear.
4. Keep this workflow advisory until its pinned action, GitHub Actions logs and security posture are independently reviewed. PR-triggered runs assess repository-level metadata and may reflect the published/default branch rather than proposed content.
5. Preserve existing Conductor ownership and qualification holds; do not merge solely to improve a score.

References: [fair-software.eu](https://fair-software.eu), [howfairis action](https://github.com/fair-software/howfairis-github-action), [ARDC checklist](https://github.com/ardc-fair-checklist/ardc-fair-checklist.github.io), [Ponytail](https://github.com/DietrichGebert/ponytail).
